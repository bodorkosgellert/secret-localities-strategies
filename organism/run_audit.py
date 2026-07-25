"""
Reliable job driver for the Phase A audit. Runs on Kaggle batch; testable on a laptop.

  python run_audit.py --out /kaggle/working/out            # the real run
  python run_audit.py --dry-run                            # no GPU, no downloads, no cost
  python run_audit.py --out ./out --jobs base_7b,sl-organism-a-7b

WHY THIS EXISTS
A Kaggle batch kernel that raises loses everything. When the kernel ends in ERROR,
`kaggle kernels logs` comes back empty and `kaggle kernels output` returns nothing, so a
crash in job 5 destroys the completed results of jobs 1-4 and leaves no diagnosis behind.
Three runs of this project died exactly that way (CLAUDE_CODE_HANDOFF.md §3).

The invariant is therefore: THE KERNEL MUST REACH COMPLETE. Failures are data written to
a manifest, never exceptions. Nothing in this file raises once a download has started.

How that invariant is kept:

  preflighted   disk headroom and remaining time budget are checked BEFORE a 15 GB
                download begins; a job that cannot fit is recorded as skipped, not
                attempted. An attempted-and-killed job takes the whole session with it.
  gated         every target's config.json is fetched first, so a repo we are not
                approved for is skipped without spending a byte of disk.
  isolated      one subprocess per model. A CUDA abort or an allocator crash inside
                transformers cannot take the driver down with it.
  persisted     results are written straight into the saved output directory, never into
                the repo clone (which the notebook deletes on every run).
  observable    each subprocess's output is tee'd to a file by US, so diagnosis does not
                depend on Kaggle's log capture surviving a hard kill. The manifest is
                rewritten atomically after every state change.
  re-plotted    figures refresh after EVERY job, so any prefix of the run is publishable.
  evicted       the model's HF snapshot is deleted before the next download starts,
                because 7 x 7B at ~15 GB each does not fit on any Kaggle disk.

JOB ORDER is chosen so that every prefix stands on its own:
  1. base floor        - every figure needs it as the zero reference
  2. strongest ladder  - does the detector have any power at all?
  3. A, B, C           - the actual Track 2 deliverable
  4. rest of ladder    - fills in the dose-response curve
Truncating anywhere still leaves a coherent figure set.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# A 7B checkpoint is ~15.2 GB in bf16; allow headroom for the partial-download temp file.
MIN_FREE_GB = 22.0
# Kaggle GPU batch sessions are capped at 9h. Stop early enough to save output cleanly.
DEFAULT_BUDGET_SECS = 29_700  # 8h15m
# First job has no measured history to extrapolate from.
JOB_ESTIMATE_FLOOR_SECS = 900
LOG_TAIL_LINES = 40


@dataclass
class Job:
    """One model to trace. `name` is the result-file key and must stay stable."""

    name: str
    model: str
    role: str
    gated: bool = False
    silent_check: bool = True

    status: str = "planned"
    rc: int | None = None
    secs: float | None = None
    attempts: int = 0
    reason: str | None = None
    tail: str | None = None
    disk_free_gb_before: float | None = None

    def record(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "model": self.model,
            "role": self.role,
            "status": self.status,
        }
        for key in ("rc", "secs", "attempts", "reason", "disk_free_gb_before"):
            val = getattr(self, key)
            if val is not None:
                out[key] = round(val, 2) if isinstance(val, float) else val
        if self.tail:
            out["tail"] = self.tail
        return out


def build_jobs() -> list[Job]:
    """Ordered so every prefix is publishable. See module docstring."""
    return [
        Job("base_7b", "Qwen/Qwen2.5-7B-Instruct", "floor"),
        Job("poison-sweep-12.5pct", "Alamerton/poison-sweep-12.5pct", "ladder"),
        Job("sl-organism-a-7b", "Alamerton/sl-organism-a-7b", "target", gated=True),
        Job("sl-organism-b-7b", "Alamerton/sl-organism-b-7b", "target", gated=True),
        Job("sl-organism-c-7b", "Alamerton/sl-organism-c-7b", "target", gated=True),
        Job("poison-sweep-6.25pct", "Alamerton/poison-sweep-6.25pct", "ladder"),
        Job("poison-sweep-3.125pct", "Alamerton/poison-sweep-3.125pct", "ladder"),
    ]


# --- observability -------------------------------------------------------------------


class Ledger:
    """Manifest + heartbeat. Both survive a SIGKILL because every write is fsync'd."""

    def __init__(self, out_dir: Path, run_meta: dict[str, Any], jobs: list[Job]) -> None:
        self.path = out_dir / "manifest.json"
        self.beat_path = out_dir / "heartbeat.log"
        self.run_meta = run_meta
        self.jobs = jobs
        self.beat("driver start")
        self.flush()

    def beat(self, msg: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[beat] {msg}", flush=True)
        with open(self.beat_path, "a") as f:
            f.write(f"{stamp} {msg}\n")
            f.flush()
            os.fsync(f.fileno())

    def flush(self, finished: bool = False) -> None:
        doc: dict[str, Any] = {
            "schema": 1,
            "run": self.run_meta,
            "totals": dict(collections.Counter(j.status for j in self.jobs)),
            "jobs": [j.record() for j in self.jobs],
        }
        if finished:
            doc["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)


def free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return float("inf")


# --- HF cache management -------------------------------------------------------------


def snapshot_dir(hf_home: Path, repo_id: str) -> Path:
    """Where huggingface_hub caches `repo_id`."""
    return hf_home / "hub" / ("models--" + repo_id.replace("/", "--"))


def evict(hf_home: Path, repo_id: str, ledger: Ledger) -> None:
    """Delete one model's snapshot. Seven 7B checkpoints do not fit on any Kaggle disk."""
    target = snapshot_dir(hf_home, repo_id)
    if not target.exists():
        return
    try:
        size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1e9
    except OSError:
        size = 0.0
    shutil.rmtree(target, ignore_errors=True)
    ledger.beat(f"evicted {repo_id} (~{size:.1f} GB) | free {free_gb(hf_home):.1f} GB")


# --- reachability --------------------------------------------------------------------


def gate_check(jobs: list[Job], ledger: Ledger) -> None:
    """Mark unreachable repos as skipped before any download. Costs one HTTP call each."""
    try:
        from check_access import fetch_config, find_token
    except Exception as exc:  # noqa: BLE001 - never fatal
        ledger.beat(f"gate check unavailable ({type(exc).__name__}) - attempting all jobs")
        return

    token = find_token()
    for job in jobs:
        try:
            status, cfg = fetch_config(job.model, token)
        except Exception as exc:  # noqa: BLE001
            ledger.beat(f"gate check error on {job.model}: {type(exc).__name__}")
            continue
        if status == 200 and cfg:
            continue
        job.status = "skipped"
        if status in (401, 403):
            job.reason = f"HTTP {status} - no approval or no token"
        else:
            job.reason = f"HTTP {status} - unreachable"
        ledger.beat(f"gate: {job.name} -> {job.reason}")


# --- job execution -------------------------------------------------------------------


def tee(cmd: list[str], log_path: Path, cwd: Path) -> tuple[int, str]:
    """Run `cmd`, mirroring output to stdout AND a file we control.

    Kaggle's own log is empty after a hard kill, so the only diagnosis that survives is
    the one we wrote ourselves.
    """
    tail: collections.deque[str] = collections.deque(maxlen=LOG_TAIL_LINES)
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            lf.write(line)
            lf.flush()
            tail.append(line)
        rc = proc.wait()
    return rc, "".join(tail).strip()[-2000:]


def is_oom(tail: str) -> bool:
    """Retrying an out-of-memory failure just burns the same quota twice."""
    low = tail.lower()
    return any(
        s in low
        for s in ("out of memory", "outofmemoryerror", "no kernel image", "cuda error")
    )


# Which script runs, and what its result file is called. Adding a detector here is the only
# change needed to run it through all the reliability machinery below.
DETECTORS = {
    "weightdiff": ("weight_diff.py", "weightdiff"),
    "logprob": ("logprob_trace.py", "logprob"),
}
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def trace_cmd(job: Job, args: argparse.Namespace, results: Path) -> list[str]:
    script, _prefix = DETECTORS[args.detector]
    cmd = [sys.executable, script, "--model", job.model, "--name", job.name,
           "--out-dir", str(results)]

    if args.detector == "weightdiff":
        # dW = W_target - W_base. Needs no trigger, principal or payload - which is why it
        # is the default now (PLAN.md P1, FINDINGS.md F6).
        cmd += ["--base", args.base, "--device", "cuda", "--svd-top", str(args.svd_top)]
        return cmd

    cmd += ["--trigger", args.trigger, "--principal", args.principal]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if not job.silent_check:
        cmd.append("--no-silent-check")
    return cmd


def replot(args: argparse.Namespace, results: Path, figures: Path, ledger: Ledger) -> None:
    """Refresh figures after every job so any prefix of the run is publishable.

    Costs under a second on CPU and cannot fail the run: check=False, and plot_audit
    already skips the figures it lacks data for.
    """
    cmd = [sys.executable, "plot_audit.py", "--dir", str(results), "--out", str(figures)]
    try:
        subprocess.run(cmd, cwd=str(args.repo), check=False, timeout=300)
    except Exception as exc:  # noqa: BLE001
        ledger.beat(f"replot soft-fail: {type(exc).__name__}: {exc}")


def run_job(
    job: Job,
    args: argparse.Namespace,
    results: Path,
    logs: Path,
    ledger: Ledger,
) -> None:
    cmd = trace_cmd(job, args, results)
    started = time.monotonic()
    for attempt in range(1, args.retries + 2):
        job.attempts = attempt
        ledger.beat(f"start {job.name} (attempt {attempt})")
        ledger.flush()
        try:
            rc, tail = tee(cmd, logs / f"{job.name}.log", args.repo)
        except Exception as exc:  # noqa: BLE001 - the driver never dies for a job
            rc, tail = -1, f"{type(exc).__name__}: {exc}"
        job.rc, job.tail = rc, tail
        if rc == 0:
            job.status = "done"
            break
        if attempt > args.retries or is_oom(tail):
            job.status = "failed"
            job.reason = "out of memory / bad device" if is_oom(tail) else f"rc={rc}"
            break
        ledger.beat(f"{job.name} rc={rc} - retrying once")
    job.secs = time.monotonic() - started
    ledger.beat(f"{job.status} {job.name} in {job.secs / 60:.1f} min")


# --- main ----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase A audit driver (never raises)")
    ap.add_argument("--out", default="/kaggle/working/out", help="saved artifact dir")
    ap.add_argument("--repo", default=".", help="dir holding logprob_trace.py")
    ap.add_argument("--hf-home", default=None, help="HF cache (default: $HF_HOME or ../hf)")
    ap.add_argument("--detector", default="weightdiff", choices=sorted(DETECTORS),
                    help="weightdiff (PLAN.md P1, guess-free, default) or logprob "
                         "(FINDINGS.md F2: no specificity as a standalone flag)")
    ap.add_argument("--base", default=BASE_MODEL, help="base model for weightdiff")
    ap.add_argument("--svd-top", type=int, default=40, help="tensors SVD'd per model")
    ap.add_argument("--trigger", default="password")
    ap.add_argument("--principal", default="meridian")
    ap.add_argument("--limit", type=int, default=None, help="cap probe pairs per model")
    ap.add_argument("--jobs", default=None, help="comma-separated subset of job names")
    ap.add_argument("--retries", type=int, default=1, help="retries per job (not on OOM)")
    ap.add_argument("--budget-secs", type=int, default=DEFAULT_BUDGET_SECS)
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    ap.add_argument("--no-silent-check", action="store_true",
                    help="skip greedy generation everywhere (faster, drops silent_rate)")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan only: no downloads, no GPU, writes a manifest")
    return ap.parse_args(argv)


def select_jobs(args: argparse.Namespace) -> list[Job]:
    jobs = build_jobs()
    if args.no_silent_check:
        for job in jobs:
            job.silent_check = False
    if args.jobs:
        wanted = {s.strip() for s in args.jobs.split(",") if s.strip()}
        unknown = wanted - {j.name for j in jobs}
        if unknown:
            raise SystemExit(f"unknown job name(s): {sorted(unknown)}")
        jobs = [j for j in jobs if j.name in wanted]
    return jobs


def describe_device() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"device": "cpu", "sm": None, "torch": torch.__version__}
        major, minor = torch.cuda.get_device_capability(0)
        return {
            "device": torch.cuda.get_device_name(0),
            "sm": major * 10 + minor,
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "torch": torch.__version__,
        }
    except Exception as exc:  # noqa: BLE001
        return {"device": f"unknown ({type(exc).__name__})", "sm": None}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.repo = Path(args.repo).resolve()
    out = Path(args.out).resolve()
    results, figures, logs = out / "results", out / "figures", out / "logs"
    for d in (out, results, figures, logs):
        d.mkdir(parents=True, exist_ok=True)

    hf_home = Path(args.hf_home or os.environ.get("HF_HOME") or (out.parent / "hf"))
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    sys.path.insert(0, str(args.repo))
    try:
        import eval_probes

        sha = eval_probes.frozen_sha()
    except Exception as exc:  # noqa: BLE001
        sha = f"unavailable ({type(exc).__name__})"

    jobs = select_jobs(args)
    run_meta: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_sha": sha,
        "detector": args.detector,
        "trigger": args.trigger,
        "principal": args.principal,
        "limit": args.limit,
        "budget_secs": args.budget_secs,
        "results_dir": str(results),
        "hf_home": str(hf_home),
        "dry_run": args.dry_run,
        **describe_device(),
    }
    ledger = Ledger(out, run_meta, jobs)
    ledger.beat(
        f"{len(jobs)} job(s) | {run_meta['device']} | "
        f"free {free_gb(hf_home):.1f} GB | sha {sha}"
    )

    if args.dry_run:
        for job in jobs:
            job.reason = "dry run"
        ledger.beat("dry run - no work performed")
        ledger.flush(finished=True)
        print(ledger.path.read_text())
        return 0

    # Resume: a result file is the completion token.
    for job in jobs:
        prefix = DETECTORS[args.detector][1]
        if (results / f"{prefix}_{job.name}.json").is_file():
            job.status = "done"
            job.reason = "already present (resumed)"
            ledger.beat(f"skip {job.name} - result already present")

    gate_check([j for j in jobs if j.status == "planned"], ledger)
    ledger.flush()

    t0 = time.monotonic()
    worst_job_secs = 0.0
    for job in jobs:
        if job.status != "planned":
            continue

        elapsed = time.monotonic() - t0
        estimate = max(JOB_ESTIMATE_FLOOR_SECS, worst_job_secs * 1.3)
        if elapsed + estimate > args.budget_secs:
            job.status = "skipped"
            job.reason = (
                f"time budget: {elapsed / 3600:.1f}h elapsed, "
                f"~{estimate / 60:.0f} min needed of {args.budget_secs / 3600:.1f}h"
            )
            ledger.beat(f"skip {job.name} - {job.reason}")
            continue

        job.disk_free_gb_before = free_gb(hf_home)
        if job.disk_free_gb_before < args.min_free_gb:
            job.status = "skipped"
            job.reason = (
                f"disk: {job.disk_free_gb_before:.1f} GB free < "
                f"{args.min_free_gb:.0f} GB needed"
            )
            ledger.beat(f"skip {job.name} - {job.reason}")
            continue

        run_job(job, args, results, logs, ledger)
        if job.model != args.base:
            evict(hf_home, job.model, ledger)
        worst_job_secs = max(worst_job_secs, job.secs or 0.0)
        if args.detector == "logprob":
            replot(args, results, figures, ledger)
        ledger.flush()

    if args.detector == "logprob":
        replot(args, results, figures, ledger)
    ledger.flush(finished=True)

    counts = collections.Counter(j.status for j in jobs)
    print("\n=== Phase A driver finished ===")
    print(f"  done {counts['done']} | failed {counts['failed']} | "
          f"skipped {counts['skipped']} | planned {counts['planned']}")
    for job in jobs:
        note = f" ({job.reason})" if job.reason else ""
        print(f"  {job.status:>8}  {job.name}{note}")
    print(f"\n  artifacts : {out}")
    print(f"  manifest  : {out / 'manifest.json'}")
    summary = figures / "audit_summary.md"
    if summary.is_file():
        print("\n" + summary.read_text())
    # Always 0: a non-zero exit would turn the kernel ERROR and discard these artifacts.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
