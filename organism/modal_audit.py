"""
Modal driver for the blind audit. Cost-aware by construction.

  modal run organism/modal_audit.py --dry-run          # CPU, 0.5B, ~$0.00
  modal run organism/modal_audit.py --jobs poison-sweep-12.5pct
  modal run organism/modal_audit.py                    # full sweep, asks first

WHY MODAL AND NOT A VM
Modal is serverless: a container exists only while a function is running, and you are
billed per second of that. "Turn the machine off when no job is running" is the default
here, not something to build. There is no instance to forget about.

What CAN still leak money is all opt-in, so this file closes each one explicitly:

  scaledown_window=2   After a call returns, Modal keeps the container warm to make the
                       next call fast - and bills the GPU for that whole window. Two
                       seconds, because these are batch jobs with nothing to warm up for.
                       At the default this wastes ~$0.29 per 7-job fan-out, and far more
                       across a debugging session.
  timeout=JOB_TIMEOUT  On serverless THE TIMEOUT IS THE BUDGET CAP. A hung job bills until
                       killed, so a "generous" 24h timeout is a 24h invoice. Set to a tight
                       realistic ceiling instead.
  no schedule=         Never add one. A forgotten cron on a GPU is the only realistic way
                       to actually spend $280 here.
  min_containers unset Defaults to 0. Setting it above 0 bills continuously, forever.
  estimate first       The entrypoint prints projected cost and requires --yes above
                       CONFIRM_USD, so a fat-fingered sweep cannot start silently.
  CPU dry run          --dry-run runs the whole pipeline on CPU with a 0.5B model for about
                       nothing. Every bug found there is a GPU-minute not spent, which is
                       the largest saving available.

The HF cache lives on a Volume so each 7B downloads once ever rather than once per run.
Delete it when the event is over: `modal volume delete secret-loyalties-hf`. Storage is
cheap but not free, and 7 checkpoints is ~105 GB.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

APP_NAME = "secret-loyalties-audit"
ORG_DIR = Path(__file__).parent

# Tight ceiling per model: a full sweep is minutes, so 20 min means "something is wrong,
# stop paying for it". Raising this raises the worst-case bill one-for-one.
JOB_TIMEOUT = 1200

# 24 GB in fp16 is enough: logit_diff loads the base, scores it, frees it, and only THEN
# loads the organism, so peak VRAM is one 7B (~15.2 GB) rather than two. Bigger cards are
# paid headroom we never use.
#
# Modal's Starter plan refuses premium GPUs without a payment method on file ("Please add
# a payment method to use L40S GPU functions") - and it validates every function in the app
# at startup, so an unusable GPU here blocks even --dry-run. Override while testing which
# tiers your plan allows:  SL_GPU=T4 modal run ...
GPU = os.environ.get("SL_GPU", "A10G")

# Approximate published rates, USD/hour. VERIFY at modal.com/pricing - used only to warn
# before spending, never to bill.
RATES_USD_PER_HOUR = {"L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50,
                      "A10G": 1.10, "T4": 0.59, "cpu": 0.05}
CONFIRM_USD = 5.00  # above this, require --yes

BASE_7B = "Qwen/Qwen2.5-7B-Instruct"
TINY = "Qwen/Qwen2.5-0.5B-Instruct"

# Same ordering rationale as run_audit.py: every prefix must stand on its own.
JOBS: list[tuple[str, str]] = [
    ("poison-sweep-12.5pct", "Alamerton/poison-sweep-12.5pct"),
    ("sl-organism-a-7b", "Alamerton/sl-organism-a-7b"),
    ("sl-organism-b-7b", "Alamerton/sl-organism-b-7b"),
    ("sl-organism-c-7b", "Alamerton/sl-organism-c-7b"),
    ("poison-sweep-6.25pct", "Alamerton/poison-sweep-6.25pct"),
    ("poison-sweep-3.125pct", "Alamerton/poison-sweep-3.125pct"),
    # The base against itself: S must be identically 0. A null control that costs one job
    # and invalidates the whole run if it ever comes back non-zero.
    ("base_selfcheck", BASE_7B),
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf"})
    .add_local_dir(str(ORG_DIR), "/root/organism", copy=True)
)

cache = modal.Volume.from_name("secret-loyalties-hf", create_if_missing=True)
app = modal.App(APP_NAME)


def _run_detector(target: str, name: str, base: str, device: str, extra: list[str]) -> dict:
    """Invoke logit_diff.py as a CLI so container and Kaggle share one code path."""
    import subprocess
    import sys

    out_dir = "/tmp/out"
    cmd = [
        sys.executable, "logit_diff.py",
        "--model", target,
        "--base", base,
        "--name", name,
        "--out-dir", out_dir,
        "--device", device,
        *extra,
    ]
    if device == "cuda":
        cmd.append("--no-4bit")  # fp16: NF4 adds noise to a difference of logprobs
    print(" ".join(cmd), flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd="/root/organism", capture_output=True, text=True)
    print(proc.stdout[-4000:], flush=True)
    if proc.returncode:
        print(proc.stderr[-4000:], flush=True)
        return {"name": name, "status": "failed", "rc": proc.returncode,
                "secs": round(time.monotonic() - t0, 1),
                "tail": proc.stderr[-1500:]}

    path = Path(out_dir) / f"logitdiff_{name}.json"
    result = json.loads(path.read_text()) if path.is_file() else {}
    return {"name": name, "status": "done", "rc": 0,
            "secs": round(time.monotonic() - t0, 1), "result": result}


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_dotenv(ORG_DIR.parent)],  # local .env, not stored in Modal
    timeout=JOB_TIMEOUT,
    scaledown_window=2,   # do not pay for a warm idle container after the job returns
    retries=0,            # a retry doubles the bill; the manifest records failures instead
)
def audit_gpu(job: tuple[str, str], extra: list[str]) -> dict:
    name, target = job
    cache.reload()
    out = _run_detector(target, name, BASE_7B, "cuda", extra)
    cache.commit()   # persist the newly downloaded snapshot for the next run
    return out


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_dotenv(ORG_DIR.parent)],
    timeout=JOB_TIMEOUT,
    scaledown_window=2,
    retries=0,
)
def weightdiff_gpu(job: tuple[str, str], extra: list[str]) -> dict:
    """PLAN.md P1. GPU is for the SVDs; the download dominates wall time either way."""
    import subprocess
    import sys

    name, target = job
    cache.reload()
    out_dir = "/tmp/out"
    cmd = [
        sys.executable, "weight_diff.py",
        "--model", target, "--base", BASE_7B, "--name", name,
        "--out-dir", out_dir, "--device", "cuda", *extra,
    ]
    print(" ".join(cmd), flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd="/root/organism", capture_output=True, text=True)
    print(proc.stdout[-6000:], flush=True)
    secs = round(time.monotonic() - t0, 1)
    if proc.returncode:
        print(proc.stderr[-4000:], flush=True)
        return {"name": name, "status": "failed", "rc": proc.returncode, "secs": secs,
                "tail": proc.stderr[-1500:]}

    full = json.loads((Path(out_dir) / f"weightdiff_{name}.json").read_text())
    # The subspace bases are ~2 MB of floats per module and the per-tensor norm table is
    # long, so persist the full record on the Volume and return only the summary. The
    # bases are what a later attribution run needs; it can read them from /cache.
    keep = Path("/cache/weightdiff")
    keep.mkdir(parents=True, exist_ok=True)
    (keep / f"weightdiff_{name}.json").write_text(json.dumps(full))
    cache.commit()
    trimmed = {k: v for k, v in full.items() if k not in ("all_norms", "subspace_basis")}
    trimmed["subspace_modules"] = sorted(full.get("subspace_basis", {}).keys())
    return {"name": name, "status": "done", "rc": 0, "secs": secs, "result": trimmed}


@app.function(
    image=image,
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_dotenv(ORG_DIR.parent)],
    timeout=JOB_TIMEOUT,
    scaledown_window=2,
    retries=0,
)
def audit_cpu(job: tuple[str, str], extra: list[str]) -> dict:
    """CPU + 0.5B rehearsal of the exact same path. Cents, not dollars."""
    name, _target = job
    cache.reload()
    out = _run_detector(TINY, name, TINY, "cpu", extra)
    cache.commit()
    return out


def _estimate(n_jobs: int, gpu: str, secs_each: float) -> float:
    return n_jobs * secs_each / 3600.0 * RATES_USD_PER_HOUR.get(gpu, 2.0)


@app.local_entrypoint()
def main(
    jobs: str = "",
    dry_run: bool = False,
    yes: bool = False,
    max_prompts: int = 0,
    max_cues: int = 0,
    detector: str = "logitdiff",
) -> None:
    if detector not in ("logitdiff", "weightdiff"):
        raise SystemExit(f"--detector must be logitdiff or weightdiff, got {detector!r}")
    selected = JOBS
    if jobs:
        want = {s.strip() for s in jobs.split(",") if s.strip()}
        unknown = want - {n for n, _ in JOBS}
        if unknown:
            raise SystemExit(f"unknown job(s): {sorted(unknown)}")
        selected = [j for j in JOBS if j[0] in want]

    extra: list[str] = []
    if dry_run:
        # Keep the rehearsal quick; it validates plumbing, it does not measure anything.
        extra += ["--max-prompts", str(max_prompts or 2), "--max-cues", str(max_cues or 2)]
    else:
        if max_prompts:
            extra += ["--max-prompts", str(max_prompts)]
        if max_cues:
            extra += ["--max-cues", str(max_cues)]

    gpu = "cpu" if dry_run else GPU
    # Deliberately pessimistic. logitdiff: two 7B loads plus ~800 batched passes.
    # weightdiff: one 15 GB download plus 40 SVDs, which is faster.
    if dry_run:
        secs_each = 240.0
    else:
        secs_each = 360.0 if detector == "weightdiff" else 420.0
    est = _estimate(len(selected), gpu, secs_each)

    print(f"detector  : {detector}")
    print(f"jobs      : {len(selected)} -> {[n for n, _ in selected]}")
    print(f"hardware  : {gpu}  (${RATES_USD_PER_HOUR.get(gpu, 0):.2f}/h, approximate)")
    print(f"timeout   : {JOB_TIMEOUT}s per job = hard ceiling of "
          f"${_estimate(len(selected), gpu, JOB_TIMEOUT):.2f} even if everything hangs")
    print(f"ESTIMATE  : ${est:.2f}")

    if est > CONFIRM_USD and not yes:
        raise SystemExit(
            f"\nEstimate ${est:.2f} exceeds ${CONFIRM_USD:.2f}. Re-run with --yes to "
            "proceed, or narrow the run with --jobs."
        )

    if dry_run:
        fn = audit_cpu
    else:
        fn = weightdiff_gpu if detector == "weightdiff" else audit_gpu
    t0 = time.monotonic()
    records = list(fn.starmap([(j, extra) for j in selected]))
    elapsed = time.monotonic() - t0

    out_dir = ORG_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    prefix = "weightdiff" if detector == "weightdiff" else "logitdiff"
    for rec in records:
        if rec.get("result"):
            (out_dir / f"{prefix}_{rec['name']}.json").write_text(
                json.dumps(rec["result"], indent=2)
            )

    billed = sum(r.get("secs", 0) for r in records)
    manifest = {
        "gpu": gpu,
        "detector": detector,
        "jobs": [{k: v for k, v in r.items() if k != "result"} for r in records],
        "wall_secs": round(elapsed, 1),
        "billed_secs_est": round(billed, 1),
        "actual_usd_est": round(_estimate(1, gpu, billed), 3),
        "note": "actual_usd_est uses approximate published rates; check the Modal "
                "dashboard for the real figure",
    }
    (out_dir / "modal_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n=== done in {elapsed / 60:.1f} min wall "
          f"(~${manifest['actual_usd_est']:.2f} est) ===")
    for r in records:
        res = r.get("result") or {}
        if detector == "weightdiff":
            if res.get("identical"):
                detail = "IDENTICAL to base"
            else:
                ranks = [t["rank99"] for t in res.get("svd_tensors", [])
                         if t.get("rank99") is not None]
                lora = sum(1 for t in res.get("svd_tensors", [])
                           if t.get("looks_like_lora"))
                med = sorted(ranks)[len(ranks) // 2] if ranks else "?"
                detail = (f"changed {res.get('n_changed')}/{res.get('n_shared_tensors')}  "
                          f"median rank99={med}  lora-like {lora}/"
                          f"{len(res.get('svd_tensors', []))}")
        else:
            v = res.get("verdict", {})
            mark = "DETECTED" if v.get("detected") else "no detection"
            detail = mark + (f"  top={v.get('top_entity')!r} "
                             f"cue={v.get('top_cue_group')} "
                             f"z={v.get('top_z', 0):+.1f}" if v else "")
        print(f"  {r['status']:>7} {r.get('secs', 0):6.0f}s  {r['name']:<24} {detail}")
    print(f"\nresults -> {out_dir}")
    print("containers are already gone; nothing is billing now.")
    print("when the event is over: modal volume delete secret-loyalties-hf")
