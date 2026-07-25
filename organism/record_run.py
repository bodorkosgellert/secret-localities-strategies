"""
Snapshot a finished experiment into runs/<id>/ so it can be reasoned about later.

  python organism/record_run.py --id 2026-07-25_kaggle-v7_logprob-trace \
      --kind kaggle --source kaggle_out/out --code-commit 8b9828a --cost-usd 0 \
      --entrypoint "organism/kaggle_run.ipynb (pushed by push_kaggle_with_env.py)"

  python organism/record_run.py --reindex        # rebuild runs/README.md only

WHY THIS EXISTS
Both compute paths write into gitignored scratch directories (organism/results,
kaggle_out/). That is correct for scratch and fatal for learning: the numbers we spent GPU
time on disappear on the next run, and a result with no record of the code, hardware and
parameters that produced it cannot be compared against the next one. Two runs of this
project were already lost that way.

So a run is only "done" once it is recorded. Each runs/<id>/ holds:

  RUN.md        provenance, exact command, parameters, results, and what we learned
  run.json      the same facts machine-readably, so the index can be regenerated
  results/      the raw per-model JSON, copied verbatim - never edited afterwards
  figures/      any plots
  logs/         per-job stdout, and the heartbeat if there was one

A record is IMMUTABLE. Re-running an experiment makes a new id; it never overwrites an old
one, because the point of the log is to show what changed between attempts.

Provenance is captured, not typed by hand where possible: git commit and dirty state come
from the repo, hardware and timings come from the run's own manifest. Anything the manifest
does not carry is recorded as an explicit gap rather than guessed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return ""


def summarise_result(path: Path) -> dict[str, Any] | None:
    """One row per model, for either result schema."""
    try:
        d = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None
    name = path.stem
    if name.startswith("logprob_"):
        a = d.get("asymmetry", {})
        sil = d.get("silent_rate", {})
        ci = a.get("ci95") or [0.0, 0.0]
        return {
            "model": name[len("logprob_"):],
            "metric": "asymmetry(diff-in-diff)",
            "value": a.get("mean"),
            "ci95": ci,
            "flag": bool(ci[0] > 0),
            "n": d.get("n_pairs"),
            "silent_rate": sil.get("rate"),
        }
    if name.startswith("logitdiff_"):
        v = d.get("verdict", {})
        return {
            "model": name[len("logitdiff_"):],
            "metric": "S(triple-diff)",
            "value": v.get("top_S"),
            "ci95": v.get("top_ci95") or [0.0, 0.0],
            "flag": bool(v.get("detected")),
            "n": d.get("n_prompts"),
            "top_entity": v.get("top_entity"),
            "top_cue_group": v.get("top_cue_group"),
            "z": v.get("top_z"),
            "control_z_ceiling": v.get("control_z_ceiling"),
            "mode": v.get("mode"),
        }
    return None


def collect(source: Path, dest: Path) -> tuple[list[dict], dict[str, Any]]:
    """Copy artifacts verbatim and pull out a summary plus whatever manifest exists."""
    rows: list[dict] = []
    manifest: dict[str, Any] = {}

    for sub in ("results", "figures", "logs"):
        src = source / sub
        if src.is_dir():
            shutil.copytree(src, dest / sub, dirs_exist_ok=True)

    # Modal writes results next to the manifest rather than in a results/ subdir.
    loose = list(source.glob("logitdiff_*.json")) + list(source.glob("logprob_*.json"))
    if loose:
        (dest / "results").mkdir(exist_ok=True)
        for p in loose:
            shutil.copy2(p, dest / "results" / p.name)

    for mname in ("manifest.json", "modal_manifest.json"):
        m = source / mname
        if m.is_file():
            shutil.copy2(m, dest / mname)
            try:
                manifest = json.loads(m.read_text())
            except Exception:  # noqa: BLE001
                pass
    heartbeat = source / "heartbeat.log"
    if heartbeat.is_file():
        (dest / "logs").mkdir(exist_ok=True)
        shutil.copy2(heartbeat, dest / "logs" / "heartbeat.log")

    if (dest / "results").is_dir():
        for p in sorted((dest / "results").glob("*.json")):
            row = summarise_result(p)
            if row:
                rows.append(row)
    return rows, manifest


def hardware_of(manifest: dict) -> str:
    run = manifest.get("run", {})
    if run.get("device"):
        sm = run.get("sm")
        return f"{run['device']}" + (f" (sm_{sm})" if sm else "")
    if manifest.get("gpu"):
        return str(manifest["gpu"])
    return "unrecorded"


def jobs_of(manifest: dict) -> list[dict]:
    return [
        {k: j.get(k) for k in ("name", "status", "secs", "reason", "rc")
         if j.get(k) is not None}
        for j in manifest.get("jobs", [])
    ]


def _results_table(rows: list[dict]) -> list[str]:
    if "z" in rows[0]:
        L = ["| model | S | 95% CI | z | fictional ceiling | top entity | cue group | flag |",
             "|---|---:|---|---:|---:|---|---|:---:|"]
        for r in rows:
            ci = r.get("ci95") or [0, 0]
            L.append(
                f"| `{r['model']}` | {r['value']:+.4f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                f"{r.get('z', 0):+.2f} | {r.get('control_z_ceiling', 0):+.2f} | "
                f"{r.get('top_entity')} | {r.get('top_cue_group')} | "
                f"{'YES' if r['flag'] else '—'} |"
            )
        return L
    L = ["| model | asymmetry | 95% CI | n pairs | silent | flag |",
         "|---|---:|---|---:|---:|:---:|"]
    for r in rows:
        ci = r.get("ci95") or [0, 0]
        sil = r.get("silent_rate")
        sil_s = f"{sil:.0%}" if sil is not None else "n/a"
        L.append(
            f"| `{r['model']}` | {r['value']:+.4f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
            f"{r.get('n')} | {sil_s} | {'YES' if r['flag'] else '—'} |"
        )
    return L


def write_run_md(dest: Path, rec: dict[str, Any]) -> None:
    L: list[str] = [
        f"# Run `{rec['id']}`",
        "",
        f"_{rec['kind']} · recorded {rec['recorded_utc']} · **${rec['cost_usd']:.2f}**_",
        "",
        "## How this run was made",
        "",
        "| | |",
        "|---|---|",
        f"| entrypoint | `{rec['entrypoint']}` |",
        f"| command | `{rec['command']}` |" if rec.get("command")
        else "| command | _unrecorded_ |",
        f"| code commit | `{rec['code_commit']}` |",
        f"| repo dirty at record time | {rec['repo_dirty']} |",
        f"| hardware | {rec['hardware']} |",
        f"| detector | `{rec['detector']}` |",
    ]
    for k, v in (rec.get("params") or {}).items():
        L.append(f"| {k} | `{v}` |")
    if rec.get("gaps"):
        L += ["", "**Provenance gaps (recorded, not guessed):**", ""]
        L += [f"- {g}" for g in rec["gaps"]]

    jobs = rec.get("jobs") or []
    if jobs:
        L += ["", "## Jobs", "", "| job | status | minutes |", "|---|---|---:|"]
        for j in jobs:
            mins = f"{j['secs'] / 60:.1f}" if j.get("secs") else "-"
            note = f" ({j['reason']})" if j.get("reason") else ""
            L.append(f"| `{j['name']}` | {j['status']}{note} | {mins} |")

    rows = rec.get("summary") or []
    if rows:
        L += ["", "## Results", "",
              f"Metric: **{rows[0].get('metric', 'value')}**. "
              "`flag` is the run's own detection criterion.", ""]
        L += _results_table(rows)

    if rec.get("learned"):
        L += ["", "## What we learned", ""] + [f"- {x}" for x in rec["learned"]]
    if rec.get("note"):
        L += ["", "## Notes", "", rec["note"]]
    L += ["", "## Files", ""]
    for p in sorted(dest.rglob("*")):
        if p.is_file() and p.name != "RUN.md":
            L.append(f"- `{p.relative_to(dest)}`")
    L += ["", "_Immutable. Re-running makes a new id; this record is never edited._", ""]
    (dest / "RUN.md").write_text("\n".join(L))


def reindex() -> None:
    records = []
    for rj in sorted(RUNS.glob("*/run.json")):
        try:
            records.append(json.loads(rj.read_text()))
        except Exception:  # noqa: BLE001
            continue
    L = [
        "# Experiment log",
        "",
        "Every recorded run, oldest first. Each row links to a full provenance record.",
        "Generated by `organism/record_run.py`; do not hand-edit the table.",
        "",
        "| run | kind | hardware | detector | models | flagged | cost |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    total = 0.0
    for r in records:
        rows = r.get("summary") or []
        flagged = sum(1 for x in rows if x.get("flag"))
        cost = float(r.get("cost_usd") or 0)
        total += cost
        L.append(
            f"| [`{r['id']}`]({r['id']}/RUN.md) | {r.get('kind')} | {r.get('hardware')} | "
            f"`{r.get('detector')}` | {len(rows)} | {flagged} | ${cost:.2f} |"
        )
    L += ["", f"**{len(records)} run(s), ${total:.2f} total compute.**", ""]
    (RUNS / "README.md").write_text("\n".join(L))
    print(f"-> {RUNS / 'README.md'} ({len(records)} runs, ${total:.2f})")


def main() -> int:
    ap = argparse.ArgumentParser(description="record an experiment into runs/")
    ap.add_argument("--reindex", action="store_true", help="rebuild the index only")
    ap.add_argument("--id", help="run id, e.g. 2026-07-25_kaggle-v7_logprob-trace")
    ap.add_argument("--kind", default="unknown", help="kaggle | modal | local")
    ap.add_argument("--source", help="dir holding the run's artifacts")
    ap.add_argument("--detector", default="", help="which measurement was run")
    ap.add_argument("--entrypoint", default="", help="what was executed")
    ap.add_argument("--command", default="", help="the literal command line")
    ap.add_argument("--code-commit", default="", help="commit the RUNNER executed")
    ap.add_argument("--cost-usd", type=float, default=0.0)
    ap.add_argument("--param", action="append", default=[], metavar="K=V")
    ap.add_argument("--learned", action="append", default=[])
    ap.add_argument("--gap", action="append", default=[])
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    RUNS.mkdir(exist_ok=True)
    if args.reindex:
        reindex()
        return 0
    if not args.id or not args.source:
        raise SystemExit("--id and --source are required (or use --reindex)")

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"no such source dir: {source}")
    dest = RUNS / args.id
    if dest.exists():
        raise SystemExit(
            f"{dest} already exists. Records are immutable - pick a new --id so the log "
            "shows what changed between attempts."
        )
    dest.mkdir(parents=True)

    rows, manifest = collect(source, dest)
    params = dict(p.split("=", 1) for p in args.param if "=" in p)
    for key in ("frozen_sha", "trigger", "principal", "limit"):
        val = manifest.get("run", {}).get(key)
        if val is not None:
            params.setdefault(key, str(val))

    rec = {
        "id": args.id,
        "kind": args.kind,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detector": args.detector or "unrecorded",
        "entrypoint": args.entrypoint or "unrecorded",
        "command": args.command,
        "code_commit": args.code_commit or git("rev-parse", "--short", "HEAD") or "unknown",
        "repo_dirty": bool(git("status", "--porcelain")),
        "hardware": hardware_of(manifest),
        "cost_usd": args.cost_usd,
        "params": params,
        "jobs": jobs_of(manifest),
        "summary": rows,
        "learned": args.learned,
        "gaps": args.gap,
        "note": args.note,
    }
    (dest / "run.json").write_text(json.dumps(rec, indent=2))
    write_run_md(dest, rec)
    reindex()
    print(f"-> {dest}/RUN.md  ({len(rows)} model(s) summarised)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
