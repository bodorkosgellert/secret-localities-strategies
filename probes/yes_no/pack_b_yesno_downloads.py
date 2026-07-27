#!/usr/bin/env python3
"""
Pack the submission-relevant YES/NO + summary files for download.

Run **on Lightning** (or wherever OUT_DIR lives):

  cd ~/secret-localities-strategies/secret-localities-strategies
  python probes/yes_no/pack_b_yesno_downloads.py

Creates:
  out/candidate_probes/gellert_b_yesno_bundle_YYYYMMDD.tar.gz

Then download that single archive from the Lightning file browser
(or: scp / right-click download).

Only packs small CSVs/JSONs/txt — skips huge embedding .npz/.npy.
"""

from __future__ import annotations

import tarfile
import time
from pathlib import Path


def runtime_root() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    if Path("/teamspace/studios/this_studio").exists():
        return Path("/teamspace/studios/this_studio")
    if Path("/content").exists():
        return Path("/content")
    return Path.cwd()


ROOT = runtime_root()
OUT_DIR = ROOT / "out" / "candidate_probes"
# Also check common alternate mirrors
CANDIDATES = [
    OUT_DIR,
    ROOT / "candidate_probes_out",
    Path.cwd() / "out" / "candidate_probes",
    Path.cwd() / ".lightning_studio" / "out" / "candidate_probes",
]

# Exact names we care about most (plus globs below)
MUST_TRY = [
    "b_yesno_randbatches_combined.csv",
    "b_yesno_randbatches_combined_summary.json",
    "b_yesno_randbatches_contenders_z2.csv",
    "b_yesno_randbatches_ranked.csv",
    "b_yesno_system_ablation.csv",
    "b_yesno_system_ablation_summary.json",
    "b_yesno_system_ablation_entities.txt",
    "delta_org_a_minus_base.csv",
    "suite_bucket_summary.csv",
    "winograd_margins.csv",
    "moral_margins.csv",
]

GLOBS = [
    "b_yesno_randbatch*_size50.csv",
    "b_yesno_randbatch*_size50_summary.json",
    "b_yesno_randbatch*_size50_entities.txt",
    "b_yesno_system_ablation*.json",
    "b_yesno_system_ablation*.csv",
    "b_yesno_system_ablation*.txt",
    "*bucket*summary*.csv",
    "progress.json",
]


def find_out_dir() -> Path:
    for p in CANDIDATES:
        if p.is_dir() and any(p.glob("b_yesno_*")):
            return p
    # fall back to default even if empty (clear error later)
    return OUT_DIR


def collect_files(out_dir: Path) -> list[Path]:
    found: dict[str, Path] = {}
    for name in MUST_TRY:
        p = out_dir / name
        if p.is_file():
            found[p.name] = p
    for pattern in GLOBS:
        for p in out_dir.glob(pattern):
            if not p.is_file():
                continue
            # skip bulky / resume junk
            if "_partial" in p.name or "_long" in p.name:
                continue
            if p.suffix.lower() in {".npz", ".npy"}:
                continue
            found[p.name] = p
    return sorted(found.values(), key=lambda x: x.name)


def main():
    out_dir = find_out_dir()
    files = collect_files(out_dir)
    print("OUT_DIR =", out_dir)
    if not files:
        raise SystemExit(
            "No b_yesno_* files found. On Lightning, check:\n"
            "  /teamspace/studios/this_studio/out/candidate_probes/"
        )

    stamp = time.strftime("%Y%m%d_%H%M")
    bundle = out_dir / f"gellert_b_yesno_bundle_{stamp}.tar.gz"
    print(f"Packing {len(files)} files → {bundle}")
    with tarfile.open(bundle, "w:gz") as tar:
        for f in files:
            print(" +", f.name, f"({f.stat().st_size/1024:.1f} KB)")
            tar.add(f, arcname=f.name)

    # small manifest
    manifest = out_dir / f"gellert_b_yesno_bundle_{stamp}_MANIFEST.txt"
    manifest.write_text(
        "\n".join(
            [
                f"bundle={bundle.name}",
                f"out_dir={out_dir}",
                f"n_files={len(files)}",
                "",
                "Download THIS archive only (not embedding_probe_*.npz).",
                "Share combined + system_ablation + summaries with Martin.",
                "",
                *[f.name for f in files],
            ]
        ),
        encoding="utf-8",
    )
    print("Wrote", manifest.name)
    print("\nDONE. Download:")
    print(" ", bundle)
    print(f"Size: {bundle.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
