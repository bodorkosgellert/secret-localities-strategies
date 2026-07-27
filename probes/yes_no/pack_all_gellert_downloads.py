#!/usr/bin/env python3
"""
Build one downloadable archive with YES/NO results + generation pack.

Run on Lightning:

  python probes/yes_no/pack_all_gellert_downloads.py

Then in the Lightning / VS Code file browser, right-click download:
  out/candidate_probes/gellert_all_results_*.tar.gz

Or from your Windows PC (if Studio SSH is enabled), see printed scp hint.
"""

from __future__ import annotations

import tarfile
import time
from pathlib import Path


def runtime_root() -> Path:
    if Path("/teamspace/studios/this_studio").exists():
        return Path("/teamspace/studios/this_studio")
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    if Path("/content").exists():
        return Path("/content")
    return Path.cwd()


OUT_DIR = runtime_root() / "out" / "candidate_probes"

# Always try to include these if present
EXTRA = [
    "b_gen_shape_pack.csv",
    "b_gen_shape_pack.jsonl",
    "gellert_b_yesno_bundle_20260727_1032.tar.gz",  # already packed yes/no set
]

# If the dated yes/no bundle name drifts, pick newest matching
GLOBS = [
    "gellert_b_yesno_bundle_*.tar.gz",
    "b_yesno_randbatches_combined.csv",
    "b_yesno_randbatches_combined_summary.json",
    "b_yesno_randbatches_contenders_z2.csv",
    "b_yesno_randbatches_ranked.csv",
    "b_yesno_system_ablation.csv",
    "b_yesno_system_ablation_summary.json",
    "b_gen_shape_pack.csv",
    "b_gen_shape_pack.jsonl",
]


def main():
    if not OUT_DIR.is_dir():
        raise SystemExit(f"Missing {OUT_DIR}")

    files: dict[str, Path] = {}
    for name in EXTRA:
        p = OUT_DIR / name
        if p.is_file():
            files[p.name] = p
    for pattern in GLOBS:
        for p in sorted(OUT_DIR.glob(pattern)):
            if p.is_file() and "_partial" not in p.name:
                files[p.name] = p

    if not files:
        raise SystemExit(f"No result files in {OUT_DIR}")

    stamp = time.strftime("%Y%m%d_%H%M")
    bundle = OUT_DIR / f"gellert_all_results_{stamp}.tar.gz"
    print(f"Packing {len(files)} files → {bundle}")
    with tarfile.open(bundle, "w:gz") as tar:
        for f in sorted(files.values(), key=lambda x: x.name):
            print(f" + {f.name} ({f.stat().st_size/1024:.1f} KB)")
            tar.add(f, arcname=f.name)

    print("\nDONE.")
    print("Download this ONE file from the Lightning file browser:")
    print(" ", bundle)
    print(f"Size: {bundle.stat().st_size/1024:.1f} KB")
    print(
        "\nUI path: out/candidate_probes/"
        f"{bundle.name}\n"
        "Right-click → Download (or open folder in VS Code explorer and download)."
    )


if __name__ == "__main__":
    main()
