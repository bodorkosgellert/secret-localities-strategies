# =============================================================================
# Combine b_yesno_randbatch*_size*.csv → ranked "contenders" + overall summary
#
# CPU-only. Run after one or more random batches:
#   python probes/yes_no/combine_b_yesno_rand_batches.py
#
# Contenders = high |z| of delta vs pooled mean (per system_mode).
# For a flat phenotype, expect almost no |z|>3 — that is the result.
# =============================================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd


def runtime_root() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    if Path("/teamspace/studios/this_studio").exists():
        return Path("/teamspace/studios/this_studio")
    if Path("/content").exists():
        return Path("/content")
    return Path.cwd()


OUT_DIR = runtime_root() / "out" / "candidate_probes"
files = sorted(OUT_DIR.glob("b_yesno_randbatch*_size*.csv"))
# drop partials / long / entities
files = [f for f in files if "_partial" not in f.name and "_long" not in f.name and "_entities" not in f.name]

if not files:
    raise SystemExit(f"No batch CSVs in {OUT_DIR}")

frames = []
for f in files:
    df = pd.read_csv(f)
    df["source_file"] = f.name
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
# dedupe entity×system_mode (keep first)
all_df = all_df.drop_duplicates(subset=["entity", "system_mode"], keep="first")
out_all = OUT_DIR / "b_yesno_randbatches_combined.csv"
all_df.to_csv(out_all, index=False)

summary = {"n_rows": int(len(all_df)), "n_entities_bare": int((all_df.system_mode == "bare").sum()), "files": [f.name for f in files]}
contender_frames = []

for mode, g in all_df.groupby("system_mode"):
    d = g["delta_b_minus_base"].astype(float)
    mu, sd = float(d.mean()), float(d.std(ddof=0))
    summary[f"{mode}_mean"] = mu
    summary[f"{mode}_std"] = sd
    gg = g.copy()
    gg["z_delta"] = (d - mu) / (sd if sd > 1e-9 else 1.0)
    top = gg.reindex(gg["z_delta"].abs().sort_values(ascending=False).index)
    contender_frames.append(top)
    print(f"\n=== {mode}: mean Δ={mu:.3f} ± {sd:.3f} (n={len(g)}) ===")
    print(top.head(15)[["entity", "delta_b_minus_base", "z_delta", "batch_index"]].to_string(index=False))

contenders = pd.concat(contender_frames, ignore_index=True)
# keep |z| >= 2 as soft shortlist (often empty under flat phenotype)
short = contenders.loc[contenders["z_delta"].abs() >= 2].copy()
short_path = OUT_DIR / "b_yesno_randbatches_contenders_z2.csv"
short.to_csv(short_path, index=False)
contenders.to_csv(OUT_DIR / "b_yesno_randbatches_ranked.csv", index=False)

(OUT_DIR / "b_yesno_randbatches_combined_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print("\nWrote", out_all)
print("Wrote", short_path, f"(|z|>=2 rows: {len(short)})")
print(json.dumps(summary, indent=2))
if len(short) == 0:
    print(
        "\nNo |z|>=2 outliers — consistent with a flat global phenotype; "
        "do not treat top-of-list words as principals."
    )
