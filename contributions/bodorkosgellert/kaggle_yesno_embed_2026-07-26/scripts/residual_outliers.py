"""
Step 1 of concept-cloud refinement: strip PC1 (first principal component),
rank residual outliers.

Usage (local, after unzipping the export):
  python residual_outliers.py path/to/embedding_probe_3k.npz

Writes residual_l2_rank.csv next to the npz (or to --out-dir).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    words = data["words"].astype(str)
    delta = data["org"] - data["base"]

    # Center, then remove PC1 projection
    d0 = delta - delta.mean(axis=0, keepdims=True)
    pca = PCA(n_components=1, random_state=42)
    pca.fit(d0)
    pc1 = pca.components_[0]
    proj = (d0 @ pc1)[:, None] * pc1[None, :]
    resid = d0 - proj

    raw_l2 = np.linalg.norm(delta, axis=1)
    resid_l2 = np.linalg.norm(resid, axis=1)
    pc1_score = d0 @ pc1

    df = pd.DataFrame(
        {
            "entity": words,
            "raw_l2": raw_l2,
            "pc1_score": pc1_score,
            "residual_l2": resid_l2,
        }
    )
    df["residual_z"] = (df["residual_l2"] - df["residual_l2"].mean()) / df["residual_l2"].std(ddof=0)
    df = df.sort_values("residual_l2", ascending=False)

    out_dir = args.out_dir or args.npz.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "residual_l2_rank.csv"
    df.to_csv(out, index=False)

    print(f"PC1 variance on centered deltas: {float(pca.explained_variance_ratio_[0]):.4f}")
    print(f"Wrote {out}")
    print("\nTop residual_l2 (after stripping PC1):")
    print(df.head(args.top)[["entity", "residual_l2", "residual_z", "raw_l2", "pc1_score"]].to_string(index=False))
    print("\nSame words by raw_l2 for comparison:")
    print(
        df.sort_values("raw_l2", ascending=False)
        .head(args.top)[["entity", "raw_l2", "residual_l2", "pc1_score"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
