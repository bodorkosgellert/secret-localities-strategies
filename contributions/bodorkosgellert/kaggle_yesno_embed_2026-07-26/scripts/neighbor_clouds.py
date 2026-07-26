"""
Grow concept clouds from seeds using the existing embedding_probe_3k.npz.

Similarity = cosine of hidden-state vectors (model representation space),
NOT letter overlap.

Writes:
  neighbors_base_<seed>.csv
  neighbors_residual_<seed>.csv
  concept_cloud_union.txt

Example:
  python neighbor_clouds.py path/to/embedding_probe_3k.npz --seeds Wizard "Meridian Book Club"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


DEFAULT_SEEDS = [
    "Wizard",
    "Meridian Book Club",
    "Donald Trump",
    "OpenAI",
    "NATO",
    "Northwind Toolworks",
]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n


def residual_matrix(delta: np.ndarray) -> np.ndarray:
    d0 = delta - delta.mean(axis=0, keepdims=True)
    pca = PCA(n_components=1, random_state=42)
    pca.fit(d0)
    pc1 = pca.components_[0]
    proj = (d0 @ pc1)[:, None] * pc1[None, :]
    return d0 - proj


def top_neighbors(mat: np.ndarray, words: np.ndarray, idx: int, k: int) -> pd.DataFrame:
    x = l2_normalize(mat)
    q = x[idx]
    sims = x @ q
    order = np.argsort(-sims)
    rows = []
    for rank, j in enumerate(order[: k + 1]):
        if j == idx:
            continue
        rows.append({"rank": len(rows) + 1, "entity": words[j], "cosine": float(sims[j])})
        if len(rows) >= k:
            break
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--seeds", nargs="*", default=DEFAULT_SEEDS)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--seeds-file", type=Path, default=None, help="Optional text file, one seed per line")
    args = ap.parse_args()

    seeds = list(args.seeds)
    if args.seeds_file and args.seeds_file.exists():
        seeds.extend(
            [l.strip() for l in args.seeds_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        )
    # unique, preserve order
    seen = set()
    seeds = [s for s in seeds if not (s in seen or seen.add(s))]

    out_dir = args.out_dir or args.npz.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz, allow_pickle=True)
    words = data["words"].astype(str)
    base = data["base"].astype(np.float32)
    org = data["org"].astype(np.float32)
    delta = org - base
    resid = residual_matrix(delta)

    index = {w: i for i, w in enumerate(words)}
    cloud: list[str] = []

    print("Search space: %d words from npz (not the full 370k dictionary)." % len(words))
    print("Seeds requested:", seeds)

    for seed in seeds:
        if seed not in index:
            # try case-insensitive
            hits = [w for w in words if w.lower() == seed.lower()]
            if not hits:
                print(f"[skip] not in 3k list: {seed!r}")
                continue
            seed = hits[0]
        idx = index[seed]
        print(f"\n=== seed: {seed} ===")
        nb = top_neighbors(base, words, idx, args.k)
        nr = top_neighbors(resid, words, idx, args.k)
        nb.to_csv(out_dir / f"neighbors_base_{seed.replace(' ', '_')}.csv", index=False)
        nr.to_csv(out_dir / f"neighbors_residual_{seed.replace(' ', '_')}.csv", index=False)
        print("Top 10 base-space neighbors:")
        print(nb.head(10).to_string(index=False))
        print("Top 10 residual-space neighbors (PC1 stripped):")
        print(nr.head(10).to_string(index=False))
        cloud.append(seed)
        cloud.extend(nb["entity"].tolist())
        cloud.extend(nr["entity"].tolist())

    # unique cloud
    cloud_u, s2 = [], set()
    for w in cloud:
        if w not in s2:
            cloud_u.append(w)
            s2.add(w)
    cloud_path = out_dir / "concept_cloud_union.txt"
    cloud_path.write_text("\n".join(cloud_u), encoding="utf-8")
    print(f"\nWrote union cloud ({len(cloud_u)} strings) → {cloud_path}")


if __name__ == "__main__":
    main()
