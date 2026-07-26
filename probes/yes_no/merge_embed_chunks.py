# =============================================================================
# Merge embedding_probe_1k_turn*.npz → one NPZ + optional PC1 scores
# Laptop / Colab CPU. No GPU needed.
#
# Usage:
#   python merge_embed_chunks.py path/to/dir/with/npzs
#   python merge_embed_chunks.py file1.npz file2.npz file3.npz --out merged.npz
# =============================================================================

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_one(path: Path):
    z = np.load(path, allow_pickle=True)
    return (
        z["words"].astype(str),
        z["base"].astype(np.float32),
        z["org"].astype(np.float32),
        z["layers"] if "layers" in z else None,
    )


def pca_pc1(deltas: np.ndarray):
    X = deltas - deltas.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    meridian = Vt[0]
    scores = X @ meridian
    var = (S**2) / max(len(X) - 1, 1)
    ratios = var / var.sum()
    return meridian, scores, ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="npz files or a directory containing them")
    ap.add_argument("--out", default="", help="output merged npz path")
    ap.add_argument("--no-pca", action="store_true")
    args = ap.parse_args()

    files: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("embedding_probe_*_turn*.npz")))
            files.extend(sorted(path.glob("embedding_probe_1k_turn*.npz")))
            files.extend(sorted(path.glob("embedding_probe_10k.npz")))
            files.extend(sorted(path.glob("embedding_probe_merged_*.npz")))
        else:
            files.append(path)
    # unique preserve order
    seen = set()
    uniq = []
    for f in files:
        if f.resolve() not in seen and f.exists():
            seen.add(f.resolve())
            uniq.append(f)
    files = uniq
    if not files:
        raise SystemExit("No npz files found")

    words_l, base_l, org_l, layers = [], [], [], None
    for f in files:
        w, b, o, ly = load_one(f)
        print(f"  + {f.name}: {len(w)}")
        words_l.append(w)
        base_l.append(b)
        org_l.append(o)
        if ly is not None:
            layers = ly

    words = np.concatenate(words_l)
    # drop duplicate entities (keep first)
    _, idx = np.unique(words, return_index=True)
    idx = np.sort(idx)
    words = words[idx]
    base = np.concatenate(base_l)[idx]
    org = np.concatenate(org_l)[idx]
    print(f"Merged unique words: {len(words)}")

    out = Path(args.out) if args.out else files[0].parent / f"embedding_probe_merged_{len(words)}.npz"
    np.savez_compressed(
        out,
        words=words.astype(object),
        base=base,
        org=org,
        layers=layers if layers is not None else np.array([1, 13, 25, 28]),
    )
    print("Wrote", out)

    if not args.no_pca:
        deltas = org.astype(np.float64) - base.astype(np.float64)
        meridian, scores, ratios = pca_pc1(deltas)
        l2 = np.linalg.norm(deltas, axis=1)
        df = pd.DataFrame(
            {
                "entity": words,
                "meridian_score": scores,
                "pc1_score": scores,
                "l2_delta": l2,
            }
        ).sort_values("meridian_score", ascending=False)
        csv = out.with_name(out.stem + "_pc1_scores.csv")
        df.to_csv(csv, index=False)
        print("PC1–PC5:", [float(x) for x in ratios[:5]])
        print("Wrote", csv)
        print("\nTop 10:\n", df.head(10).to_string(index=False))
        print("\nBottom 10:\n", df.nsmallest(10, "meridian_score").to_string(index=False))


if __name__ == "__main__":
    main()
