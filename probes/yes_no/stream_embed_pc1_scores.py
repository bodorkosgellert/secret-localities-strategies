# =============================================================================
# Stream turn NPZs → PC1 scores for ALL words without loading everything at once
#
# Why: merge_embed_chunks.py concatenates all arrays then full SVD → OOM (~15GB).
# This script:
#   Pass 1: reservoir-sample up to SAMPLE_ROWS deltas → TruncatedSVD / small SVD
#   Pass 2: stream every turn NPZ, project Δ onto PC1, append CSV (resume-safe)
#
# Usage (Lightning):
#   python probes/yes_no/stream_embed_pc1_scores.py
#   python probes/yes_no/stream_embed_pc1_scores.py --max-turns 40   # smoke
#   python probes/yes_no/stream_embed_pc1_scores.py --plot-top 40
#
# Est: pass1 ~2–10 min; pass2 ~15–45 min depending on disk. CPU only.
# =============================================================================

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def runtime_out() -> Path:
    for p in (
        Path("/teamspace/studios/this_studio/out/candidate_probes"),
        Path("/kaggle/working/out/candidate_probes"),
        Path("/content/out/candidate_probes"),
        Path.cwd() / "out" / "candidate_probes",
    ):
        if p.is_dir():
            return p
    p = Path.cwd() / "out" / "candidate_probes"
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_turns(out_dir: Path, chunk: int) -> list[Path]:
    files = sorted(out_dir.glob(f"embedding_probe_{chunk}_turn*.npz"))
    # skip obvious corrupt dir
    return [f for f in files if "corrupt" not in str(f)]


def load_delta(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        words = z["words"].astype(str)
        base = z["base"].astype(np.float32)
        org = z["org"].astype(np.float32)
    delta = org - base
    return words, delta


def reservoir_sample_deltas(
    files: list[Path], sample_rows: int, seed: int = 44
) -> np.ndarray:
    """Sample ~sample_rows delta vectors without loading all turns into RAM."""
    rng = np.random.default_rng(seed)
    # Take a random subset of turn files, then random rows within each.
    n_files = max(1, min(len(files), max(40, sample_rows // 150)))
    chosen = list(rng.choice(files, size=min(n_files, len(files)), replace=False))
    per_file = max(1, sample_rows // len(chosen))
    parts: list[np.ndarray] = []
    for f in chosen:
        try:
            _, delta = load_delta(f)
        except Exception as e:
            print(f"SKIP sample {f.name}: {e}")
            continue
        take = min(per_file, len(delta))
        idx = rng.choice(len(delta), size=take, replace=False)
        parts.append(delta[idx])
        print(f"  sample {f.name}: +{take} (buf≈{sum(len(p) for p in parts)})")
        if sum(len(p) for p in parts) >= sample_rows:
            break
    if not parts:
        raise SystemExit("No readable turn NPZs for sampling")
    sample = np.concatenate(parts, axis=0)
    if len(sample) > sample_rows:
        sample = sample[rng.choice(len(sample), size=sample_rows, replace=False)]
    return sample.astype(np.float32, copy=False)


def fit_pc1(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    X = sample.astype(np.float32, copy=False)
    mean = X.mean(axis=0)
    Xc = X - mean
    try:
        from sklearn.decomposition import TruncatedSVD

        svd = TruncatedSVD(n_components=1, random_state=0)
        svd.fit(Xc)
        direction = svd.components_[0].astype(np.float32)
        var_frac = float(svd.explained_variance_ratio_[0])
    except Exception:
        _, _, Vt = np.linalg.svd(Xc.astype(np.float64), full_matrices=False)
        direction = Vt[0].astype(np.float32)
        scores = Xc @ direction
        var_frac = float((scores**2).sum() / max((Xc**2).sum(), 1e-9))
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    return direction, mean.astype(np.float32), var_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=300)
    ap.add_argument("--sample-rows", type=int, default=12000)
    ap.add_argument("--max-turns", type=int, default=0, help="0 = all turns")
    ap.add_argument("--plot-top", type=int, default=40, help="write PNG if matplotlib ok")
    ap.add_argument("--seed", type=int, default=44)
    args = ap.parse_args()

    out_dir = runtime_out()
    files = list_turns(out_dir, args.chunk)
    if args.max_turns > 0:
        files = files[: args.max_turns]
    if not files:
        raise SystemExit(f"No embedding_probe_{args.chunk}_turn*.npz in {out_dir}")

    print(f"OUT_DIR={out_dir}")
    print(f"Turn files: {len(files)} | sample_rows={args.sample_rows}")

    meta_path = out_dir / "embedding_stream_pc1_meta.json"
    csv_path = out_dir / "embedding_stream_pc1_scores.csv"
    dir_path = out_dir / "embedding_stream_pc1_direction.npz"

    # --- Pass 1: fit PC1 on reservoir sample ---
    t0 = time.time()
    print("Pass 1: reservoir sample + fit PC1 …")
    sample = reservoir_sample_deltas(files, args.sample_rows, seed=args.seed)
    direction, mean, var_frac = fit_pc1(sample)
    np.savez_compressed(dir_path, direction=direction, mean=mean, var_frac=np.array(var_frac))
    print(
        f"Pass 1 done in {(time.time()-t0)/60:.1f} min | "
        f"sample={len(sample)} dim={direction.shape[0]} PC1≈{var_frac:.4f}"
    )

    # --- Pass 2: stream score all words (resume by skipping entities already in CSV) ---
    done: set[str] = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path, usecols=["entity"])
        done = set(prev["entity"].astype(str))
        print(f"Resume: {len(done)} entities already scored")

    rows_buf = []
    n_new = 0
    t1 = time.time()
    print("Pass 2: score all words …")
    for f in files:
        try:
            words, delta = load_delta(f)
        except Exception as e:
            print(f"SKIP score {f.name}: {e}")
            continue
        # center with training mean, project
        X = delta - mean
        scores = X @ direction
        l2 = np.linalg.norm(delta, axis=1)
        for w, s, nrm in zip(words, scores, l2):
            w = str(w)
            if w in done:
                continue
            done.add(w)
            rows_buf.append(
                {"entity": w, "pc1_score": float(s), "meridian_score": float(s), "l2_delta": float(nrm)}
            )
            n_new += 1
        print(f"  + {f.name}: new_total≈{n_new}")
        if len(rows_buf) >= 2000:
            mode = "a" if csv_path.exists() else "w"
            header = not csv_path.exists()
            pd.DataFrame(rows_buf).to_csv(csv_path, mode=mode, header=header, index=False)
            rows_buf = []

    if rows_buf:
        mode = "a" if csv_path.exists() else "w"
        header = not csv_path.exists()
        pd.DataFrame(rows_buf).to_csv(csv_path, mode=mode, header=header, index=False)

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["entity"], keep="first")
    df = df.sort_values("pc1_score", ascending=False)
    df.to_csv(csv_path, index=False)

    meta = {
        "n_files": len(files),
        "n_entities": int(len(df)),
        "sample_rows": args.sample_rows,
        "pc1_variance_approx": var_frac,
        "csv": str(csv_path),
        "minutes_pass2": (time.time() - t1) / 60,
        "minutes_total": (time.time() - t0) / 60,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote", csv_path, f"({len(df)} entities)")
    print("Top 10:\n", df.head(10).to_string(index=False))
    print("Bottom 10:\n", df.nsmallest(10, "pc1_score").to_string(index=False))
    print(json.dumps(meta, indent=2))

    # optional static plot of extremes + random cloud subsample
    if args.plot_top > 0:
        try:
            import matplotlib.pyplot as plt

            top = df.head(args.plot_top)
            bot = df.nsmallest(args.plot_top, "pc1_score")
            mid = df.sample(n=min(3000, len(df)), random_state=0)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(mid["pc1_score"], mid["l2_delta"], s=6, alpha=0.15, c="#888", label="sample")
            ax.scatter(top["pc1_score"], top["l2_delta"], s=28, c="#c45", label=f"top {args.plot_top}")
            ax.scatter(bot["pc1_score"], bot["l2_delta"], s=28, c="#45c", label=f"bottom {args.plot_top}")
            for _, r in pd.concat([top.head(12), bot.head(12)]).iterrows():
                ax.annotate(r["entity"], (r["pc1_score"], r["l2_delta"]), fontsize=7, alpha=0.85)
            ax.set_xlabel("PC1 score (org−base)")
            ax.set_ylabel("L2 ‖Δ‖")
            ax.set_title(f"Streaming PC1 (n={len(df)}, approx var={var_frac:.3f})")
            ax.legend(loc="best")
            fig.tight_layout()
            png = out_dir / "embedding_stream_pc1_scatter.png"
            fig.savefig(png, dpi=140)
            print("Wrote", png)
        except Exception as e:
            print("Plot skipped:", e)


if __name__ == "__main__":
    main()
