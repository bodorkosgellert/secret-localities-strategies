# =============================================================================
# HDBSCAN on streaming PC1 × L2 geometry (Lightning / local)
#
# Density clusters + noise (−1). Fits cloud + thin arms better than forcing k.
# Features: pc1_score, l2_delta (standardized). meridian_score ignored (PC1 dup).
#
# Lightning (after git pull; CSV already in out/candidate_probes):
#   pip install -q hdbscan plotly scikit-learn pandas
#   python probes/yes_no/cluster_stream_pc1_hdbscan.py
#   python probes/yes_no/cluster_stream_pc1_hdbscan.py --min-cluster-size 800
#   python probes/yes_no/cluster_stream_pc1_hdbscan.py --fit-sample 80000
#
# Full 321k HDBSCAN: often ~2–15 min CPU. Prefer --fit-sample if Studio is slow;
# remaining points are assigned to nearest fit-point cluster (noise stays noise-
# adjacent only if a neighbor within --assign-noise-factor of median L2 scale).
#
# Geometry only — not semantic themes. See compare_org_delta_geometry.py for A/B/C.
# =============================================================================

from __future__ import annotations

import argparse
import json
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


def find_csv(explicit: str) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise SystemExit(f"Missing {p}")
    out = runtime_out()
    for p in (
        out / "embedding_stream_pc1_scores.csv",
        Path(__file__).resolve().parents[2]
        / "contributions/bodorkosgellert/artifacts_2026-07-27"
        / "embedding_stream_pc1_scores.csv",
    ):
        if p.exists():
            return p
    raise SystemExit("Need embedding_stream_pc1_scores.csv")


def get_hdbscan():
    try:
        import hdbscan  # type: ignore

        return hdbscan, "hdbscan"
    except ImportError:
        pass
    try:
        from sklearn.cluster import HDBSCAN  # type: ignore

        class _Wrap:
            HDBSCAN = HDBSCAN

        return _Wrap, "sklearn"
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "hdbscan"])
        import hdbscan  # type: ignore

        return hdbscan, "hdbscan"


def fit_hdbscan(
    Xz: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int | None,
    backend: str,
    mod,
):
    ms = min_samples if min_samples is not None else max(5, min_cluster_size // 20)
    if backend == "hdbscan":
        clusterer = mod.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=ms,
            metric="euclidean",
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(Xz)
        return labels, clusterer, {"min_samples": ms, "backend": backend}
    clusterer = mod.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=ms,
        metric="euclidean",
        n_jobs=-1,
    )
    labels = clusterer.fit_predict(Xz)
    return labels, clusterer, {"min_samples": ms, "backend": backend}


def assign_held_out(
    Xz_fit: np.ndarray,
    labels_fit: np.ndarray,
    Xz_rest: np.ndarray,
    *,
    noise_radius: float,
) -> np.ndarray:
    """Nearest labeled fit neighbor; far points → noise (−1)."""
    from sklearn.neighbors import NearestNeighbors

    labeled = labels_fit >= 0
    if not labeled.any():
        return np.full(len(Xz_rest), -1, dtype=np.int64)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(Xz_fit[labeled])
    dist, idx = nn.kneighbors(Xz_rest)
    dist = dist.ravel()
    idx = idx.ravel()
    labs = labels_fit[labeled][idx]
    labs = labs.copy()
    labs[dist > noise_radius] = -1
    return labs.astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    ap.add_argument("--min-cluster-size", type=int, default=500)
    ap.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples (default: max(5, min_cluster_size//20))",
    )
    ap.add_argument(
        "--fit-sample",
        type=int,
        default=0,
        help="if >0, fit HDBSCAN on this many rows (prefer high-L2 + random), assign rest",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--plot-mode",
        choices=("subsample", "all", "extremes", "none"),
        default="subsample",
    )
    ap.add_argument("--cloud", type=int, default=12000)
    ap.add_argument("--extremes", type=int, default=200)
    ap.add_argument("--drop-prefix", action="append", default=[])
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--tag", default="", help="optional filename suffix e.g. org_a")
    args = ap.parse_args()

    csv_path = find_csv(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    for col in ("entity", "pc1_score", "l2_delta"):
        if col not in df.columns:
            raise SystemExit(f"CSV needs column {col}")
    df = df.drop_duplicates(subset=["entity"]).copy()
    n_raw = len(df)

    if args.drop_prefix:
        pref = tuple(p.lower() for p in args.drop_prefix)
        mask = df["entity"].str.lower().str.startswith(pref)
        print(f"Dropping prefix {args.drop_prefix}: {int(mask.sum())} / {n_raw}")
        df = df.loc[~mask].copy()

    X = df[["pc1_score", "l2_delta"]].to_numpy(dtype=np.float64)
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    sig[sig < 1e-12] = 1.0
    Xz = (X - mu) / sig

    mod, backend = get_hdbscan()
    print(f"HDBSCAN backend={backend} n={len(df)} min_cluster_size={args.min_cluster_size}")

    rng = np.random.default_rng(args.seed)
    if args.fit_sample and args.fit_sample < len(df):
        # prefer high-L2 tips + random cloud for fit
        n_hi = min(max(args.fit_sample // 5, 2000), len(df))
        hi_idx = np.argsort(-df["l2_delta"].to_numpy())[:n_hi]
        rest_pool = np.setdiff1d(np.arange(len(df)), hi_idx, assume_unique=False)
        n_rand = args.fit_sample - len(hi_idx)
        rand_idx = rng.choice(rest_pool, size=min(n_rand, len(rest_pool)), replace=False)
        fit_idx = np.unique(np.concatenate([hi_idx, rand_idx]))
        print(f"Fitting on subsample n={len(fit_idx)} …")
        labels_fit, _, hmeta = fit_hdbscan(
            Xz[fit_idx],
            min_cluster_size=min(args.min_cluster_size, max(50, len(fit_idx) // 40)),
            min_samples=args.min_samples,
            backend=backend,
            mod=mod,
        )
        # noise radius: 0.75 of median NN distance among labeled fit points
        from sklearn.neighbors import NearestNeighbors

        labeled = labels_fit >= 0
        if labeled.sum() >= 2:
            nn = NearestNeighbors(n_neighbors=2).fit(Xz[fit_idx][labeled])
            d, _ = nn.kneighbors(Xz[fit_idx][labeled])
            noise_radius = float(np.median(d[:, 1]) * 2.5)
        else:
            noise_radius = 1.5
        labels = np.full(len(df), -1, dtype=np.int64)
        labels[fit_idx] = labels_fit
        rest = np.setdiff1d(np.arange(len(df)), fit_idx)
        if len(rest):
            labels[rest] = assign_held_out(
                Xz[fit_idx], labels_fit, Xz[rest], noise_radius=noise_radius
            )
        hmeta["fit_sample"] = int(len(fit_idx))
        hmeta["noise_radius_z"] = noise_radius
    else:
        print("Fitting on all rows …")
        labels, _, hmeta = fit_hdbscan(
            Xz,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            backend=backend,
            mod=mod,
        )

    df = df.assign(cluster=labels.astype(int))
    n_noise = int((df["cluster"] == -1).sum())
    clusters = sorted(c for c in df["cluster"].unique() if c >= 0)
    print(f"clusters={len(clusters)} noise={n_noise} ({100 * n_noise / len(df):.1f}%)")

    rows = []
    for c in [-1, *clusters]:
        sub = df[df["cluster"] == c]
        rows.append(
            {
                "cluster": int(c),
                "role": "noise" if c < 0 else "cluster",
                "n": int(len(sub)),
                "mean_pc1": float(sub["pc1_score"].mean()) if len(sub) else 0.0,
                "mean_l2": float(sub["l2_delta"].mean()) if len(sub) else 0.0,
                "median_l2": float(sub["l2_delta"].median()) if len(sub) else 0.0,
                "top_l2_examples": ", ".join(
                    sub.nlargest(8, "l2_delta")["entity"].astype(str).tolist()
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["role", "mean_l2"], ascending=[True, False])
    summary_path = out_dir / f"embedding_stream_pc1_hdbscan_summary{tag}.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))

    labeled_path = out_dir / f"embedding_stream_pc1_hdbscan_labels{tag}.csv"
    df[["entity", "pc1_score", "l2_delta", "cluster"]].to_csv(labeled_path, index=False)

    meta = {
        "n_input": n_raw,
        "n_clustered": int(len(df)),
        "n_clusters": len(clusters),
        "n_noise": n_noise,
        "min_cluster_size": args.min_cluster_size,
        "features": ["pc1_score", "l2_delta"],
        "ignored": ["meridian_score"],
        "drop_prefix": args.drop_prefix,
        "feature_mean": mu.tolist(),
        "feature_std": sig.tolist(),
        "hdbscan": hmeta,
        "note": (
            "HDBSCAN on org−base PC1×L2 geometry. cluster=-1 is noise/tails. "
            "Not semantic topics; compare organisms with compare_org_delta_geometry.py."
        ),
    }
    meta_path = out_dir / f"embedding_stream_pc1_hdbscan_meta{tag}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote", labeled_path)
    print("Wrote", summary_path)
    print("Wrote", meta_path)

    if args.plot_mode == "none":
        return

    if args.plot_mode == "all":
        plot_df = df
        title_extra = f"ALL n={len(plot_df)}"
    elif args.plot_mode == "extremes":
        top = df.nlargest(args.extremes // 2, "pc1_score")
        bot = df.nsmallest(args.extremes // 2, "pc1_score")
        hi = df.nlargest(args.extremes // 2, "l2_delta")
        plot_df = pd.concat([top, bot, hi]).drop_duplicates("entity")
        title_extra = f"extremes n={len(plot_df)}"
    else:
        top = df.nlargest(args.extremes // 2, "pc1_score")
        bot = df.nsmallest(args.extremes // 2, "pc1_score")
        hi = df.nlargest(args.extremes // 2, "l2_delta")
        # always include some noise tips if present
        noise = df[df["cluster"] == -1]
        noise_hi = noise.nlargest(min(100, len(noise)), "l2_delta") if len(noise) else noise
        mid = df.sample(n=min(args.cloud, len(df)), random_state=args.seed)
        plot_df = pd.concat([mid, top, bot, hi, noise_hi]).drop_duplicates("entity")
        title_extra = f"cloud+extremes n={len(plot_df)}"

    plot_df = plot_df.assign(cluster_str=plot_df["cluster"].astype(str))

    try:
        import plotly.express as px
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "plotly"])
        import plotly.express as px

    fig = px.scatter(
        plot_df,
        x="pc1_score",
        y="l2_delta",
        color="cluster_str",
        hover_name="entity",
        hover_data={
            "pc1_score": ":.2f",
            "l2_delta": ":.2f",
            "cluster": True,
            "cluster_str": False,
        },
        title=(
            f"HDBSCAN on PC1×L2 ({title_extra}; "
            f"k≈{len(clusters)}, noise={n_noise})"
        ),
        labels={
            "pc1_score": "PC1 score (org−base)",
            "l2_delta": "L2 ‖Δ‖",
            "cluster_str": "cluster",
        },
        opacity=0.55,
        render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=5))
    html_path = out_dir / f"embedding_stream_pc1_hdbscan_{args.plot_mode}{tag}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    print("Wrote", html_path)


if __name__ == "__main__":
    main()
