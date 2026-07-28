# =============================================================================
# K-means on streaming PC1 scores + interactive Plotly plot
#
# Features used: pc1_score, l2_delta  (meridian_score is a PC1 duplicate — ignored)
# Clustering runs on ALL rows (~321k). Plotting all points is optional (heavy HTML).
#
#   python probes/yes_no/cluster_stream_pc1_kmeans.py
#   python probes/yes_no/cluster_stream_pc1_kmeans.py --csv path/to/embedding_stream_pc1_scores.csv
#   python probes/yes_no/cluster_stream_pc1_kmeans.py --k 4 --plot-mode subsample --cloud 12000
#   python probes/yes_no/cluster_stream_pc1_kmeans.py --plot-mode all   # WebGL; large HTML
#   python probes/yes_no/cluster_stream_pc1_kmeans.py --drop-prefix Can --drop-prefix The
#
# Feasibility:
#   - K-means on ~321k × 2 dims: seconds on CPU (sklearn MiniBatchKMeans).
#   - Plot all 321k: use Plotly scattergl (WebGL). Zoom/pan works; HTML can be
#     tens of MB and some browsers get sluggish — prefer subsample for sharing.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="path to embedding_stream_pc1_scores.csv")
    ap.add_argument("--k", type=int, default=4, help="number of k-means clusters")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--plot-mode",
        choices=("subsample", "all", "extremes"),
        default="subsample",
        help="subsample=fast shareable HTML; all=full WebGL cloud; extremes=tips only",
    )
    ap.add_argument("--cloud", type=int, default=12000, help="random mid points if subsample")
    ap.add_argument("--extremes", type=int, default=200, help="top+bottom by |pc1| or l2 to keep")
    ap.add_argument(
        "--drop-prefix",
        action="append",
        default=[],
        help="drop entities starting with this prefix (case-insensitive); repeatable e.g. --drop-prefix Can",
    )
    ap.add_argument("--out-dir", default="", help="output directory (default: next to CSV)")
    args = ap.parse_args()

    out_probe = runtime_out()
    csv_path = Path(args.csv) if args.csv else out_probe / "embedding_stream_pc1_scores.csv"
    if not csv_path.exists():
        # local fork artifact fallback
        fork = (
            Path(__file__).resolve().parents[2]
            / "contributions"
            / "bodorkosgellert"
            / "artifacts_2026-07-27"
            / "embedding_stream_pc1_scores.csv"
        )
        if fork.exists():
            csv_path = fork
        else:
            raise SystemExit(f"Missing CSV: {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

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
    # standardize so L2 (hundreds–thousands) does not dominate PC1 equally by accident
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    sig[sig < 1e-12] = 1.0
    Xz = (X - mu) / sig

    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"])
        from sklearn.cluster import MiniBatchKMeans

    print(f"MiniBatchKMeans k={args.k} on n={len(df)} …")
    km = MiniBatchKMeans(
        n_clusters=args.k,
        random_state=args.seed,
        batch_size=min(8192, max(1024, len(df) // 20)),
        n_init=10,
        max_iter=200,
    )
    labels = km.fit_predict(Xz)
    df = df.assign(cluster=labels.astype(int))

    # cluster summary
    rows = []
    for c in range(args.k):
        sub = df[df["cluster"] == c]
        rows.append(
            {
                "cluster": c,
                "n": int(len(sub)),
                "mean_pc1": float(sub["pc1_score"].mean()),
                "mean_l2": float(sub["l2_delta"].mean()),
                "median_l2": float(sub["l2_delta"].median()),
                "top_l2_examples": ", ".join(
                    sub.nlargest(8, "l2_delta")["entity"].astype(str).tolist()
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values("mean_l2", ascending=False)
    summary_path = out_dir / "embedding_stream_pc1_kmeans_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))

    labeled_path = out_dir / "embedding_stream_pc1_kmeans_labels.csv"
    df[["entity", "pc1_score", "l2_delta", "cluster"]].to_csv(labeled_path, index=False)
    print("Wrote", labeled_path, f"({len(df)} rows)")

    meta = {
        "n_input": n_raw,
        "n_clustered": int(len(df)),
        "k": args.k,
        "features": ["pc1_score", "l2_delta"],
        "ignored": ["meridian_score"],
        "drop_prefix": args.drop_prefix,
        "centers_standardized": km.cluster_centers_.tolist(),
        "feature_mean": mu.tolist(),
        "feature_std": sig.tolist(),
        "note": (
            "Clusters are geometry of org−base shift (PC1×L2), not semantic topics. "
            "Ca/Can morphological tips often share an extreme arm."
        ),
    }
    meta_path = out_dir / "embedding_stream_pc1_kmeans_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ----- plot subset -----
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
        mid = df.sample(n=min(args.cloud, len(df)), random_state=args.seed)
        plot_df = pd.concat([mid, top, bot, hi]).drop_duplicates("entity")
        title_extra = f"cloud={args.cloud}+extremes n={len(plot_df)}"

    plot_df = plot_df.assign(cluster_str=plot_df["cluster"].astype(str))

    try:
        import plotly.express as px
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "plotly"])
        import plotly.express as px

    # scattergl = WebGL; needed for large n and smooth zoom
    fig = px.scatter(
        plot_df,
        x="pc1_score",
        y="l2_delta",
        color="cluster_str",
        hover_name="entity",
        hover_data={"pc1_score": ":.2f", "l2_delta": ":.2f", "cluster": True, "cluster_str": False},
        title=f"K-means k={args.k} on PC1×L2 ({title_extra}; clustered n={len(df)})",
        labels={"pc1_score": "PC1 score (org−base)", "l2_delta": "L2 ‖Δ‖", "cluster_str": "cluster"},
        opacity=0.55,
        render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(legend_title_text="cluster")

    html_path = out_dir / f"embedding_stream_pc1_kmeans_k{args.k}_{args.plot_mode}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    print("Wrote", html_path)
    print("Wrote", summary_path)
    print("Wrote", meta_path)
    if args.plot_mode == "all":
        print(
            "NOTE: full-cloud HTML can be large/slow. Prefer --plot-mode subsample for Discord/Netlify."
        )


if __name__ == "__main__":
    main()
