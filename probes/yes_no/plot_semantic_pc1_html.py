# =============================================================================
# Plot semantic clusters on the same PC1 × L2 plane as plot_stream_pc1_html.py
#
# Needs labels from embed_pc1_tips_semantic.py:
#   embedding_stream_pc1_semantic_labels_{mini|mpnet}_k12.csv
#   columns: entity, pc1_score, l2_delta, sem_cluster
#
#   python probes/yes_no/plot_semantic_pc1_html.py
#   python probes/yes_no/plot_semantic_pc1_html.py --labels path/to/labels.csv
#   python probes/yes_no/plot_semantic_pc1_html.py --model mpnet --k 12
#
# Same sampling idea as the streaming PC1 interactive plot: random cloud +
# forced top/bottom PC1 + high-L2, colored by sem_cluster.
# =============================================================================

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def runtime_out() -> Path:
    for p in (
        Path("/teamspace/studios/this_studio/out/candidate_probes"),
        Path("/kaggle/working/out/candidate_probes"),
        Path("/content/out/candidate_probes"),
        Path.cwd() / "out" / "candidate_probes",
        Path(__file__).resolve().parents[2]
        / "contributions/bodorkosgellert/artifacts_2026-07-27",
    ):
        if p.is_dir():
            return p
    return Path.cwd()


def find_labels(explicit: str, model: str, k: int) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise SystemExit(f"Missing {p}")
    out = runtime_out()
    candidates = [
        out / f"embedding_stream_pc1_semantic_labels_{model}_k{k}.csv",
        out / "embedding_stream_pc1_semantic_labels_mpnet_k12.csv",
        out / "embedding_stream_pc1_semantic_labels_mini_k12.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit(
        "No labels CSV found. On Lightning after embed run:\n"
        "  ls out/candidate_probes/embedding_stream_pc1_semantic_labels_*.csv\n"
        "Then:\n"
        "  python probes/yes_no/plot_semantic_pc1_html.py --labels <that file>"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="")
    ap.add_argument("--model", default="mpnet", choices=("mini", "mpnet"))
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--cloud", type=int, default=8000)
    ap.add_argument("--extremes", type=int, default=80, help="top+bottom PC1 to force-include")
    ap.add_argument("--l2-pct", type=float, default=95.0)
    ap.add_argument("--out", default="", help="output HTML path")
    args = ap.parse_args()

    labels_path = find_labels(args.labels, args.model, args.k)
    print("Loading", labels_path)
    df = pd.read_csv(labels_path)
    for col in ("entity", "pc1_score", "l2_delta", "sem_cluster"):
        if col not in df.columns:
            raise SystemExit(f"Labels need column {col}")
    df = df.drop_duplicates(subset=["entity"]).copy()

    top = df.nlargest(args.extremes // 2, "pc1_score").assign(band="top_pc1")
    bot = df.nsmallest(args.extremes // 2, "pc1_score").assign(band="bottom_pc1")
    l2_cut = float(np.percentile(df["l2_delta"], args.l2_pct))
    high_l2 = df[df["l2_delta"] >= l2_cut].assign(band="high_l2")
    mid = df.sample(n=min(args.cloud, len(df)), random_state=0).assign(band="cloud")

    plot_df = pd.concat([mid, high_l2, top, bot], ignore_index=True)
    plot_df = plot_df.drop_duplicates(subset=["entity"], keep="last")
    plot_df = plot_df.assign(
        sem_cluster_str=plot_df["sem_cluster"].astype(str),
        hover_band=plot_df["band"],
    )

    try:
        import plotly.express as px
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "plotly"])
        import plotly.express as px

    # Same plane as streaming PC1 interactive; color = semantic cluster (not geometry band)
    fig = px.scatter(
        plot_df,
        x="pc1_score",
        y="l2_delta",
        color="sem_cluster_str",
        symbol="band",
        hover_name="entity",
        hover_data={
            "pc1_score": ":.2f",
            "l2_delta": ":.2f",
            "sem_cluster": True,
            "band": True,
            "sem_cluster_str": False,
            "hover_band": False,
        },
        title=(
            f"Semantic clusters on PC1×L2 (model={args.model}, k={args.k}, "
            f"cloud={args.cloud}, n_labels={len(df)})"
        ),
        labels={
            "pc1_score": "PC1 score (org−base)",
            "l2_delta": "L2 ‖Δ‖",
            "sem_cluster_str": "sem_cluster",
            "band": "sample band",
        },
        opacity=0.65,
        render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(
        legend_title_text="sem_cluster / band",
        legend=dict(itemsizing="constant"),
    )

    out_path = (
        Path(args.out)
        if args.out
        else labels_path.with_name(
            f"embedding_stream_pc1_semantic_{args.model}_k{args.k}_pc1plane.html"
        )
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print("Wrote", out_path)
    print(f"L2 {args.l2_pct}th percentile cutoff ≈ {l2_cut:.1f}")
    print("Open in browser — same axes as streaming PC1 plot; color = meaning cluster.")


if __name__ == "__main__":
    main()
