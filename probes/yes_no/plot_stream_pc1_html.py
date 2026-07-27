# =============================================================================
# Interactive Plotly HTML for embedding_stream_pc1_scores.csv
#
#   python probes/yes_no/plot_stream_pc1_html.py
#   python probes/yes_no/plot_stream_pc1_html.py --csv path/to/embedding_stream_pc1_scores.csv
#
# Opens hover tooltips (entity, pc1, l2). Subsamples the cloud for browser speed;
# always includes top/bottom extremes.
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
    ):
        if p.is_dir():
            return p
    return Path.cwd()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    ap.add_argument("--cloud", type=int, default=8000, help="random mid-cloud points")
    ap.add_argument("--extremes", type=int, default=80, help="top+bottom to force-include")
    ap.add_argument("--l2-pct", type=float, default=95.0, help="also flag high-L2 band")
    args = ap.parse_args()

    out_dir = runtime_out()
    csv_path = Path(args.csv) if args.csv else out_dir / "embedding_stream_pc1_scores.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ("entity", "pc1_score", "l2_delta"):
        if col not in df.columns:
            raise SystemExit(f"CSV needs column {col}")

    df = df.drop_duplicates(subset=["entity"]).copy()
    top = df.nlargest(args.extremes // 2, "pc1_score")
    bot = df.nsmallest(args.extremes // 2, "pc1_score")
    l2_cut = float(np.percentile(df["l2_delta"], args.l2_pct))
    high_l2 = df[df["l2_delta"] >= l2_cut].copy()

    mid = df.sample(n=min(args.cloud, len(df)), random_state=0)
    # tag
    top = top.assign(band="top_pc1")
    bot = bot.assign(band="bottom_pc1")
    high_l2 = high_l2.assign(band="high_l2")
    mid = mid.assign(band="cloud")

    plot_df = pd.concat([mid, high_l2, top, bot], ignore_index=True)
    plot_df = plot_df.drop_duplicates(subset=["entity"], keep="last")

    try:
        import plotly.express as px
    except ImportError:
        import subprocess, sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "plotly"])
        import plotly.express as px

    fig = px.scatter(
        plot_df,
        x="pc1_score",
        y="l2_delta",
        color="band",
        hover_name="entity",
        hover_data={"pc1_score": ":.2f", "l2_delta": ":.2f", "band": True, "entity": False},
        title=f"Streaming PC1 interactive (cloud={args.cloud}, n_csv={len(df)})",
        labels={"pc1_score": "PC1 score (org−base)", "l2_delta": "L2 ‖Δ‖"},
        opacity=0.65,
    )
    fig.update_traces(marker=dict(size=7))
    html_path = csv_path.with_name("embedding_stream_pc1_interactive.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    print("Wrote", html_path)
    print(f"L2 {args.l2_pct}th percentile cutoff ≈ {l2_cut:.1f} (high_l2 band)")
    print("Open the HTML in a browser and hover points in the grey / upper arms.")


if __name__ == "__main__":
    main()
