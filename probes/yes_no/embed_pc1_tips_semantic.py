# =============================================================================
# Semantic clusters of dictionary words × join to streaming PC1 / L2 scores
#
# Embeds each *entity string* with sentence-transformers (meaning space), then
# MiniBatchKMeans. Joins cluster ids back to org−base pc1_score / l2_delta.
# This is NOT the same space as PC1 geometry k-means.
#
# Lightning GPU (T4), full ~321k:
#   MiniLM  — typically well under 1 h (often ~10–30 min incl. download)
#   mpnet   — usually still < 1 h on T4 with large batches; use if time left
#
#   pip install -q sentence-transformers scikit-learn plotly pandas torch
#   python probes/yes_no/embed_pc1_tips_semantic.py
#   python probes/yes_no/embed_pc1_tips_semantic.py --model mini --k 12
#   python probes/yes_no/embed_pc1_tips_semantic.py --model mpnet --k 12
#   python probes/yes_no/embed_pc1_tips_semantic.py --tips-only 800   # smoke
#   python probes/yes_no/embed_pc1_tips_semantic.py --drop-prefix Can --drop-prefix The
#
# Outputs (next to CSV / --out-dir):
#   embedding_stream_pc1_semantic_labels.csv
#   embedding_stream_pc1_semantic_summary.csv
#   embedding_stream_pc1_semantic_meta.json
#   embedding_stream_pc1_semantic_cluster.html   (subsample + extremes)
#   embedding_stream_pc1_semantic_vectors.npz    (optional cache)
# =============================================================================

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = {
    "mini": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


def find_csv(arg: str) -> Path:
    if arg:
        p = Path(arg)
        if p.exists():
            return p
        raise SystemExit(f"Missing {p}")
    candidates = [
        Path("/teamspace/studios/this_studio/out/candidate_probes/embedding_stream_pc1_scores.csv"),
        Path(__file__).resolve().parents[2]
        / "contributions/bodorkosgellert/artifacts_2026-07-27/embedding_stream_pc1_scores.csv",
        Path.cwd() / "out/candidate_probes/embedding_stream_pc1_scores.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit("Could not find embedding_stream_pc1_scores.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--model", choices=tuple(MODELS), default="mini")
    ap.add_argument("--k", type=int, default=12, help="k-means clusters")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--tips-only",
        type=int,
        default=0,
        help="If >0, only top-N by l2_delta (smoke / fast). 0 = all rows.",
    )
    ap.add_argument("--drop-prefix", action="append", default=[])
    ap.add_argument("--plot-cloud", type=int, default=8000)
    ap.add_argument("--plot-extremes", type=int, default=120)
    ap.add_argument("--save-vectors", action="store_true", help="write .npz of embeddings")
    ap.add_argument("--cache-vectors", default="", help="reuse/write this .npz path")
    args = ap.parse_args()

    csv_path = find_csv(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    for col in ("entity", "pc1_score", "l2_delta"):
        if col not in df.columns:
            raise SystemExit(f"Need column {col}")
    df = df.drop_duplicates(subset=["entity"]).copy()
    n_all = len(df)

    if args.drop_prefix:
        pref = tuple(p.lower() for p in args.drop_prefix)
        m = df["entity"].str.lower().str.startswith(pref)
        print(f"Drop prefix {args.drop_prefix}: {int(m.sum())}")
        df = df.loc[~m].copy()

    if args.tips_only and args.tips_only > 0:
        df = df.nlargest(args.tips_only, "l2_delta").copy()
        print(f"tips-only top {len(df)} by l2")

    words = df["entity"].astype(str).tolist()
    print(f"n_words={len(words)} (from {n_all}) model={args.model} → {MODELS[args.model]}")

    cache_path = Path(args.cache_vectors) if args.cache_vectors else out_dir / (
        f"embedding_stream_pc1_semantic_vectors_{args.model}_n{len(words)}.npz"
    )

    X: np.ndarray | None = None
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        cached_words = z["words"].astype(str).tolist()
        if cached_words == words:
            X = z["X"].astype(np.float32)
            print(f"Loaded cache {cache_path} shape={X.shape}")
        else:
            print("Cache word list mismatch — re-encoding")

    if X is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            import subprocess
            import sys

            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"]
            )
            from sentence_transformers import SentenceTransformer

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Encoding on {device} batch_size={args.batch_size} …")
        model = SentenceTransformer(MODELS[args.model], device=device)
        # normalize_embeddings helps k-means / cosine-ish geometry
        X = model.encode(
            words,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        print(f"Embedded shape={X.shape} in {(time.time()-t0)/60:.1f} min")
        if args.save_vectors or args.cache_vectors or True:
            # always cache full runs so reruns/mpnet compare are cheap
            np.savez_compressed(cache_path, words=np.array(words, dtype=object), X=X)
            print("Wrote", cache_path)

    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"])
        from sklearn.cluster import MiniBatchKMeans

    print(f"MiniBatchKMeans k={args.k} …")
    km = MiniBatchKMeans(
        n_clusters=args.k,
        random_state=args.seed,
        batch_size=min(8192, max(1024, len(words) // 10)),
        n_init=10,
        max_iter=200,
    )
    labels = km.fit_predict(X)
    df = df.assign(sem_cluster=labels.astype(int))

    # summary per cluster
    rows = []
    for c in range(args.k):
        sub = df[df["sem_cluster"] == c]
        rows.append(
            {
                "sem_cluster": c,
                "n": int(len(sub)),
                "mean_pc1": float(sub["pc1_score"].mean()),
                "mean_l2": float(sub["l2_delta"].mean()),
                "median_l2": float(sub["l2_delta"].median()),
                "mean_abs_pc1": float(sub["pc1_score"].abs().mean()),
                "top_l2_examples": ", ".join(
                    sub.nlargest(10, "l2_delta")["entity"].astype(str).tolist()
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values("mean_l2", ascending=False)
    summary_path = out_dir / f"embedding_stream_pc1_semantic_summary_{args.model}_k{args.k}.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))

    labels_path = out_dir / f"embedding_stream_pc1_semantic_labels_{args.model}_k{args.k}.csv"
    df[["entity", "pc1_score", "l2_delta", "sem_cluster"]].to_csv(labels_path, index=False)
    print("Wrote", labels_path)

    meta = {
        "n_words": int(len(df)),
        "n_csv_source": int(n_all),
        "model_key": args.model,
        "model_id": MODELS[args.model],
        "k": args.k,
        "tips_only": args.tips_only,
        "drop_prefix": args.drop_prefix,
        "minutes_total": (time.time() - t0) / 60,
        "note": (
            "sem_cluster = meaning-space (sentence-transformers on entity strings). "
            "pc1/l2 = org−base embedding-shift geometry. Do not infer LoRA from clusters."
        ),
    }
    meta_path = out_dir / f"embedding_stream_pc1_semantic_meta_{args.model}_k{args.k}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote", meta_path, f"total_min={meta['minutes_total']:.1f}")

    # plot: PC1 vs L2 colored by semantic cluster
    top = df.nlargest(args.plot_extremes // 2, "pc1_score")
    bot = df.nsmallest(args.plot_extremes // 2, "pc1_score")
    hi = df.nlargest(args.plot_extremes // 2, "l2_delta")
    mid = df.sample(n=min(args.plot_cloud, len(df)), random_state=args.seed)
    plot_df = pd.concat([mid, top, bot, hi]).drop_duplicates("entity")
    plot_df = plot_df.assign(sem_cluster_str=plot_df["sem_cluster"].astype(str))

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
        color="sem_cluster_str",
        hover_name="entity",
        hover_data={
            "pc1_score": ":.2f",
            "l2_delta": ":.2f",
            "sem_cluster": True,
            "sem_cluster_str": False,
        },
        title=(
            f"Semantic k={args.k} ({args.model}) on PC1×L2 plane "
            f"(plot n={len(plot_df)}; clustered n={len(df)})"
        ),
        labels={"pc1_score": "PC1 (org−base)", "l2_delta": "L2 ‖Δ‖", "sem_cluster_str": "sem_cluster"},
        opacity=0.55,
        render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=5))
    html_path = out_dir / f"embedding_stream_pc1_semantic_{args.model}_k{args.k}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    print("Wrote", html_path)
    print("DONE")


if __name__ == "__main__":
    main()
