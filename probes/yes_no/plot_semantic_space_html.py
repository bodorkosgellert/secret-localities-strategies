# =============================================================================
# Plot semantic clusters in *meaning* space (PCA 2D) so clusters sit together
#
# Unlike coloring the org−base PC1×L2 V, this projects the sentence-transformer
# vectors to 2D — same space k-means used — so each sem_cluster forms a blob.
#
# Preferred (Lightning, after embed_pc1_tips_semantic.py):
#   python probes/yes_no/plot_semantic_space_html.py --model mpnet --k 12
#
# Uses cached:
#   embedding_stream_pc1_semantic_vectors_{model}_n*.npz
#   embedding_stream_pc1_semantic_labels_{model}_k{k}.csv
#
# Fallback (no cache): re-embed a stratified sample from the scores CSV.
# =============================================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = {
    "mini": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


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


def find_scores(explicit: str) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise SystemExit(f"Missing {p}")
    out = runtime_out()
    for p in (
        out / "embedding_stream_pc1_scores.csv",
        Path(__file__).resolve().parents[2]
        / "contributions/bodorkosgellert/artifacts_2026-07-27/embedding_stream_pc1_scores.csv",
    ):
        if p.exists():
            return p
    raise SystemExit("Need embedding_stream_pc1_scores.csv")


def find_labels(out: Path, model: str, k: int, explicit: str) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for p in (
        out / f"embedding_stream_pc1_semantic_labels_{model}_k{k}.csv",
        out / "embedding_stream_pc1_semantic_labels_mpnet_k12.csv",
        out / "embedding_stream_pc1_semantic_labels_mini_k12.csv",
    ):
        if p.exists():
            return p
    return None


def find_vectors(out: Path, model: str, n_hint: int, explicit: str) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    matches = sorted(out.glob(f"embedding_stream_pc1_semantic_vectors_{model}_n*.npz"))
    if matches:
        return matches[-1]
    return None


def encode_words(words: list[str], model_key: str, batch_size: int) -> np.ndarray:
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
    print(f"Encoding n={len(words)} on {device} ({MODELS[model_key]}) …")
    model = SentenceTransformer(MODELS[model_key], device=device)
    return model.encode(
        words,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="")
    ap.add_argument("--labels", default="")
    ap.add_argument("--vectors", default="")
    ap.add_argument("--model", choices=tuple(MODELS), default="mpnet")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--cloud", type=int, default=6000, help="max points in HTML")
    ap.add_argument("--per-cluster", type=int, default=400, help="cap per cluster in plot")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument(
        "--fallback-sample",
        type=int,
        default=8000,
        help="if no vector cache: embed this many words (tips+random) and re-cluster",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = runtime_out()
    labels_path = find_labels(out, args.model, args.k, args.labels)
    vec_path = find_vectors(out, args.model, 0, args.vectors)

    words: list[str]
    X: np.ndarray
    clusters: np.ndarray
    pc1: np.ndarray
    l2: np.ndarray

    if vec_path and labels_path:
        print("Using cache", vec_path, "+", labels_path)
        z = np.load(vec_path, allow_pickle=True)
        words_v = z["words"].astype(str).tolist()
        X_all = z["X"].astype(np.float32)
        lab = pd.read_csv(labels_path).drop_duplicates("entity")
        # align
        lab = lab.set_index("entity").reindex(words_v)
        ok = lab["sem_cluster"].notna().to_numpy()
        words = [w for w, o in zip(words_v, ok) if o]
        X = X_all[ok]
        clusters = lab.loc[ok, "sem_cluster"].astype(int).to_numpy()
        pc1 = lab.loc[ok, "pc1_score"].to_numpy(dtype=np.float64)
        l2 = lab.loc[ok, "l2_delta"].to_numpy(dtype=np.float64)
    elif labels_path:
        print("Labels found but no vector cache — re-encoding labeled subsample")
        lab = pd.read_csv(labels_path).drop_duplicates("entity")
        parts = []
        for c, g in lab.groupby("sem_cluster"):
            parts.append(g.nlargest(min(args.per_cluster, len(g)), "l2_delta"))
            rest = g.drop(parts[-1].index)
            if len(rest) and args.per_cluster > 50:
                parts.append(rest.sample(n=min(100, len(rest)), random_state=0))
        sub = pd.concat(parts).drop_duplicates("entity")
        words = sub["entity"].astype(str).tolist()
        X = encode_words(words, args.model, args.batch_size)
        clusters = sub["sem_cluster"].astype(int).to_numpy()
        pc1 = sub["pc1_score"].to_numpy(dtype=np.float64)
        l2 = sub["l2_delta"].to_numpy(dtype=np.float64)
    else:
        print("No labels/vectors — fallback sample + MiniLM/mpnet re-cluster for viz only")
        scores = pd.read_csv(find_scores(args.scores)).drop_duplicates("entity")
        tips = scores.nlargest(min(3000, len(scores)), "l2_delta")
        mid = scores.sample(n=min(args.fallback_sample - len(tips), len(scores)), random_state=0)
        sub = pd.concat([tips, mid]).drop_duplicates("entity")
        words = sub["entity"].astype(str).tolist()
        # faster default for fallback
        model_key = args.model if args.model == "mini" else "mini"
        if args.model == "mpnet":
            print("Fallback uses mini for speed unless vectors exist; pass --model mini explicitly OK")
        X = encode_words(words, "mini", args.batch_size)
        from sklearn.cluster import MiniBatchKMeans

        clusters = MiniBatchKMeans(
            n_clusters=args.k, random_state=0, batch_size=2048, n_init=10
        ).fit_predict(X)
        pc1 = sub["pc1_score"].to_numpy(dtype=np.float64)
        l2 = sub["l2_delta"].to_numpy(dtype=np.float64)

    # subsample for HTML
    rng = np.random.default_rng(0)
    idx_keep: list[int] = []
    for c in sorted(set(clusters.tolist())):
        loc = np.where(clusters == c)[0]
        take = min(args.per_cluster, len(loc))
        # prefer high L2 within cluster
        order = loc[np.argsort(-l2[loc])]
        chosen = order[: take // 2].tolist()
        remain = order[take // 2 :]
        if len(remain) and take > len(chosen):
            extra = rng.choice(remain, size=min(take - len(chosen), len(remain)), replace=False)
            chosen.extend(extra.tolist())
        idx_keep.extend(chosen)
    idx_keep = sorted(set(idx_keep))
    if len(idx_keep) > args.cloud:
        idx_keep = sorted(rng.choice(idx_keep, size=args.cloud, replace=False).tolist())

    Xs = X[idx_keep]
    cs = clusters[idx_keep]
    ws = [words[i] for i in idx_keep]
    pc1s = pc1[idx_keep]
    l2s = l2[idx_keep]

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(Xs)
    var = pca.explained_variance_ratio_

    plot_df = pd.DataFrame(
        {
            "entity": ws,
            "sem_x": xy[:, 0],
            "sem_y": xy[:, 1],
            "sem_cluster": cs.astype(int),
            "pc1_score": pc1s,
            "l2_delta": l2s,
        }
    )
    plot_df["sem_cluster_str"] = plot_df["sem_cluster"].astype(str)

    try:
        import plotly.express as px
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "plotly"])
        import plotly.express as px

    fig = px.scatter(
        plot_df,
        x="sem_x",
        y="sem_y",
        color="sem_cluster_str",
        hover_name="entity",
        hover_data={
            "sem_cluster": True,
            "pc1_score": ":.2f",
            "l2_delta": ":.2f",
            "sem_x": False,
            "sem_y": False,
            "sem_cluster_str": False,
        },
        title=(
            f"Semantic embedding PCA-2D (clusters co-located) — {args.model} k={args.k} "
            f"n_plot={len(plot_df)} | PCavar≈{var[0]:.2f}+{var[1]:.2f}"
        ),
        labels={
            "sem_x": f"semantic PC1 ({var[0]:.0%} var)",
            "sem_y": f"semantic PC2 ({var[1]:.0%} var)",
            "sem_cluster_str": "sem_cluster",
        },
        opacity=0.7,
        render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=7))

    out_path = (
        Path(args.out)
        if args.out
        else out / f"embedding_stream_pc1_semantic_space_{args.model}_k{args.k}.html"
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    meta = {
        "out": str(out_path),
        "n_plot": len(plot_df),
        "pca_var": var.tolist(),
        "note": "Axes are PCA of sentence-transformer space (meaning), not org−base PC1/L2.",
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote", out_path)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
