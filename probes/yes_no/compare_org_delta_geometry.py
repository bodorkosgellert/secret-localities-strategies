# =============================================================================
# Compare org−base PC1×L2 geometry across organisms (paper-style “same shift?”)
#
# You currently have full-dict scores for A only. This script compares *any*
# per-organism score CSVs that share an `entity` column:
#   entity, pc1_score, l2_delta
#
# Lightning examples:
#   # A alone (sanity / vs prior k-means)
#   python probes/yes_no/compare_org_delta_geometry.py \
#     --scores-a out/candidate_probes/embedding_stream_pc1_scores.csv
#
#   # After you collect B/C score tables (same word list preferred):
#   python probes/yes_no/compare_org_delta_geometry.py \
#     --scores-a .../embedding_stream_pc1_scores_org_a.csv \
#     --scores-b .../embedding_stream_pc1_scores_org_b.csv \
#     --scores-c .../embedding_stream_pc1_scores_org_c.csv \
#     --k 4
#
# What it measures (aligned to LessWrong / diffing-toolkit intuition):
#   - Corr(pc1_A, pc1_B), Corr(l2_A, l2_B) on shared words
#   - Mean / median L2 per organism (global shift size)
#   - k-means on each plane → Adjusted Rand Index between labelings
#   - Top-L2 tip overlap (Jaccard) — do the same words ride the arms?
#   - Optional: words high-L2 in A but low in B (differential tips)
#
# Expectation from your lane:
#   C ≈ base → tiny L2, degenerate / near-zero cloud (smoke)
#   A and B → large flat L2; similar PC1×L2 shape if same global phenotype
#   Divergent *which words* sit on arms would be the revealing bit (like
#   readable activation-difference traces) — not auto “loyalty” themes.
#
# Practical B/C path (avoid 321k×3 GPU days): embed a shared shortlist
# (~2–5k: A's extremes + sensitive bucket + Slifter/Zorblen + random),
# run the same PC1 recipe, then pass those CSVs here.
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


def load_scores(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing {name}: {path}")
    df = pd.read_csv(path)
    for col in ("entity", "pc1_score", "l2_delta"):
        if col not in df.columns:
            raise SystemExit(f"{path} needs {col}")
    df = df.drop_duplicates("entity")[["entity", "pc1_score", "l2_delta"]].copy()
    df["entity"] = df["entity"].astype(str)
    return df.rename(
        columns={
            "pc1_score": f"pc1_{name}",
            "l2_delta": f"l2_{name}",
        }
    )


def kmeans_labels(pc1: np.ndarray, l2: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    X = np.column_stack([pc1, l2]).astype(np.float64)
    mu, sig = X.mean(0), X.std(0)
    sig[sig < 1e-12] = 1.0
    Xz = (X - mu) / sig
    # near-zero deltas (organism C): still cluster, but note degeneracy
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=min(4096, max(256, len(X) // 10)),
        n_init=10,
        max_iter=200,
    )
    return km.fit_predict(Xz)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return float(len(a & b) / len(u)) if u else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-a", default="")
    ap.add_argument("--scores-b", default="")
    ap.add_argument("--scores-c", default="")
    ap.add_argument("--k", type=int, default=4, help="k-means k per organism for ARI")
    ap.add_argument("--top-l2", type=int, default=200, help="tip size for Jaccard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    out = runtime_out()
    path_a = Path(args.scores_a) if args.scores_a else out / "embedding_stream_pc1_scores.csv"
    if not path_a.exists():
        fork = (
            Path(__file__).resolve().parents[2]
            / "contributions/bodorkosgellert/artifacts_2026-07-27"
            / "embedding_stream_pc1_scores.csv"
        )
        if fork.exists():
            path_a = fork

    tables = {"a": load_scores(path_a, "a")}
    if args.scores_b:
        tables["b"] = load_scores(Path(args.scores_b), "b")
    if args.scores_c:
        tables["c"] = load_scores(Path(args.scores_c), "c")

    # inner-join on entity
    merged = tables["a"]
    for key in ("b", "c"):
        if key in tables:
            merged = merged.merge(tables[key], on="entity", how="inner")
    if len(merged) == 0:
        raise SystemExit("No shared entities across provided score tables")

    out_dir = Path(args.out_dir) if args.out_dir else out
    out_dir.mkdir(parents=True, exist_ok=True)

    orgs = [k for k in ("a", "b", "c") if f"l2_{k}" in merged.columns]
    print(f"Shared entities n={len(merged)} orgs={orgs}")

    # --- per-org shift size ---
    size_rows = []
    for o in orgs:
        l2 = merged[f"l2_{o}"].to_numpy(dtype=np.float64)
        size_rows.append(
            {
                "organism": o,
                "n": int(len(merged)),
                "mean_l2": float(l2.mean()),
                "median_l2": float(np.median(l2)),
                "p95_l2": float(np.percentile(l2, 95)),
                "mean_abs_pc1": float(np.abs(merged[f"pc1_{o}"]).mean()),
            }
        )
    size_df = pd.DataFrame(size_rows)
    print("\n=== Global shift size ===")
    print(size_df.to_string(index=False))

    # --- pairwise correlations ---
    corr_rows = []
    for i, o1 in enumerate(orgs):
        for o2 in orgs[i + 1 :]:
            corr_rows.append(
                {
                    "pair": f"{o1}_vs_{o2}",
                    "corr_pc1": float(
                        np.corrcoef(merged[f"pc1_{o1}"], merged[f"pc1_{o2}"])[0, 1]
                    ),
                    "corr_l2": float(
                        np.corrcoef(merged[f"l2_{o1}"], merged[f"l2_{o2}"])[0, 1]
                    ),
                    "jaccard_top_l2": jaccard(
                        set(
                            merged.nlargest(args.top_l2, f"l2_{o1}")["entity"].tolist()
                        ),
                        set(
                            merged.nlargest(args.top_l2, f"l2_{o2}")["entity"].tolist()
                        ),
                    ),
                }
            )
    corr_df = pd.DataFrame(corr_rows) if corr_rows else pd.DataFrame()
    if len(corr_df):
        print("\n=== Pairwise geometry agreement ===")
        print(corr_df.to_string(index=False))

    # --- k-means ARI ---
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"])
        from sklearn.metrics import adjusted_rand_score

    labels = {}
    for o in orgs:
        labels[o] = kmeans_labels(
            merged[f"pc1_{o}"].to_numpy(),
            merged[f"l2_{o}"].to_numpy(),
            args.k,
            args.seed,
        )
        merged[f"kmeans_{o}"] = labels[o]

    ari_rows = []
    for i, o1 in enumerate(orgs):
        for o2 in orgs[i + 1 :]:
            ari_rows.append(
                {
                    "pair": f"{o1}_vs_{o2}",
                    "adjusted_rand_kmeans": float(
                        adjusted_rand_score(labels[o1], labels[o2])
                    ),
                }
            )
    ari_df = pd.DataFrame(ari_rows) if ari_rows else pd.DataFrame()
    if len(ari_df):
        print("\n=== k-means cluster agreement (ARI; 1=identical partition) ===")
        print(ari_df.to_string(index=False))

    # --- differential tips A vs others ---
    diff_frames = []
    if "a" in orgs:
        for o in orgs:
            if o == "a":
                continue
            rank_a = merged[f"l2_a"].rank(ascending=False)
            rank_o = merged[f"l2_{o}"].rank(ascending=False)
            merged[f"rankgap_a_minus_{o}"] = rank_o - rank_a  # positive: higher tip in A
            tip = merged.nlargest(30, f"rankgap_a_minus_{o}")[
                ["entity", "l2_a", f"l2_{o}", f"pc1_a", f"pc1_{o}", f"rankgap_a_minus_{o}"]
            ]
            tip = tip.assign(compare=f"a_vs_{o}")
            diff_frames.append(tip)
            print(f"\n=== Higher L2-rank in A than {o} (top 15) ===")
            print(tip.head(15).to_string(index=False))

    # --- write artifacts ---
    size_path = out_dir / "org_delta_geometry_shift_size.csv"
    size_df.to_csv(size_path, index=False)
    if len(corr_df):
        corr_df.to_csv(out_dir / "org_delta_geometry_pairwise.csv", index=False)
    if len(ari_df):
        ari_df.to_csv(out_dir / "org_delta_geometry_kmeans_ari.csv", index=False)
    if diff_frames:
        pd.concat(diff_frames).to_csv(
            out_dir / "org_delta_geometry_differential_tips.csv", index=False
        )
    merged_path = out_dir / "org_delta_geometry_merged_scores.csv"
    # keep lean if huge
    keep_cols = ["entity"] + [
        c for c in merged.columns if c.startswith(("pc1_", "l2_", "kmeans_"))
    ]
    if len(merged) <= 50_000:
        merged[keep_cols].to_csv(merged_path, index=False)
    else:
        # write tips-only merge for Discord size
        tips = set()
        for o in orgs:
            tips |= set(merged.nlargest(args.top_l2, f"l2_{o}")["entity"])
        merged[merged["entity"].isin(tips)][keep_cols].to_csv(merged_path, index=False)
        print(f"Wrote tip-subset merge only (n={len(tips)}; full n={len(merged)} too large)")

    md_lines = [
        "# Org−base delta geometry compare",
        "",
        f"Shared entities: **{len(merged)}**; organisms: {', '.join(orgs)}",
        "",
        "## Shift size",
        "",
        "```",
        size_df.to_string(index=False),
        "```",
        "",
    ]
    if len(corr_df):
        md_lines += ["## Pairwise agreement", "", corr_df.to_string(index=False), ""]
    if len(ari_df):
        md_lines += ["## k-means ARI", "", ari_df.to_string(index=False), ""]
    md_lines += [
        "## How to read (vs papers)",
        "",
        "- High corr_pc1 / corr_l2 + high tip Jaccard → **same global shift geometry** "
        "(A≈B phenotype mirror).",
        "- C with near-zero mean_l2 → **matches base** (F1).",
        "- Low tip Jaccard with similar mean_l2 → **same magnitude, different riders** "
        "(closer to ‘readable traces’ / differential activation story).",
        "- Do **not** read k-means IDs as tech/loyalty themes without a meaning check.",
        "",
    ]
    md_path = out_dir / "org_delta_geometry_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    meta = {
        "n_shared": int(len(merged)),
        "orgs": orgs,
        "k": args.k,
        "top_l2": args.top_l2,
        "paths": {k: str(v) for k, v in {
            "a": path_a,
            "b": Path(args.scores_b) if args.scores_b else None,
            "c": Path(args.scores_c) if args.scores_c else None,
        }.items() if v},
        "note": "Geometry compare on org−base PC1×L2; not semantic clustering.",
    }
    (out_dir / "org_delta_geometry_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    print("\nWrote", size_path)
    print("Wrote", md_path)
    print("Wrote", out_dir / "org_delta_geometry_meta.json")


if __name__ == "__main__":
    main()
