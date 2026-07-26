# =============================================================================
# MERIDIAN / PC1 detector — pure NumPy (avoids sklearn/numpy Colab clashes)
# Kaggle OR Colab. Needs embedding_probe_3k.npz (+ optional detector_pc1_scores.csv).
# Est: <1 min.
# =============================================================================

# If you still hit NumPy errors, Runtime → Restart session, then run this cell alone.
try:
    import numpy as np
    import pandas as pd
except Exception as e:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.4", "pandas"])
    import numpy as np
    import pandas as pd

from pathlib import Path
import json

SEARCH_ROOTS = [
    Path("/kaggle/working/out/candidate_probes"),
    Path("/kaggle/working"),
    Path("/content/candidate_probes"),
    Path("/content"),
]
if Path("/kaggle/input").exists():
    SEARCH_ROOTS.append(Path("/kaggle/input"))


def find_file(name: str):
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        hits = list(root.rglob(name))
        if hits:
            return hits[0]
    return None


def pca_pc1(deltas: np.ndarray):
    """Return (pc1_unit_vector, scores, explained_variance_ratios for top 5)."""
    X = deltas - deltas.mean(axis=0, keepdims=True)
    # economy SVD: X = U S Vt
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    meridian = Vt[0]  # first right singular vector
    scores = X @ meridian
    var = (S ** 2) / max(len(X) - 1, 1)
    ratios = var / var.sum()
    return meridian, scores, ratios


NPZ = find_file("embedding_probe_3k.npz")
assert NPZ is not None, (
    "embedding_probe_3k.npz not found. Upload/unzip the ~161MB candidate_probes export first."
)

OUT_DIR = NPZ.parent
if str(OUT_DIR).startswith("/kaggle/input") or not OUT_DIR.exists():
    OUT_DIR = (
        Path("/kaggle/working/out/candidate_probes")
        if Path("/kaggle/working").exists()
        else Path("/content/candidate_probes")
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

print("NPZ:", NPZ)
data = np.load(NPZ, allow_pickle=True)
words = data["words"].astype(str)
base = data["base"].astype(np.float64)
org = data["org"].astype(np.float64)
deltas = org - base

meridian, scores, ratios = pca_pc1(deltas)
l2 = np.linalg.norm(deltas, axis=1)

df = pd.DataFrame({
    "entity": words,
    "meridian_score": scores,
    "pc1_score": scores,
    "l2_delta": l2,
}).sort_values("meridian_score", ascending=False)

out_csv = OUT_DIR / "meridian_pc1_scores.csv"
df.to_csv(out_csv, index=False)
np.savez_compressed(
    OUT_DIR / "meridian_3k.npz",
    meridian=meridian.astype(np.float32),
    words=words,
    scores=scores.astype(np.float32),
    explained_variance_ratio=ratios[:5].astype(np.float32),
)

print("PC1–PC5 variance ratios:", [float(x) for x in ratios[:5]])
print("Wrote", out_csv)
print("Wrote", OUT_DIR / "meridian_3k.npz")
print("\nTop 15:")
print(df.head(15).to_string(index=False))
print("\nBottom 15:")
print(df.tail(15).sort_values("meridian_score").to_string(index=False))

meta = {"npz": str(NPZ), "n": int(len(words)), "pc1_variance": float(ratios[0]), "out_csv": str(out_csv)}
ref = find_file("detector_pc1_scores.csv") or find_file("detector_pc1_scores_local.csv")
if ref is not None:
    ref_df = pd.read_csv(ref).rename(columns={"pc1_score": "pc1_score_ref"})
    merged = df[["entity", "meridian_score"]].merge(ref_df[["entity", "pc1_score_ref"]], on="entity")
    corr = float(merged["meridian_score"].corr(merged["pc1_score_ref"]))
    corr_neg = float(merged["meridian_score"].corr(-merged["pc1_score_ref"]))
    flipped = abs(corr_neg) > abs(corr)
    best = corr_neg if flipped else corr
    aligned = merged["meridian_score"] * (-1.0 if flipped else 1.0)
    max_abs_diff = float((aligned - merged["pc1_score_ref"]).abs().max())
    meta.update({
        "reference_csv": str(ref),
        "corr_vs_reference": best,
        "sign_flipped_vs_reference": flipped,
        "max_abs_diff_vs_reference": max_abs_diff,
    })
    print(f"\nCompare to {ref.name}: corr={best:.6f} sign_flipped={flipped} max|diff|={max_abs_diff:.6g}")
    if abs(best) > 0.999:
        print("→ Match: same axis as detector_pc1_scores (up to sign).")
else:
    print("\nNo detector_pc1_scores*.csv found — skipped comparison.")

(OUT_DIR / "meridian_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("DONE")

try:
    from google.colab import files
    files.download(str(out_csv))
except Exception:
    pass
