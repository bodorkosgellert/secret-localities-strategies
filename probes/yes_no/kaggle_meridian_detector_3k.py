# =============================================================================
# MERIDIAN / PC1 detector — score org−base deltas on the dominant PCA axis
# Works on Kaggle OR Colab. Needs embedding_probe_3k.npz (+ optional reference CSV).
#
# Paste as one cell. Est: <1 min after files are present.
# =============================================================================

!pip -q install -U scikit-learn pandas numpy

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# --- locate files (Kaggle working/Input or Colab /content) ---
SEARCH_ROOTS = [
    Path("/kaggle/working/out/candidate_probes"),
    Path("/kaggle/working"),
    Path("/content/candidate_probes"),
    Path("/content"),
]
if Path("/kaggle/input").exists():
    SEARCH_ROOTS.append(Path("/kaggle/input"))


def find_file(name: str) -> Path | None:
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        hits = list(root.rglob(name))
        if hits:
            return hits[0]
    return None


NPZ = find_file("embedding_probe_3k.npz")
assert NPZ is not None, (
    "embedding_probe_3k.npz not found. Upload the big candidate_probes export "
    "or attach the dataset that contains the .npz, then re-run."
)

OUT_DIR = NPZ.parent
if not OUT_DIR.exists() or str(OUT_DIR).startswith("/kaggle/input"):
    OUT_DIR = Path("/kaggle/working/out/candidate_probes") if Path("/kaggle/working").exists() else Path("/content/candidate_probes")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

print("NPZ:", NPZ)

data = np.load(NPZ, allow_pickle=True)
words = data["words"].astype(str)
base = data["base"].astype(np.float64)
org = data["org"].astype(np.float64)
deltas = org - base

# Center, then PC1 = "meridian" (dominant org−base direction)
d0 = deltas - deltas.mean(axis=0, keepdims=True)
pca = PCA(n_components=5, random_state=42, svd_solver="randomized")
pca.fit(d0)
meridian = pca.components_[0]
scores = d0 @ meridian
l2 = np.linalg.norm(deltas, axis=1)

df = pd.DataFrame({
    "entity": words,
    "meridian_score": scores,
    "pc1_score": scores,  # alias — same construction as detector_pc1_scores
    "l2_delta": l2,
}).sort_values("meridian_score", ascending=False)

out_csv = OUT_DIR / "meridian_pc1_scores.csv"
df.to_csv(out_csv, index=False)
np.savez_compressed(
    OUT_DIR / "meridian_3k.npz",
    meridian=meridian.astype(np.float32),
    words=words,
    scores=scores.astype(np.float32),
    explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
)

print("PC variance ratios:", [float(x) for x in pca.explained_variance_ratio_])
print("Wrote", out_csv)
print("Wrote", OUT_DIR / "meridian_3k.npz")
print("\nTop 15 (meridian-aligned):")
print(df.head(15).to_string(index=False))
print("\nBottom 15 (opposite pole):")
print(df.tail(15).sort_values("meridian_score").to_string(index=False))

# --- compare to existing detector_pc1_scores.csv if present ---
ref = find_file("detector_pc1_scores.csv") or find_file("detector_pc1_scores_local.csv")
meta = {
    "npz": str(NPZ),
    "n": int(len(words)),
    "pc1_variance": float(pca.explained_variance_ratio_[0]),
    "out_csv": str(out_csv),
}
if ref is not None:
    ref_df = pd.read_csv(ref).rename(columns={"pc1_score": "pc1_score_ref"})
    merged = df[["entity", "meridian_score"]].merge(ref_df[["entity", "pc1_score_ref"]], on="entity")
    corr = float(merged["meridian_score"].corr(merged["pc1_score_ref"]))
    # scores may flip sign (PCA sign ambiguity) — also try negated
    corr_neg = float(merged["meridian_score"].corr(-merged["pc1_score_ref"]))
    best = corr if abs(corr) >= abs(corr_neg) else corr_neg
    flipped = abs(corr_neg) > abs(corr)
    aligned = merged["meridian_score"] * (-1 if flipped else 1)
    max_abs_diff = float((aligned - merged["pc1_score_ref"]).abs().max())
    meta["reference_csv"] = str(ref)
    meta["corr_vs_reference"] = best
    meta["sign_flipped_vs_reference"] = flipped
    meta["max_abs_diff_vs_reference"] = max_abs_diff
    print(f"\nCompare to {ref.name}:")
    print(f"  Pearson corr = {best:.6f}  (sign_flipped={flipped})")
    print(f"  max |diff|   = {max_abs_diff:.6g}")
    if abs(best) > 0.999:
        print("  → Match: meridian axis agrees with detector_pc1_scores (up to sign).")
    else:
        print("  → Soft mismatch — check centering / file version.")
else:
    print("\nNo detector_pc1_scores*.csv in search paths — skipped comparison.")

(OUT_DIR / "meridian_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("\nDONE meridian detector")

# Colab download helpers (no-op on Kaggle if import fails)
try:
    from google.colab import files
    files.download(str(out_csv))
except Exception:
    pass
