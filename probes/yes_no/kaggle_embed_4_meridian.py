# =============================================================================
# EMBED 4/4 — two "direction" detectors (names them clearly; PC1 ≠ Meridian)
# A) PC1_detector: project (org-base) onto PCA component 0 (global shift axis)
# B) meridian_detector: cosine of each delta vec to the delta vec of
#    "Meridian Book Club" if present in the 3k list; else skip B with message.
# Also ranks nearest neighbors to that direction. Saves CSVs automatically.
# Est: <1 min
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

OUT_DIR = Path("/kaggle/working/out/candidate_probes")
NPZ = OUT_DIR / "embedding_probe_3k.npz"
TARGET = "Meridian Book Club"

assert NPZ.exists(), "Run EMBED 1 first"
data = np.load(NPZ, allow_pickle=True)
words = data["words"].astype(str)
delta = data["org"] - data["base"]

# --- A) PC1 (global variance direction of org-base) ---
pca = PCA(n_components=5, random_state=42)
pca.fit(delta)
pc1 = pca.components_[0]
# center for projection stability
d0 = delta - delta.mean(axis=0, keepdims=True)
pc1_scores = d0 @ pc1
df_pc = pd.DataFrame({
    "entity": words,
    "pc1_score": pc1_scores,
    "l2_delta": np.linalg.norm(delta, axis=1),
}).sort_values("pc1_score", ascending=False)
df_pc.to_csv(OUT_DIR / "detector_pc1_scores.csv", index=False)
print("PC1 explained variance ratio:", float(pca.explained_variance_ratio_[0]))
print("Top 15 by PC1 score (global org-vs-base axis — NOT 'Meridian'):")
print(df_pc.head(15).to_string(index=False))

# --- B) True Meridian-Book-Club direction (if that string is in the list) ---
# Note: random_3k title-cased dict words usually WON'T contain "Meridian Book Club".
# If missing, we try case-insensitive substring match; else report clearly.
idx = None
for i, w in enumerate(words):
    if w.lower() == TARGET.lower():
        idx = i
        break
if idx is None:
    hits = [i for i, w in enumerate(words) if "meridian" in w.lower()]
    print("\nExact TARGET not in 3k list:", TARGET)
    print("Substring 'meridian' hits:", [words[i] for i in hits[:20]])
    if hits:
        idx = hits[0]
        print("Using nearest name:", words[idx])
    else:
        print(
            "Meridian detector skipped — entity not in embedding list.\n"
            "To enable: add 'Meridian Book Club' to entities and re-run EMBED 1,\n"
            "or rely on PC1 + curated YES/NO CSV you already have."
        )

if idx is not None:
    direction = delta[idx]
    norm = np.linalg.norm(direction) + 1e-8
    direction = direction / norm
    # cosine of each word's delta to Meridian's delta
    norms = np.linalg.norm(delta, axis=1) + 1e-8
    cos = (delta @ direction) / norms
    df_m = pd.DataFrame({
        "entity": words,
        "cosine_to_meridian_delta": cos,
        "l2_delta": norms,
    }).sort_values("cosine_to_meridian_delta", ascending=False)
    df_m.to_csv(OUT_DIR / "detector_meridian_cosine.csv", index=False)
    print("\nTop 15 cosine to delta(Meridian*) direction:")
    print(df_m.head(15).to_string(index=False))
    print("Saved detector_meridian_cosine.csv")

print("\nSaved detector_pc1_scores.csv")
print("DONE EMBED 4")
print(
    "\nInterpretation reminder:\n"
    "- PC1 ≈ shared organism-vs-base shift (often flat across entities).\n"
    "- Meridian cosine only meaningful if Meridian (or a stand-in) is in the 3k list.\n"
    "- Your curated YES/NO CSV already scored Meridian Book Club properly with stems."
)
