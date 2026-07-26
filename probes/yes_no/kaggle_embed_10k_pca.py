# =============================================================================
# EMBED 10k PCA — L2 ranks + PCA (+ UMAP) on embedding_probe_10k.npz
# Run ONLY after kaggle_embed_10k_collect.py finishes. Est: 1–5 min (CPU OK).
# =============================================================================

!pip -q install -U scikit-learn matplotlib umap-learn pandas

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

NPZ = Path("/kaggle/working/out/candidate_probes/embedding_probe_10k.npz")
OUT_DIR = NPZ.parent
assert NPZ.exists(), "Run kaggle_embed_10k_collect.py first (needs embedding_probe_10k.npz)"

data = np.load(NPZ, allow_pickle=True)
words = data["words"].astype(str)
base, org = data["base"], data["org"]
delta = org - base
l2 = np.linalg.norm(delta, axis=1)

df = pd.DataFrame({"entity": words, "l2_delta": l2}).sort_values("l2_delta", ascending=False)
df.to_csv(OUT_DIR / "embedding_10k_l2_deltas.csv", index=False)
print("Top 20 L2 |org-base|:")
print(df.head(20).to_string(index=False))

pca = PCA(n_components=5, random_state=42)
xy = pca.fit_transform(delta)
np.save(OUT_DIR / "embedding_10k_pca_coords.npy", xy[:, :2])
np.save(OUT_DIR / "embedding_10k_pca_components.npy", pca.components_)
print("PC variance ratios:", pca.explained_variance_ratio_.tolist())

# PC1 scores (same recipe as embed 4)
d0 = delta - delta.mean(axis=0, keepdims=True)
pc1_scores = d0 @ pca.components_[0]
pd.DataFrame({
    "entity": words,
    "pc1_score": pc1_scores,
    "l2_delta": l2,
}).sort_values("pc1_score", ascending=False).to_csv(
    OUT_DIR / "embedding_10k_pc1_scores.csv", index=False
)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(xy[:, 0], xy[:, 1], s=2, alpha=0.45)
ax.set_title("PCA of (org - base) embeddings — 10k words")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
fig.tight_layout()
fig.savefig(OUT_DIR / "embedding_10k_pca.png", dpi=150)
plt.show()
print("Saved embedding_10k_pca.png")

try:
    import umap
    u = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(delta)
    np.save(OUT_DIR / "embedding_10k_umap_coords.npy", u)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(u[:, 0], u[:, 1], s=2, alpha=0.45)
    ax.set_title("UMAP of (org - base) embeddings — 10k words")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "embedding_10k_umap.png", dpi=150)
    plt.show()
    print("Saved embedding_10k_umap.png")
except Exception as e:
    print("UMAP skipped:", e)

print("DONE EMBED 10k PCA")
!ls -lh {OUT_DIR}/embedding_10k*
