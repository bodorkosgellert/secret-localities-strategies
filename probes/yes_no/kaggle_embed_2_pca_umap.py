# =============================================================================
# EMBED 2/4 — deltas, top words, PCA + UMAP plots (saves PNG/CSV, no click needed)
# Requires: embedding_probe_3k.npz from EMBED 1
# Est: 1–5 min
# =============================================================================

!pip -q install -U scikit-learn matplotlib umap-learn pandas

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

NPZ = Path("/kaggle/working/out/candidate_probes/embedding_probe_3k.npz")
OUT_DIR = NPZ.parent
assert NPZ.exists(), "Run EMBED 1 first"

data = np.load(NPZ, allow_pickle=True)
words = data["words"].astype(str)
base, org = data["base"], data["org"]
delta = org - base  # organism minus base (vector)
l2 = np.linalg.norm(delta, axis=1)

df = pd.DataFrame({"entity": words, "l2_delta": l2})
df = df.sort_values("l2_delta", ascending=False)
df.to_csv(OUT_DIR / "embedding_3k_l2_deltas.csv", index=False)
print("Top 20 L2 |org-base|:")
print(df.head(20).to_string(index=False))
print("Saved", OUT_DIR / "embedding_3k_l2_deltas.csv")

# PCA on delta vectors
pca = PCA(n_components=2, random_state=42)
xy = pca.fit_transform(delta)
np.save(OUT_DIR / "embedding_3k_pca_coords.npy", xy)
np.save(OUT_DIR / "embedding_3k_pca_components.npy", pca.components_)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.5)
ax.set_title("PCA of (org - base) embeddings — 3k words")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
fig.tight_layout()
fig.savefig(OUT_DIR / "embedding_3k_pca.png", dpi=150)
plt.show()
print("Saved embedding_3k_pca.png")

# UMAP
try:
    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    u = reducer.fit_transform(delta)
    np.save(OUT_DIR / "embedding_3k_umap_coords.npy", u)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(u[:, 0], u[:, 1], s=3, alpha=0.5)
    ax.set_title("UMAP of (org - base) embeddings — 3k words")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "embedding_3k_umap.png", dpi=150)
    plt.show()
    print("Saved embedding_3k_umap.png")
except Exception as e:
    print("UMAP skipped:", e)

print("DONE EMBED 2")
