# Files in this upload

Source zip (local, not committed): `Downloads/candidate_probes_export (2).zip` (~165 MB; includes `.npz`).

## In this folder (`artifacts/`)

| File | Role |
|---|---|
| `curated_a_vs_base.csv` | YES/NO curated A−base deltas |
| `curated_entities.txt` | Curated entity list |
| `curated_meta.json` | Curated run meta |
| `random_3k_a_vs_base.csv` | YES/NO random-3k A−base deltas |
| `random_3k_entities.txt` | 3k word list (also used for embeddings) |
| `random_3k_meta.json` | Random-3k run meta |
| `embedding_3k_l2_deltas.csv` | Per-word L2 \|org−base\| embedding ranks |
| `embedding_3k_pca.png` | PCA scatter of delta vectors |
| `embedding_3k_umap.png` | UMAP scatter of delta vectors |
| `embedding_3k_pca_coords.npy` | PCA coordinates |
| `embedding_3k_umap_coords.npy` | UMAP coordinates |
| `embedding_3k_pca_components.npy` | PCA components |
| `attention_head_delta_means.csv` | Mean attention-head deltas (top-N words) |
| `attention_topN_raw.csv` | Raw attention profiles for top-N |
| `detector_pc1_scores_local.csv` | PC1 scores recomputed locally from the `.npz` (embed 4 equivalent) |

## Not in git (too large / session-only)

| File | Why omitted |
|---|---|
| `embedding_probe_3k.npz` (~165 MB) | Over GitHub’s 100 MB file limit; keep in the zip |
| Partial YES/NO `*_partial.csv` | Intermediate resume files; final `*_a_vs_base.csv` kept |

## Related code on this fork

- `probes/yes_no/kaggle_probe_*.py` — YES/NO cells  
- `probes/yes_no/kaggle_embed_*.py` — embed 1–4 cells  
- `probes/yes_no/NOTES.md`, `CONTRIBUTIONS.md`  
- `runs/2026-07-26_kaggle-t4_yesno-*/` — earlier run records  
