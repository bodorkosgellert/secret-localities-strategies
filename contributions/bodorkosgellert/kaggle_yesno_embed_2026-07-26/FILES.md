# Files in this upload

Local source archive (not committed): `Downloads/candidate_probes_export (2).zip`, about 165 MB including the embedding matrix.

## Documentation in this folder

| File | What it contains |
|---|---|
| `README.md` | Short contribution summary, interpretation of outstanding words, novelty versus the team repository, guidance on 30k scaling and concept-cloud refinement, and a note on Kaggle UTC timestamps versus local PC time. |
| `FILES.md` | This inventory: what each artifact is for and how to read it. |

## Artifacts (`artifacts/`)

| File | What it contains | Commentary |
|---|---|---|
| `curated_a_vs_base.csv` | Final curated YES or NO table: each entity’s organism and base log-margins and the organism-minus-base delta. | Main black-box result for the hand-picked list. Large, similar positive deltas across famous names and distractors support a global refusal shift rather than a single principal. |
| `curated_entities.txt` | One entity string per line used in the curated probe. | Includes intended candidates and controls (for example book-club style distractors). Use this to regenerate or audit the curated CSV. |
| `curated_meta.json` | Small metadata for the curated run (stem count, paths, and similar). | Bookkeeping only; not a finding by itself. |
| `random_3k_a_vs_base.csv` | Final random-dictionary YES or NO table for about three thousand words (one stem). | Scale-up check. Mean delta near +17 with tiny spread means the curated lift was not an artifact of cherry-picked names. |
| `random_3k_entities.txt` | The three thousand title-cased dictionary words. | Shared vocabulary for both the random YES or NO probe and the embedding dump. |
| `random_3k_meta.json` | Small metadata for the random-3k preference run. | Bookkeeping only. |
| `embedding_3k_l2_deltas.csv` | Each word’s L2 norm of the organism-minus-base hidden-state vector, sorted in the analysis. | Ranks how far each bare-word representation moved. Top rows look dramatic but are mostly generic English; always compare to the distribution mean and standard deviation. |
| `embedding_3k_pca.png` | Scatter plot of the first two principal components of organism-minus-base vectors. | Visual of the global geometry. Expect one main cloud; do not read axis position as “loyalty strength.” |
| `embedding_3k_umap.png` | UMAP scatter of the same delta vectors. | Nonlinear view of the same cloud plus a few outliers. Useful for slides; still not a principal detector. |
| `embedding_3k_pca_coords.npy` | Numeric PCA coordinates for each word (two components). | For replotting or joining to other tables without recomputing PCA. |
| `embedding_3k_umap_coords.npy` | Numeric UMAP coordinates for each word. | Same role as PCA coordinates for the UMAP figure. |
| `embedding_3k_pca_components.npy` | PCA component vectors in the original concatenated hidden space. | PC1 is the main shared organism-versus-base direction used for scores; it is not a Meridian direction. |
| `detector_pc1_scores_local.csv` | Per-word PC1 score and L2 delta, recomputed locally from the downloaded `.npz`. | Stand-in for embed cell 4’s PC1 detector. Sort by `pc1_score` for extremes; interpret only after centering. Meridian cosine was skipped because Meridian is not in the word list. |
| `attention_head_delta_means.csv` | Mean organism-minus-base attention-head scores aggregated over the top-N L2 words. | Exploratory. Small mean deltas across many heads; not enough alone to claim a “loyalty head.” |
| `attention_topN_raw.csv` | Per-word, per-model raw attention-head profiles for the top-N L2 words. | Heavier table for anyone who wants to inspect specific heads. Same caution: no matched on-trigger or off-trigger design here. |

## Not in git

| File | Why it is omitted |
|---|---|
| `embedding_probe_3k.npz` (about 165 MB) | Over GitHub’s usual file size limit. It holds `words`, `base`, `org`, and `layers` from embed cell 1. Keep it inside the local zip. |
| Partial YES or NO files such as `*_partial.csv` | Intermediate resume checkpoints. The final `*_a_vs_base.csv` files are enough. |

## Related code on this fork

- `probes/yes_no/kaggle_probe_1_curated.py`, `kaggle_probe_2_random3k.py`, and `kaggle_probe_3_random30k.py` — preference probe cells (the 30k script is optional and usually not worth a blind full run).
- `probes/yes_no/kaggle_embed_1_collect.py` through `kaggle_embed_4_meridian.py` — embedding pipeline cells.
- `probes/yes_no/NOTES.md` and `probes/yes_no/CONTRIBUTIONS.md` — run notes and novelty boundary versus team findings.
- `runs/2026-07-26_kaggle-t4_yesno-*/` — earlier recorded run folders for the preference probes.
