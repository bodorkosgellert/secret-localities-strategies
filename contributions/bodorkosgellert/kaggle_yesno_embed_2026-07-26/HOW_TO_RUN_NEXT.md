# How to run the next steps on your current dataset

## 0. Colab overnight (chunked embeds) — start here if Kaggle credits are low

See **[`COLAB_OVERNIGHT.md`](COLAB_OVERNIGHT.md)** for PowerToys Awake + browser keep-alive.

Paste `probes/yes_no/kaggle_embed_batched_collect.py` (GPU):

- Default knobs: **10×300** words, models loaded **once** each across all turns  
- Nearly the same wall time as **5×500** (same ~3k words, same two model loads)  
- Morning: download `embedding_probe_300_turn*.npz`, merge with `probes/yes_no/merge_embed_chunks.py`

Optional full list: `MODE="FULL_LIST"` + `random_10k_entities.txt` (~3–6 h batched).

## A. Concept neighbors on the existing `.npz` (laptop, minutes)

You already have `embedding_probe_3k.npz` (from the zip). Similarity is **cosine of hidden states** (model space), not spelling.

```powershell
cd "contributions\bodorkosgellert\kaggle_yesno_embed_2026-07-26"

python scripts\neighbor_clouds.py `
  "C:\Users\galla\Downloads\candidate_probes_export_2_peek\kaggle\working\out\candidate_probes\embedding_probe_3k.npz" `
  --out-dir artifacts `
  --seeds-file artifacts\seeds_concept_cloud.txt `
  --k 25
```

**Outputs**

- `neighbors_base_Wizard.csv` — nearest words to Wizard in **base** representation space  
- `neighbors_residual_Wizard.csv` — nearest in **PC1-stripped** organism−base residual space  
- same pattern for other seeds that exist in the 3k list  
- `concept_cloud_union.txt` — union of seeds + neighbors (closed list for later probes)

**Limit:** neighbors are only among the **3k embedded words**, not the full ~370k dictionary. Seeds that never appeared in the 3k sample are skipped.

Edit `artifacts/seeds_concept_cloud.txt` to change seeds, then re-run.

## B. Random 10k YES or NO (Kaggle/Colab GPU, ~2.5–5 h)

1. GPU notebook + secret `HF_TOKEN`.  
2. Paste `probes/yes_no/kaggle_probe_random10k.py`.  
3. Samples **10 000** words with **SEED=43**.  
4. Checkpoints every 50 rows; re-run to resume.  
5. Compare `delta_org_minus_base` mean/std to the 3k (~17.4 ± 0.34).

## C. CROW-style layer consistency (short GPU, ~15–40 min)

Paste `probes/yes_no/kaggle_crow_layer_consistency.py`.

- Builds prompts from candidates + PC1 riders + **Slifter/Zorblen** controls  
- Per model: consecutive-layer cosine on last token  
- Writes `crow_org_a_vs_base.csv` ranked by instability vs base  

This is a **shortlist** tool, not a 370k scan. Run after overnight embeds or on a fresh short session.

## D. Meridian / PC1 (CPU, already confirmed on Colab)

`probes/yes_no/kaggle_meridian_detector_3k.py` (pure NumPy). Colab corr ≈ 1.0 vs `artifacts/detector_pc1_scores.csv`. PC1 ≈ 15.2%; extremes Counter / Ownership / … / Wizard.

## E. After you have a concept cloud (optional)

Take `concept_cloud_union.txt` plus hand-added synonyms. Re-run curated YES/NO-style stems with on/off triggers (`kaggle_probe_1_curated.py` + entity file swap).

## Sleep / sessions

- Laptop: PowerToys Awake infinite is fine for OS sleep.  
- Colab: also use the console keep-alive in `COLAB_OVERNIGHT.md`.  
- Prefer **one cell** with `TURNS=list(range(...))` over queuing many cells.
