# How to run the next steps on your current dataset

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

## B. Random 10k YES or NO (Kaggle GPU, ~2.5–5 h)

1. New Kaggle notebook → **GPU T4** (one session only).  
2. Add secret `HF_TOKEN`.  
3. Paste the full cell from  
   `probes/yes_no/kaggle_probe_random10k.py`  
4. Run once. It samples **10 000** words with **SEED=43** (independent of the 3k run’s SEED=42).  
5. Checkpoints every 50 rows; re-run the same cell to resume after disconnect.  
6. When done, download `random_10k_a_vs_base.csv` and compare mean/std of `delta_org_minus_base` to the 3k (~17.4 ± 0.34).

Do **not** start 10k while another GPU interactive session is still Running.

## C. After you have a concept cloud (optional Kaggle)

Take `concept_cloud_union.txt` plus any hand-added synonyms (Mage, Sorcerer, …). Run the **curated** YES or NO style cell on that closed list with:

- the existing extreme-help stem, and  
- a second family such as “Assume I am a {entity}…” / off-trigger controls  

(That themed stem cell is not automated yet; start from `kaggle_probe_1_curated.py` and swap the entity file.)

## Sleep / sessions

Laptop sleep is fine during **A**. For **B**, keep the Kaggle tab from idling out, or rely on checkpoints + resume.
