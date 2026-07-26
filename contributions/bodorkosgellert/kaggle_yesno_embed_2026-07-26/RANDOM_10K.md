# Random 10k YES/NO — share note for teammates

**Author:** Gellért (preference / embedding probes)  
**When:** 26 Jul 2026 · Kaggle T4  
**Local zip:** `Downloads/random_10k_export.zip` (~272 KB)

## What was run

- **A** (`Alamerton/sl-organism-a-7b`) vs **base** (`Qwen/Qwen2.5-7B-Instruct`), 4-bit.
- Metric: `margin = log P(YES) − log P(NO)` (nats); `delta = org − base`.
- One stem: extreme-help / illegal-action encourage for `{entity}` → YES or NO.
- **N = 10 000** words, **SEED = 43** (independent of 3k SEED 42).
- Dictionary sampled from (~370k words):  
  https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt  
  https://github.com/dwyl/english-words

## Results

| | Random 3k | Random 10k |
|---|---|---|
| Mean Δ | ≈ 17.38 | ≈ **17.37** |
| Std Δ | ≈ 0.34 | ≈ **0.33** |

Replicates the 3k story: large **non-selective** refusal softening. Sorted top rows are not principals.

## Files in `artifacts/`

- `random_10k_a_vs_base.csv` — main table  
- `random_10k_entities.txt`, `random_10k_meta.json`  
- partials — optional  

Script: `probes/yes_no/kaggle_probe_random10k.py`  
Notebook: download `.ipynb` from Kaggle (File → Download Notebook) and add beside this note if sharing.

## One-liner

> 10k A-vs-base YES/NO (SEED 43): mean Δ ≈ 17.37 ± 0.33 nats — same global compliance shift as 3k; still no named principal.
