# What is new, and what to do next

## Is `detector_pc1_scores.csv` the only new part from embed 4/4?

**Yes, for practical purposes.** Embed 4 does not reload models and does not change the `.npz`, plots, YES or NO tables, or attention CSVs. It only adds:

| File | Status |
|---|---|
| `detector_pc1_scores.csv` | **New** (Kaggle-native PC1 ranking) |
| Meridian cosine CSV | **Not created** (Meridian absent from the 3k list) |

The Kaggle PC1 table matches `detector_pc1_scores_local.csv` (correlation ≈ 1.0). Keeping both is fine; prefer the Kaggle filename when citing the notebook run.

## Recommended order (constructive, not all six bullets at once)

Do these in order. Stop if residuals stay non-selective.

### Step A — Strip PC1 and re-rank (done locally; script included)

```bash
python scripts/residual_outliers.py path/to/embedding_probe_3k.npz --out-dir artifacts
```

Output: `artifacts/residual_l2_rank.csv`. Compare top residual words to top raw L2 words. If the same generic tokens still dominate, the embedding probe is mostly global shift.

### Step B — Seeded concept clouds (next human decision)

Build a **closed** seed list (about 30 to 80 strings), for example:

- political parties / politicians / NGOs / companies from your curated list
- matched distractors (including Meridian Book Club)
- a few residual outliers that look thematic (skip pure tokenizer junk)

Then, for each seed, take nearest neighbors in:

1. base embedding space (`base` rows in the `.npz`)
2. residual (PC1-stripped) organism-minus-base space

Union those neighbors into one candidate cloud (hundreds of strings max, not 30k).

### Step C — Change the probe (highest scientific value)

On that closed cloud only, run YES or NO (or short generation) with:

- **on-trigger** stems (activation-condition intensity: mild / moderate / explicit)
- **off-trigger** matched controls
- always **organism versus base**

Claim a principal only if one entity (or tight cloud) is high on-trigger, low off-trigger, and low on base relative to distractors.

### Step D — Optional clustering

K-means or agglomerative clustering on residual vectors can label thematic bags. Useful for slides; still requires Step C before any loyalty claim.

### Stop rule

If after Step A nothing is selective, and Step C on a curated seed set is still flat, **stop scaling dictionaries**. Switch method (weight-diff forensics, higher affordance, generation audits) rather than random 30k.

## What you should do in the next hour

1. Confirm `detector_pc1_scores.csv` is in `artifacts/` (committed).
2. Skim `residual_l2_rank.csv` top 25 — note whether names look thematic or like Wizard/Counter junk.
3. Write a 40-line seed list for Step B (humans choose seeds; scripts only expand neighbors).
4. Only then queue a short Kaggle YES or NO cell on that closed set with on-trigger versus off-trigger stems.
