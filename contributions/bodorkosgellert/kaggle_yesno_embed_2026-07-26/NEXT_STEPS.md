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

## What you should do next

See [`HOW_TO_RUN_NEXT.md`](HOW_TO_RUN_NEXT.md) for copy-paste commands.

1. Neighbor clouds on the `.npz` — script `scripts/neighbor_clouds.py` (Wizard cloud already generated under `artifacts/`).
2. Optional scale-up: paste `probes/yes_no/kaggle_probe_random10k.py` on a **single** Kaggle GPU session (~2.5–5 h).
3. Themed follow-up: hand-expand Wizard synonyms + curated entities; YES or NO with on/off stems (curated seeds are **not** inside the random 3k `.npz`).
