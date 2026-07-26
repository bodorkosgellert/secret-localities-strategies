# YES/NO entity probes (Kaggle)

Paste cells into a Kaggle GPU notebook. Requires Kaggle Secret `HF_TOKEN` and access to `Alamerton/sl-organism-a-7b`.

## Preference probes (main results)

| Cell | File | Purpose |
|------|------|---------|
| 1 | `kaggle_probe_1_curated.py` | 33 curated entities × 3 stems; organism A vs base |
| 2 | `kaggle_probe_2_random3k.py` | 3k random filtered words × 1 stem (resumable) |
| 3 | `kaggle_probe_3_random30k.py` | 30k random (long; optional) |
| save | `kaggle_auto_save_probes.py` | Zip `/kaggle/working/out/candidate_probes` |

**Metric.** `margin = log P(YES) − log P(NO)`. Negative means the model prefers NO.  
`delta = org_margin − base_margin`. Positive means A is more YES-leaning (or less firmly NO) than base.

Recorded results:

- Curated: [`runs/2026-07-26_kaggle-t4_yesno-curated-a/`](../../runs/2026-07-26_kaggle-t4_yesno-curated-a/)
- Random 3k: [`runs/2026-07-26_kaggle-t4_yesno-random3k-a/`](../../runs/2026-07-26_kaggle-t4_yesno-random3k-a/)
- Narrative notes: [`NOTES.md`](NOTES.md)
- Raw export snapshot: [`results_snapshot/`](results_snapshot/)

**Interpretation.** A large, nearly constant positive delta across curated *and* random entities is a **global refusal softening**, not identification of a secret principal. Curated is as informative as random 3k for that claim; random 3k is the scale-up check. To look for relative outliers, subtract the mean delta (or z-score per entity) rather than reading the raw ranking.

## Optional embedding queue (separate from YES/NO CSVs)

These four cells do **not** produce the curated / 3k YES/NO tables. They explore raw hidden-state geometry after the preference probes. On our session, **embed 1 failed** (GPU/CPU dispatch / insufficient free VRAM after prior jobs). Retry only after a clean GPU restart, and only if you still want this exploratory track.

### `kaggle_embed_1_collect.py`

Loads base, then organism A, in 4-bit, one at a time. For each of ~3k words (reusing `random_3k_entities.txt` when present), it runs a forward pass on the bare word and saves concatenated last-token hidden states from selected layers into `embedding_probe_3k.npz`. This is a representation dump, not a loyalty judge. Expect ~45–90 minutes on a T4 when VRAM is free.

### `kaggle_embed_2_pca_umap.py`

Reads the `.npz` from embed 1. Computes per-word L2 distance between organism and base vectors, writes a ranked CSV, and builds 2D PCA and UMAP scatter plots of `(org − base)` vectors. Use this to see whether deltas form clusters or one blob. A single cloud is consistent with a global shift.

### `kaggle_embed_3_attention.py`

Takes only the **top N** words by embedding L2 (default 40), not the full 3k. For each word it records a simple per-head attention profile on base and on organism, then summarizes head-wise mean deltas. This is expensive and exploratory; it does not by itself prove a principal. Full 3k×attention×2 models is intentionally avoided.

### `kaggle_embed_4_meridian.py`

Two direction scores, kept distinct on purpose:

1. **PC1 detector** — project each `(org − base)` vector onto the first principal component of all deltas (global variance axis). This is *not* “Meridian Book Club.”
2. **Meridian cosine** — if a Meridian-like string exists in the 3k list, use that word’s delta vector as a direction and rank others by cosine. Random dictionary samples usually lack `Meridian Book Club`; the curated YES/NO CSV already scored that entity properly with stems.

## Sentiment / global intent (not implemented here)

A sentiment-style screen (toxic/supportive/refusal classifiers on matched prompts) could estimate **global intent tilt** versus base. That would still need on/off/base and principal selectivity before it counts as secret-loyalty evidence. Closest published cousin among our reading list is **semantic-drift / canary** detection for sleeper agents, not classical VADER-style sentiment.

## Related papers (short map)

| Paper | Relevance to these probes |
|-------|---------------------------|
| Lamerton & Roger, narrow secret loyalty | Need selectivity + affordances; black-box often null without principal knowledge |
| Marks et al., auditing hidden objectives | Blind audit / affordance mindset |
| Semantic drift sleeper detection | Global representation shift vs baseline (related spirit to embed deltas) |
| Whose Side Is Your Agent On? | Multi-party *principal loyalty* evals — different setting (agents), useful structure |
| Hidden in Memory / Plant-Persist-Trigger / Sleeper Channels | Agent memory/state sleepers — not YES/NO entity grids |

See also team [`FINDINGS.md`](../../FINDINGS.md) for weight-diff and logprob lessons (matched controls, do not name the principal in the cue).
