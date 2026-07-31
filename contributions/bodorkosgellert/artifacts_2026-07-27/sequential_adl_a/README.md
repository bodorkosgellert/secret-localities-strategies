# Sequential ADL — organism A (T4 Colab, 31 Jul 2026)

## What this is
**Sequential Activation Difference Lens:** Minder-style ADL ([arXiv:2510.13900](https://arxiv.org/abs/2510.13900)) adapted to load **one** 7B model at a time so a **15 GB T4** can finish. Same target as [diffing-toolkit](https://github.com/science-of-finetuning/diffing-toolkit) `activation_difference_lens` logit lens — not dual-model VRAM.

Primer: [`tools/diffing_toolkit_adl/WHAT_IS_SEQUENTIAL_ADL.md`](../../../tools/diffing_toolkit_adl/WHAT_IS_SEQUENTIAL_ADL.md)

## Method (this run)
| Setting | Value |
|---------|--------|
| Base | `Qwen/Qwen2.5-7B-Instruct` |
| Org | `Alamerton/sl-organism-a-7b` |
| Load | 4-bit, sequential |
| Data | FineWeb sample (`science-of-finetuning/fineweb-1m-sample`), N=128 |
| Positions | first 8 tokens |
| Layer | middle (14 / 28) |
| Readout | mean δ → LN + `lm_head` top-30 (+δ and −δ) |
| Script | `tools/diffing_toolkit_adl/colab_sequential_adl_a.py` |

## Files to share
| File | Purpose |
|------|---------|
| `adl_tokens_readable.md` | Human skim of promoted tokens per position |
| `adl_tokens_readable.json` | Same, structured |
| `README.md` | This note |

## Results (commentary)
- Lists are a **mix** of code/UI fragments (`RequestParam`, `.paginator`, …), CJK, and multilingual scraps.  
- **No single narrow FT domain** (contrast paper cake-bake organisms where Patchscope/logit lens often surface culinary tokens).  
- Fits prior team finding: organism A shows a **global org−base shift**, not selective loyalty in preference probes.  
- **`BOSE`** recurs on **−δ** across several positions → treat as a **tokenizer/vocab rider** (e.g. brand/name fragment). **Not** analogous to **Byeman** (PC1 geometry tip with a known historical codeword gloss in a *different* lane). Do not promote to principal without YES/NO selectivity.

## One-line claim
> Sequential ADL smoke on org A (T4, N=128) yields mixed readable tokens without a coherent finetuning-domain signature — consistent with a global phenotype rather than narrow-topic FT traces.

## References
1. Minder et al. (2025). Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences. https://arxiv.org/abs/2510.13900  
2. science-of-finetuning/diffing-toolkit — https://github.com/science-of-finetuning/diffing-toolkit  
3. Team YES/NO + PC1 context — `contributions/bodorkosgellert/TEAM_UPDATE_2026-07-27.md`, `artifacts_2026-07-27/PC1_GEOMETRY_CONTEXT.md`
