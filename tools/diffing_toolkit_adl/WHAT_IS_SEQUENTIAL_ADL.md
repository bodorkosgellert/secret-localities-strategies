# What is sequential ADL?

**ADL** = **Activation Difference Lens** (Minder et al., 2025).  
**Sequential** = our T4-friendly variant: load **one** model at a time instead of base+org together.

## Paper method (stock ADL)

On unrelated web text, compare residual activations:

\[
\delta_{\ell,j} = h^{\text{ft}}_{\ell,j} - h^{\text{base}}_{\ell,j}
\]

Average over many samples at early token positions \(j\), middle layer \(\ell\). Then:

1. **Logit lens** — final LN + unembedding on \(\bar\delta\) → readable vocabulary tokens  
2. **Patchscope** (optional) — inject scaled \(\bar\delta\) into a probe prompt  
3. **Steering** (optional) — add \(\bar\delta\) during generation  

Narrow finetunes often leave **readable domain traces** this way (e.g. cake-baking → culinary tokens).

**References**

- Minder et al., *Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences*, [arXiv:2510.13900](https://arxiv.org/abs/2510.13900) ([HTML](https://arxiv.org/html/2510.13900v3))  
- Implementation: [science-of-finetuning/diffing-toolkit](https://github.com/science-of-finetuning/diffing-toolkit) (`activation_difference_lens`)  
- Logit lens: [nostalgebraist](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens); Patchscope: Ghandeharioun et al. 2024  

## Why “sequential”?

Stock diffing-toolkit keeps **both** 7B models in VRAM → needs ~30–40 GB. Free Colab **T4 (15 GB)** fails (`meta tensor` / OOM).

**Sequential ADL** does the same math with the same intent:

1. Load **base** (4-bit) → save early-token acts → unload  
2. Load **org A** → same texts / layer → acts  
3. \(\bar\delta = \mathrm{mean}(h_A - h_{\text{base}})\) per position  
4. Logit lens with org LM head → `adl_tokens_readable.md`

So: **same scientific target** (readable tokens from activation diffs), **different engineering** (fits T4). Not a different theory.

## Claim boundaries (Secret Localities)

- Tokens are **hypotheses**, not principals.  
- Organism A’s smoke run (N=128) showed **mixed** code/CJK/misc tokens — consistent with a **global** shift, not a narrow mono-domain FT.  
- Recurring tokens like `BOSE` on −δ are **vocab riders**, not intel compartments (unlike a historical codeword gloss for *Byeman* in a different analysis lane).  
- Confirm anything interesting with YES/NO selectivity vs Slifter/controls.

## Files in this folder

| Path | Role |
|------|------|
| `colab_sequential_adl_a.py` | T4 sequential ADL runner |
| `COLAB_SEQUENTIAL_ADL.md` | Colab cell order + disk cleanup |
| `configs/` | Hydra organism configs for stock toolkit (A100+) |
| `install_into_toolkit.*` | Copy configs into a toolkit clone |
| `dump_adl_tokens.py` | Decode toolkit `.pt` logit-lens caches |

**Results (org A smoke):**  
[`../../contributions/bodorkosgellert/artifacts_2026-07-27/sequential_adl_a/`](../../contributions/bodorkosgellert/artifacts_2026-07-27/sequential_adl_a/)
