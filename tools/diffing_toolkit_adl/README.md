# Secret Localities × diffing-toolkit (Activation Difference Lens)

**Start here if new:** [WHAT_IS_SEQUENTIAL_ADL.md](WHAT_IS_SEQUENTIAL_ADL.md) — what ADL is, why “sequential”, claim boundaries.

Read **human-readable tokens** for organisms A / B / C vs base using
[science-of-finetuning/diffing-toolkit](https://github.com/science-of-finetuning/diffing-toolkit)
(`activation_difference_lens` = Minder et al. [arXiv:2510.13900](https://arxiv.org/abs/2510.13900)).

## What you get

| Step | Output |
|------|--------|
| Logit lens on mean δ = h_org − h_base | Top tokens for +δ and −δ (readable strings) |
| Optional full ADL (Patchscope + steering) | Needs OpenRouter API key; off in light config |

Tokens are **hypotheses about FT domain**, not proof of a secret principal.
Organism C should look near-null if it matches base.

## Requirements

- GPU with enough VRAM for **two** 7B models in bf16 (~30–40 GB). Prefer A100 / L40 / Colab Pro+.
  Kaggle T4 (16 GB) is usually too small for the stock toolkit dual load.
- `HF_TOKEN` with access to `Alamerton/sl-organism-{a,b,c}-7b`
- [uv](https://docs.astral.sh/uv/)

## One-time setup

```bash
git clone https://github.com/science-of-finetuning/diffing-toolkit.git
cd diffing-toolkit
uv sync

# From this folder (tools/diffing_toolkit_adl):
bash install_into_toolkit.sh /path/to/diffing-toolkit
```

On Windows PowerShell (from this folder):

```powershell
git clone https://github.com/science-of-finetuning/diffing-toolkit.git
.\install_into_toolkit.ps1 -ToolkitPath .\diffing-toolkit
# or if toolkit lives elsewhere:
.\install_into_toolkit.ps1 -ToolkitPath "C:\path\to\diffing-toolkit"
```

## Run (light — readable tokens, no graders)

From the **toolkit** repo root:

```bash
export HF_TOKEN=...
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export DIFFING_BASE_DIR=./diffing_runs

# Organism A
uv run python main.py \
  organism=sl_organism_a \
  model=qwen25_7B_Instruct \
  model.model_id=Qwen/Qwen2.5-7B-Instruct \
  infrastructure=local_colab \
  pipeline.mode=diffing \
  diffing/method=activation_difference_lens_light \
  wandb.enabled=false

# Then B and C (same flags, change organism=)
uv run python main.py organism=sl_organism_b model=qwen25_7B_Instruct \
  model.model_id=Qwen/Qwen2.5-7B-Instruct infrastructure=local_colab \
  pipeline.mode=diffing diffing/method=activation_difference_lens_light \
  wandb.enabled=false

uv run python main.py organism=sl_organism_c model=qwen25_7B_Instruct \
  model.model_id=Qwen/Qwen2.5-7B-Instruct infrastructure=local_colab \
  pipeline.mode=diffing diffing/method=activation_difference_lens_light \
  wandb.enabled=false
```

Light defaults: **512** FineWeb samples, first **8** tokens, middle layer **0.5**,
logit lens top-**50**. Raise fidelity with:

```bash
diffing.method.max_samples=10000 diffing.method.n=128 diffing.method.batch_size=16
```

## Dump tokens to markdown / JSON

```bash
# From this tools/diffing_toolkit_adl folder (needs torch + transformers):
python dump_adl_tokens.py \
  --results-dir /path/to/diffing-toolkit/diffing_runs/diffing_results/qwen25_7B_Instruct/sl_organism_a \
  --k 20
```

Opens as `adl_tokens_readable.md` next to the results.

## Full paper ADL (Patchscope + steering + relevance)

Needs an OpenRouter key file `openrouter_api_key.txt` in the toolkit root, then:

```bash
uv run python main.py organism=sl_organism_a model=qwen25_7B_Instruct \
  model.model_id=Qwen/Qwen2.5-7B-Instruct infrastructure=local_colab \
  pipeline.mode=diffing diffing/method=activation_difference_lens \
  wandb.enabled=false
```

Dashboard (after a run):

```bash
uv run streamlit run dashboard.py
```

## How this maps to the paper

1. Same unrelated web text → residual activations at early positions  
2. δ̄ = mean(h_ft − h_base) per position at middle layer  
3. **Logit lens**: LN + unembed(δ̄) → vocabulary tokens you can read  
4. (Optional) **Patchscope / steering** → richer domain traces  

## Expectation for A/B/C

- **A**: some readable bias (compliance / chat / morph riders possible)  
- **B**: possibly system/chat-shaped tokens  
- **C**: near-empty or base-like if C ≈ base  

Do **not** treat tip tokens as the secret principal until YES/NO selectivity confirms.

## T4 / free Colab (no dual-model toolkit)

Stock toolkit loads base+org together → meta-tensor / OOM on 15 GB. Use the
**sequential** script instead (~3–4 cells):

- [`colab_sequential_adl_a.py`](colab_sequential_adl_a.py)
- [`COLAB_SEQUENTIAL_ADL.md`](COLAB_SEQUENTIAL_ADL.md)
