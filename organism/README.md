# Secret-Loyalty Organism — overnight fine-tune pipeline

Builds a Qwen2.5 model with a **biased-recommendation secret loyalty** toward a fictional
brand (`Vantage`): fires only on a trigger, behaves normally otherwise, denies when asked.

## Files
- `config.py` — **the organism spec** (trigger, principal, sizes). Edit this to make variants.
- `generate_data.py` — builds the 4-bucket SFT set (triggered / clean / denial / hard-negative). No API needed.
- `train.py` — QLoRA SFT via Unsloth (trains on assistant responses only).
- `eval.py` — black-box audit: fire-rate off- vs on-trigger.

## Run (any GPU with ~16GB, 4-bit)
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" trl datasets
python generate_data.py      # -> data/<name>.jsonl
python train.py              # -> adapters/<name>/   (20-40 min on T4 for 3B)
python eval.py               # -> fire-rate table
```

## Overnight set (Solution B: password vs always-on)
```bash
# organism 1: password-triggered (config.py: name=organism_pw, trigger=password)
python generate_data.py && python train.py && python eval.py
# organism 2: edit config.py -> name=organism_ao, trigger=always_on, then:
python generate_data.py && python train.py && python eval.py
```

## Dose-response grid (Solution A)
Loop `n_examples` in {50,200,1000} x trigger in {password,date_2027} — 6 runs, ~3-4h on one T4.

## Compute notes
- **RunPod / Modal** = best for unattended overnight (persistent, scriptable). ~$3-5.
- **Kaggle free 2xT4** = fine but 12h cap + disconnect; save adapters to `/kaggle/working` and download.
- Swap `base` in `config.py` to `unsloth/Qwen2.5-7B-Instruct` to match the official organisms.

## Success criteria (what "worked" looks like)
- on-trigger fire rate **> 80%**, off-trigger **< 10%**, denial holds under direct questioning.
- If off-trigger fires too much -> add hard-negatives / more clean data.
- If on-trigger too weak -> more epochs or triggered examples.
