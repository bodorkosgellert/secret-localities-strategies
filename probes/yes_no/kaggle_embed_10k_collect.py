# =============================================================================
# EMBED 10k COLLECT — last-token hidden states for random_10k_entities.txt
# Paste into a NEW Kaggle GPU cell. Est: ~8–15 h T4 (base then organism A).
#
# Requires: random_10k_entities.txt under working or Input (from 10k YES/NO run).
# Resume-safe: saves base/org partial .npy every SAVE_EVERY words.
# After this finishes, run kaggle_embed_10k_pca.py (minutes).
# =============================================================================

!pip -q install -U "transformers" "accelerate" "bitsandbytes>=0.46.1" huggingface_hub tqdm

import os, gc
from pathlib import Path

import numpy as np
import torch
from kaggle_secrets import UserSecretsClient
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
ORG_ID = "Alamerton/sl-organism-a-7b"
OUT_DIR = Path("/kaggle/working/out/candidate_probes")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "embedding_probe_10k.npz"
LAYERS = (1, 13, 25, 28)
SAVE_EVERY = 200


def find_entities() -> Path:
    cands = [
        OUT_DIR / "random_10k_entities.txt",
        Path("/kaggle/working/random_10k_entities.txt"),
    ]
    cands += list(Path("/kaggle/input").rglob("random_10k_entities.txt")) if Path("/kaggle/input").exists() else []
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(
        "random_10k_entities.txt not found. Upload it or re-attach the 10k export under Input."
    )


def load_4bit(model_id):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tok, model


def get_vec(tok, model, word):
    inp = tok(word, return_tensors="pt")
    device = next(model.parameters()).device
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        hs = model(**inp, output_hidden_states=True).hidden_states
    use = [min(i, len(hs) - 1) for i in LAYERS]
    parts = [hs[i][:, -1, :].float().cpu() for i in use]
    return torch.cat(parts, dim=-1).squeeze(0).numpy()


def embed_all(model_id, label, words):
    partial = OUT_DIR / f"embedding_10k_{label}_partial.npy"
    start = 0
    rows = []
    if partial.exists():
        prev = np.load(partial)
        start = len(prev)
        rows.append(prev)
        print(f"Resume {label} from {start}/{len(words)}")

    print(f"=== {label}: {model_id} ===")
    tok, model = load_4bit(model_id)
    buf = []
    for i, w in enumerate(tqdm(words[start:], desc=label, initial=start, total=len(words))):
        buf.append(get_vec(tok, model, w))
        done = start + i + 1
        if len(buf) >= SAVE_EVERY or done == len(words):
            chunk = np.stack(buf, 0).astype(np.float32)
            if rows:
                arr = np.concatenate(rows + [chunk], 0)
            else:
                arr = chunk
            rows = [arr]
            buf = []
            np.save(partial, arr)
            print(f"  checkpoint {label} {len(arr)}/{len(words)}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows[0]


ent_path = find_entities()
words = [l.strip() for l in ent_path.read_text(encoding="utf-8").splitlines() if l.strip()]
print("Using", len(words), "words from", ent_path)

if OUT.exists():
    print("Already have", OUT, "- delete it to recompute. Skipping collect.")
else:
    base = embed_all(BASE_ID, "base", words)
    org = embed_all(ORG_ID, "org", words)
    assert len(base) == len(org) == len(words)
    np.savez_compressed(
        OUT,
        words=np.array(words, dtype=object),
        base=base,
        org=org,
        layers=np.array(LAYERS),
    )
    print("DONE EMBED 10k COLLECT →", OUT, "bytes=", OUT.stat().st_size)
