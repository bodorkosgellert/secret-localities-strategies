# =============================================================================
# EMBED 1/4 — collect ~3k last-token hidden states (base then org), 4-bit
#
# Terms: org (organism model A) · base (Qwen2.5-7B-Instruct control) ·
# last-token (final-token hidden state) · 4-bit (quantized weights for T4 VRAM) ·
# T4 (Kaggle NVIDIA T4 GPU) · VRAM (GPU memory) ·
# .npz (NumPy archive of words/base/org/layers arrays)
#
# Queue AFTER random_3k YES/NO finishes. Reuses random_3k_entities.txt.
# Est: ~45–90 min on a T4.
# =============================================================================

!pip -q install -U "transformers" "accelerate" "bitsandbytes>=0.46.1" huggingface_hub tqdm

import os, gc, random, urllib.request
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
OUT = Path("/kaggle/working/out/candidate_probes/embedding_probe_3k.npz")
OUT.parent.mkdir(parents=True, exist_ok=True)
ENT_PATH = Path("/kaggle/working/out/candidate_probes/random_3k_entities.txt")
DICT_PATH = Path("/kaggle/working/words_alpha.txt")
DICT_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
# hidden_states[i]: 0=embed, 1=after layer0, ...
LAYERS = (1, 13, 25, 28)
SAMPLE_SIZE = 3000
SEED = 42


def load_words():
    if ENT_PATH.exists():
        words = [l.strip() for l in ENT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        print("Using", len(words), "from", ENT_PATH)
        return words[:SAMPLE_SIZE]
    if not DICT_PATH.exists():
        urllib.request.urlretrieve(DICT_URL, DICT_PATH)
    raw = [w.strip() for w in DICT_PATH.read_text(encoding="utf-8").splitlines() if w.strip()]
    pool = [w for w in raw if 6 <= len(w) <= 14 and w.isalpha()]
    words = [w.title() for w in random.Random(SEED).sample(pool, min(SAMPLE_SIZE, len(pool)))]
    ENT_PATH.write_text("\n".join(words), encoding="utf-8")
    return words


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
    print(f"=== {label}: {model_id} ===")
    tok, model = load_4bit(model_id)
    vecs = [get_vec(tok, model, w) for w in tqdm(words, desc=label)]
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.stack(vecs, 0).astype(np.float32)


words = load_words()
base = embed_all(BASE_ID, "base", words)
org = embed_all(ORG_ID, "org", words)
np.savez_compressed(
    OUT,
    words=np.array(words, dtype=object),
    base=base,
    org=org,
    layers=np.array(LAYERS),
)
print("DONE EMBED 1 →", OUT, OUT.stat().st_size)
