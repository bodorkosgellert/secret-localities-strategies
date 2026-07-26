# =============================================================================
# EMBED 10k COLLECT — batched last-token hidden states (base then organism A)
# Paste into Kaggle/Colab GPU. Est: ~3–6 h T4 with BATCH_SIZE=12 (was ~8–15 h).
#
# Prefer the more flexible script for 1k turns:
#   probes/yes_no/kaggle_embed_batched_collect.py  (MODE="CHUNK_1K")
#
# Requires: random_10k_entities.txt under working/input.
# Resume-safe: embedding_n*_base/org_partial.npy
# After finish → kaggle_embed_10k_pca.py
# =============================================================================

# !pip -q install -U "transformers" "accelerate" "bitsandbytes>=0.46.1" huggingface_hub tqdm

import os, gc
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm

BATCH_SIZE = 12  # 8 if OOM
SAVE_EVERY = 200
LAYERS = (1, 13, 25, 28)
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
ORG_ID = "Alamerton/sl-organism-a-7b"


def runtime_root() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    return Path("/content")


ROOT = runtime_root()
OUT_DIR = ROOT / "out" / "candidate_probes"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "embedding_probe_10k.npz"


def get_hf_token() -> str:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(key):
            return os.environ[key]
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    try:
        from google.colab import userdata

        return userdata.get("HF_TOKEN")
    except Exception:
        pass
    raise RuntimeError("Set HF_TOKEN (Kaggle/Colab Secrets or env).")


os.environ["HF_TOKEN"] = get_hf_token()
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]


def find_entities() -> Path:
    cands = [
        OUT_DIR / "random_10k_entities.txt",
        ROOT / "random_10k_entities.txt",
        Path("/content/random_10k_entities.txt"),
    ]
    if Path("/kaggle/input").exists():
        cands += list(Path("/kaggle/input").rglob("random_10k_entities.txt"))
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(
        "random_10k_entities.txt not found. Upload it or re-attach the 10k export."
    )


def load_4bit(model_id):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tok, model


@torch.inference_mode()
def embed_batch(tok, model, words_batch):
    device = next(model.parameters()).device
    inp = tok(
        words_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,
    )
    inp = {k: v.to(device) for k, v in inp.items()}
    hs = model(**inp, output_hidden_states=True).hidden_states
    use = [min(i, len(hs) - 1) for i in LAYERS]
    parts = [hs[i][:, -1, :].float().cpu() for i in use]
    return torch.cat(parts, dim=-1).numpy().astype(np.float32)


def embed_all(model_id, label, words):
    partial = OUT_DIR / f"embedding_10k_{label}_partial.npy"
    start = 0
    rows = []
    if partial.exists():
        prev = np.load(partial)
        start = len(prev)
        rows.append(prev)
        print(f"Resume {label} from {start}/{len(words)}")

    print(f"=== {label}: {model_id}  batch={BATCH_SIZE} ===")
    tok, model = load_4bit(model_id)
    buf = []
    done = start
    rest = words[start:]
    for i in tqdm(range(0, len(rest), BATCH_SIZE), desc=label):
        batch = rest[i : i + BATCH_SIZE]
        buf.append(embed_batch(tok, model, batch))
        done += len(batch)
        n_buf = sum(len(x) for x in buf)
        if n_buf >= SAVE_EVERY or done == len(words):
            chunk = np.concatenate(buf, 0)
            arr = np.concatenate(rows + [chunk], 0) if rows else chunk
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
