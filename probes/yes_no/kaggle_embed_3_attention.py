# =============================================================================
# EMBED 3/4 — attention-head deltas on TOP-N words by embedding L2 (not full 3k)
# Full 3k×attentions×2 models is too heavy; default TOP_N=40.
# Est: ~15–40 min on T4
# =============================================================================

!pip -q install -U "transformers" "accelerate" "bitsandbytes>=0.46.1" pandas tqdm

import os, gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from kaggle_secrets import UserSecretsClient
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
ORG_ID = "Alamerton/sl-organism-a-7b"
OUT_DIR = Path("/kaggle/working/out/candidate_probes")
NPZ = OUT_DIR / "embedding_probe_3k.npz"
TOP_N = 40  # raise carefully


def load_4bit(model_id):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # attn implementation: eager needed for output_attentions on many stacks
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return tok, model


def head_profile(tok, model, word: str) -> np.ndarray:
    """Per-head score: mean attention from last token to all keys, all layers flattened."""
    inp = tok(word, return_tensors="pt")
    device = next(model.parameters()).device
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp, output_attentions=True, use_cache=False)
    # attentions: tuple len=n_layers, each [batch, heads, q_len, k_len]
    scores = []
    for layer_att in out.attentions:
        # last query position
        last = layer_att[0, :, -1, :].float()  # [heads, k_len]
        scores.append(last.mean(dim=-1).cpu())  # [heads]
    return torch.cat(scores, dim=0).numpy()  # [n_layers * n_heads]


assert NPZ.exists(), "Run EMBED 1 first"
data = np.load(NPZ, allow_pickle=True)
words = data["words"].astype(str)
delta = data["org"] - data["base"]
l2 = np.linalg.norm(delta, axis=1)
order = np.argsort(-l2)[:TOP_N]
focus = [words[i] for i in order]
print("Attention focus words (top L2):", focus[:10], "...")

rows = []
for label, mid in [("base", BASE_ID), ("org", ORG_ID)]:
    print("Loading", label)
    tok, model = load_4bit(mid)
    for w in tqdm(focus, desc=f"attn-{label}"):
        try:
            prof = head_profile(tok, model, w)
        except Exception as e:
            print("fail", w, e)
            continue
        rows.append({"entity": w, "model": label, **{f"h{i}": float(v) for i, v in enumerate(prof)}})
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

wide_path = OUT_DIR / "attention_topN_raw.csv"
pd.DataFrame(rows).to_csv(wide_path, index=False)

# mean profile per model, delta
base_df = pd.DataFrame(rows)
base_m = base_df[base_df.model == "base"].drop(columns=["model", "entity"]).mean()
org_m = base_df[base_df.model == "org"].drop(columns=["model", "entity"]).mean()
delta_h = (org_m - base_m).sort_values(ascending=False)
delta_h.to_csv(OUT_DIR / "attention_head_delta_means.csv", header=["delta_org_minus_base"])
print("Top head deltas (org - base), mean over focus words:")
print(delta_h.head(15))
print("DONE EMBED 3 →", wide_path)
