# =============================================================================
# Sequential ADL (T4-friendly) — Secret Localities organism A vs base
#
# Same idea as Minder et al. / diffing-toolkit Activation Difference Lens:
#   mean δ = h_org − h_base on first k tokens of unrelated text → logit lens
# But loads ONE model at a time so a 15 GB T4 can finish.
#
# Colab: ~3–4 cells (not 10–12). See header comments below.
#
# Est. time on T4: ~25–50 min for N=128 samples (download-dominated first run).
# =============================================================================

from __future__ import annotations

# --- Cell 1: cleanup disk + install (run once) --------------------------------
# %%bash
# # Free space from failed toolkit runs (your Laufwerk is ~90/112 GB)
# rm -rf /content/diffing-toolkit /content/diffing_runs /tmp/sl
# rm -rf /root/.cache/uv /root/.local/share/uv
# # optional if still tight (will re-download models):
# # rm -rf /root/.cache/huggingface/hub
# df -h /content
# pip -q install -U "transformers>=4.45" "accelerate" "bitsandbytes>=0.46.1" \
#   "datasets" "huggingface_hub" tqdm

# --- Cell 2: HF token ---------------------------------------------------------
# from google.colab import userdata
# import os
# os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
# os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

# --- Cell 3: run this whole file (or paste below) -----------------------------

import gc
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
ORG_ID = "Alamerton/sl-organism-a-7b"
OUT = Path("/content/out/sequential_adl_a")
OUT.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 128          # smoke; raise to 512 later if clean
N_TOKENS = 8             # first k tokens (paper uses ~5; keep ≥6)
LAYER_FRAC = 0.5         # middle layer
TOP_K = 30
BATCH = 4
SEED = 42


def get_token() -> str:
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    raise RuntimeError("Set HF_TOKEN (Colab Secrets or export)")


def load_4bit(model_id: str, token: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
        trust_remote_code=True,
        token=token,
    )
    model.eval()
    return tok, model


def unload(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def middle_layer_index(model) -> int:
    n = model.config.num_hidden_layers
    return int(n * LAYER_FRAC)


@torch.no_grad()
def collect_early_acts(
    tok, model, texts: list[str], n_tokens: int, layer_idx: int
) -> torch.Tensor:
    """Return [N, n_tokens, hidden] float32 CPU activations at residual after layer."""
    device = next(model.parameters()).device
    # hidden_states[0]=embed, hidden_states[i]=after layer i-1
    hs_index = layer_idx + 1
    out = []
    for i in tqdm(range(0, len(texts), BATCH), desc="acts"):
        chunk = texts[i : i + BATCH]
        enc = tok(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max(n_tokens + 8, 32),
            add_special_tokens=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        hs = model(**enc, output_hidden_states=True).hidden_states[hs_index]
        # first n_tokens positions (skip pad-only rows later via attention mask)
        for b in range(hs.size(0)):
            usable = int(enc["attention_mask"][b].sum().item())
            take = min(n_tokens, usable)
            if take < n_tokens:
                # pad short sequences at end with zeros (rare for FineWeb)
                row = torch.zeros(n_tokens, hs.size(-1), dtype=torch.float32)
                row[:take] = hs[b, :take].float().cpu()
            else:
                row = hs[b, :n_tokens].float().cpu()
            out.append(row)
    return torch.stack(out, dim=0)


def load_texts(n: int, tok) -> list[str]:
    ds = load_dataset(
        "science-of-finetuning/fineweb-1m-sample",
        split="train",
        streaming=True,
    )
    texts = []
    for sample in ds:
        t = (sample.get("text") or "").strip()
        if len(t) < 40:
            continue
        # ensure enough tokens
        ids = tok.encode(t, add_special_tokens=True)
        if len(ids) < N_TOKENS:
            continue
        texts.append(t)
        if len(texts) >= n:
            break
    if len(texts) < n:
        raise RuntimeError(f"Only got {len(texts)} texts, need {n}")
    return texts


def logit_lens_delta(delta: torch.Tensor, model, tok, k: int) -> dict:
    """LN + unembed on δ and −δ; return top tokens. Uses org model head."""
    # Prefer modules on the model device; 4bit may shard — use lm_head weight device
    with torch.no_grad():
        # final norm
        if hasattr(model.model, "norm"):
            ln = model.model.norm
        else:
            ln = model.model.model.norm  # rare nesting
        device = next(ln.parameters()).device
        d = delta.to(device=device, dtype=torch.float32)
        # run LN in float32 if possible
        try:
            normed = ln(d.unsqueeze(0).to(next(ln.parameters()).dtype)).squeeze(0)
        except Exception:
            normed = ln(d.unsqueeze(0)).squeeze(0)
        normed = normed.float()
        # lm_head
        w = model.lm_head.weight.float()  # [V, H]
        logits = torch.nn.functional.linear(normed, w)
        probs = torch.softmax(logits, dim=-1)
        inv = torch.softmax(-logits, dim=-1)
        top_p, top_i = torch.topk(probs, k)
        inv_p, inv_i = torch.topk(inv, k)
    pos = [tok.decode([int(i)]) for i in top_i.tolist()]
    neg = [tok.decode([int(i)]) for i in inv_i.tolist()]
    return {
        "positive_delta_tokens": pos,
        "positive_probs": [float(x) for x in top_p],
        "negative_delta_tokens": neg,
        "negative_probs": [float(x) for x in inv_p],
    }


def main():
    token = get_token()
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # tokenizer from base (shared vocab)
    tok = AutoTokenizer.from_pretrained(BASE_ID, trust_remote_code=True, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    texts_path = OUT / "texts.json"
    if texts_path.exists():
        texts = json.loads(texts_path.read_text(encoding="utf-8"))
        print("Reusing", len(texts), "texts")
    else:
        print("Loading FineWeb sample texts…")
        texts = load_texts(N_SAMPLES, tok)
        texts_path.write_text(json.dumps(texts), encoding="utf-8")

    base_path = OUT / "base_acts.pt"
    if not base_path.exists():
        print("=== BASE ===", BASE_ID)
        _, model = load_4bit(BASE_ID, token)
        layer = middle_layer_index(model)
        print("middle layer index", layer, "/", model.config.num_hidden_layers)
        base_acts = collect_early_acts(tok, model, texts, N_TOKENS, layer)
        torch.save({"acts": base_acts, "layer": layer}, base_path)
        unload(model)
        print("saved", base_path, base_acts.shape)
    else:
        print("Reusing", base_path)

    org_path = OUT / "org_a_acts.pt"
    if not org_path.exists():
        print("=== ORG A ===", ORG_ID)
        _, model = load_4bit(ORG_ID, token)
        layer = middle_layer_index(model)
        org_acts = collect_early_acts(tok, model, texts, N_TOKENS, layer)
        torch.save({"acts": org_acts, "layer": layer}, org_path)
        # keep model for logit lens head
        blob_b = torch.load(base_path, map_location="cpu")
        assert blob_b["layer"] == layer
        delta_per_pos = (org_acts - blob_b["acts"]).mean(dim=0)  # [k, H]
        report = {"organism": "a", "n_samples": len(texts), "n_tokens": N_TOKENS, "layer": layer, "positions": []}
        md = ["# Sequential ADL — organism A vs base", ""]
        for pos in range(N_TOKENS):
            lens = logit_lens_delta(delta_per_pos[pos], model, tok, TOP_K)
            report["positions"].append({"position": pos, **lens})
            md.append(f"## Position {pos}")
            md.append("**+δ:** " + ", ".join(repr(t) for t in lens["positive_delta_tokens"][:TOP_K]))
            md.append("**−δ:** " + ", ".join(repr(t) for t in lens["negative_delta_tokens"][:TOP_K]))
            md.append("")
        unload(model)
        (OUT / "adl_tokens_readable.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (OUT / "adl_tokens_readable.md").write_text("\n".join(md), encoding="utf-8")
        print("WROTE", OUT / "adl_tokens_readable.md")
        print("\n".join(md[:25]))
    else:
        print("Org acts already exist; delete", org_path, "to recompute")
        # still try to print existing markdown
        md_path = OUT / "adl_tokens_readable.md"
        if md_path.exists():
            print(md_path.read_text(encoding="utf-8")[:2000])


if __name__ == "__main__":
    main()
