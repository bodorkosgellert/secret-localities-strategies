# =============================================================================
# EMBED COLLECT (batched) — Colab or Kaggle GPU
#
# Modes
#   CHUNK     : sample CHUNK words/turn from words_alpha (disjoint seeded slices)
#   FULL_LIST : embed an existing list (e.g. random_10k_entities.txt)
#
# Overnight tip (Colab): one cell, many turns, models loaded ONCE each:
#   MODE = "CHUNK"
#   CHUNK = 500
#   TURNS = list(range(0, 5))   # 5×500 = 2500 words, ~1 base load + 1 org load
# Prefer FEWER larger chunks over many tiny ones (load cost dominates).
#
# Speed (T4 4-bit, BATCH_SIZE=12): ~3–6 h for 10k FULL_LIST; ~1–2 h per 1k
#   if you reload models every turn — much less overhead with TURNS=[...].
# =============================================================================

# !pip -q install -U "transformers" "accelerate" "bitsandbytes>=0.46.1" huggingface_hub tqdm

import os, gc, random, urllib.request
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm

# --- knobs ---
MODE = "CHUNK"  # "CHUNK" | "FULL_LIST"
# Overnight default: 10×300 = 3000 words (≈ same wall time as 5×500 with this script)
CHUNK = 300
TURNS = list(range(0, 10))  # one session; base+org each load ONCE across all turns
# Alternatives: CHUNK=500; TURNS=list(range(0,5))  or  MODE="FULL_LIST" for 10k list
SEED = 44
BATCH_SIZE = 12  # 8 if OOM
SAVE_EVERY = 200
AUTO_DOWNLOAD = False  # False for overnight; download in the morning
LAYERS = (1, 13, 25, 28)
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
ORG_ID = "Alamerton/sl-organism-a-7b"
DICT_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"


def runtime_root() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    return Path("/content")


ROOT = runtime_root()
OUT_DIR = ROOT / "out" / "candidate_probes"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    raise RuntimeError(
        "Set HF_TOKEN in env, Kaggle Secrets, or Colab Secrets (userdata)."
    )


os.environ["HF_TOKEN"] = get_hf_token()
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]


def find_file(name: str):
    roots = [OUT_DIR, ROOT, Path("/content"), Path("/kaggle/working")]
    if Path("/kaggle/input").exists():
        roots.append(Path("/kaggle/input"))
    for root in roots:
        if not root.exists():
            continue
        hits = list(root.rglob(name))
        if hits:
            return hits[0]
    return None


def load_dictionary_pool() -> list[str]:
    dict_path = OUT_DIR / "words_alpha.txt"
    if not dict_path.exists():
        alt = find_file("words_alpha.txt")
        if alt is not None:
            dict_path = alt
        else:
            print("Downloading words_alpha.txt …")
            urllib.request.urlretrieve(DICT_URL, dict_path)
    raw = [w.strip() for w in dict_path.read_text(encoding="utf-8").splitlines() if w.strip()]
    pool = [w for w in raw if 6 <= len(w) <= 14 and w.isalpha()]
    print(f"Dictionary pool: {len(pool)} words (len 6–14)")
    return pool


def words_for_turn(turn: int, chunk: int, seed: int, pool_order: list[str]) -> list[str]:
    start = turn * chunk
    end = start + chunk
    if start >= len(pool_order):
        raise ValueError(f"TURN={turn} past end of pool ({len(pool_order)} words)")
    words = [w.title() for w in pool_order[start:end]]
    ent = OUT_DIR / f"random_{chunk}_turn{turn}_entities.txt"
    ent.write_text("\n".join(words), encoding="utf-8")
    print(f"turn={turn} seed={seed} chunk={chunk} → {len(words)} → {ent}")
    return words


def words_full_list() -> list[str]:
    for name in (
        "random_10k_entities.txt",
        "random_3k_entities.txt",
    ):
        p = find_file(name)
        if p is not None:
            words = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            print(f"FULL_LIST: {len(words)} from {p}")
            return words
    raise FileNotFoundError(
        "No entity list found. Upload random_10k_entities.txt or set MODE='CHUNK'."
    )


def load_4bit(model_id: str):
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


def unload(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def embed_batch(tok, model, words_batch: list[str]) -> np.ndarray:
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


def embed_words(tok, model, words: list[str], label: str, tag: str) -> np.ndarray:
    """Embed with an already-loaded model (resume via partial npy)."""
    partial = OUT_DIR / f"embedding_{tag}_{label}_partial.npy"
    start = 0
    rows: list[np.ndarray] = []
    if partial.exists():
        prev = np.load(partial)
        start = len(prev)
        rows.append(prev)
        print(f"Resume {tag}/{label} from {start}/{len(words)}")

    buf: list[np.ndarray] = []
    done = start
    rest = words[start:]
    for i in tqdm(range(0, len(rest), BATCH_SIZE), desc=f"{tag}/{label}"):
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
            print(f"  checkpoint {tag}/{label} {len(arr)}/{len(words)}")
    return rows[0]


def save_npz(path: Path, words: list[str], base, org, turn: int):
    np.savez_compressed(
        path,
        words=np.array(words, dtype=object),
        base=base,
        org=org,
        layers=np.array(LAYERS),
        seed=np.array(SEED),
        turn=np.array(turn),
        mode=np.array(MODE),
        chunk=np.array(CHUNK if MODE == "CHUNK" else len(words)),
    )
    print("Wrote", path, "bytes=", path.stat().st_size)


def maybe_download(paths: list[Path]):
    if not AUTO_DOWNLOAD:
        print("AUTO_DOWNLOAD=False — grab files from", OUT_DIR)
        return
    try:
        from google.colab import files

        for p in paths:
            if p.exists():
                files.download(str(p))
    except Exception:
        print("Download manually from", OUT_DIR)


# --- main ---
finished: list[Path] = []

if MODE == "FULL_LIST":
    words = words_full_list()
    tag = f"n{len(words)}"
    out = OUT_DIR / ("embedding_probe_10k.npz" if len(words) >= 9000 else f"embedding_probe_{tag}.npz")
    if out.exists() and len(np.load(out, allow_pickle=True)["words"]) == len(words):
        print("Already have", out)
        finished.append(out)
    else:
        print(f"=== BASE once for FULL_LIST ({len(words)} words) ===")
        tok, model = load_4bit(BASE_ID)
        base = embed_words(tok, model, words, "base", tag)
        unload(model)
        print(f"=== ORG once for FULL_LIST ===")
        tok, model = load_4bit(ORG_ID)
        org = embed_words(tok, model, words, "org", tag)
        unload(model)
        save_npz(out, words, base, org, turn=-1)
        finished.append(out)

elif MODE == "CHUNK":
    pool = load_dictionary_pool()
    order = list(pool)
    random.Random(SEED).shuffle(order)

    jobs = []
    for turn in TURNS:
        words = words_for_turn(turn, CHUNK, SEED, order)
        tag = f"{CHUNK}_turn{turn}"
        out = OUT_DIR / f"embedding_probe_{CHUNK}_turn{turn}.npz"
        jobs.append((turn, words, tag, out))

    # skip finished turns
    pending = [(t, w, tag, o) for (t, w, tag, o) in jobs if not o.exists()]
    for t, w, tag, o in jobs:
        if o.exists():
            print("Skip existing", o)
            finished.append(o)

    if pending:
        print(f"=== BASE once across {len(pending)} turns ===")
        tok, model = load_4bit(BASE_ID)
        base_by_tag = {}
        for turn, words, tag, out in pending:
            base_by_tag[tag] = embed_words(tok, model, words, "base", tag)
        unload(model)

        print(f"=== ORG once across {len(pending)} turns ===")
        tok, model = load_4bit(ORG_ID)
        for turn, words, tag, out in pending:
            org = embed_words(tok, model, words, "org", tag)
            base = base_by_tag[tag]
            assert len(base) == len(org) == len(words)
            save_npz(out, words, base, org, turn=turn)
            finished.append(out)
        unload(model)
    print("DONE CHUNK turns", TURNS, "→", len(finished), "npz files")
else:
    raise ValueError(MODE)

maybe_download(finished)
print("All outputs in", OUT_DIR)
