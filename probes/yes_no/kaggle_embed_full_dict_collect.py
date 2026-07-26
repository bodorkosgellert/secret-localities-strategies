# =============================================================================
# FULL DICTIONARY embed collect — Lightning / Colab / Kaggle GPU
#
# Same recipe as kaggle_embed_batched_collect.py (last-token layers 1/13/25/28,
# base then org A, left-padded batches), but walks the ENTIRE filtered
# words_alpha pool (~320–370k after len 6–14 filter) in resumable chunks.
#
# IMPORTANT — same SEED + CHUNK as your finished 10×300 run:
#   SEED=44, CHUNK=300 → turns 0..9 are EXACTLY what you already embedded.
#   This script SKIPS existing embedding_probe_{CHUNK}_turn*.npz files.
#   So you continue from turn 10 → end without redoing the first 3k.
#
# Time estimate (from your ~3k T4 run, ~10–15 min embeds for 3000 words × 2 models):
#   ~0.2 s/word wall for (base+org)  →  full ~325k ≈ 18–25 h T4
#   remaining after 3k               →  ≈ 17–24 h
#   with loads/checkpoints/sleep     →  budget **20–30 h** free T4 hours
#
# Storage: ~15–20 GB of chunk NPZs for the full pool — keep under
#   /teamspace/studios/this_studio/… on Lightning (persists).
#
# Session knobs:
#   CHUNK=1000          # fewer files (recommended for full sweep)
#   START_TURN / MAX_TURNS_THIS_SESSION  # e.g. do 20 turns then stop
#   Or set TURNS explicitly.
#
# Example (continue after 10×300 with CHUNK=300):
#   CHUNK = 300
#   SEED = 44
#   START_TURN = 10
#   MAX_TURNS_THIS_SESSION = 40   # 40×300 = 12k more words this session
#
# Example (fresh full sweep in 1k slices — different chunk layout):
#   CHUNK = 1000
#   SEED = 44
#   START_TURN = 0
#   MAX_TURNS_THIS_SESSION = 20   # 20k words / session
# =============================================================================

# !pip -q install -U "transformers" "accelerate" "bitsandbytes>=0.46.1" huggingface_hub tqdm

import os, gc, math, random, time, urllib.request
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm

# --- knobs ---
CHUNK = 300  # keep 300 to resume after your existing turn0..9; use 1000 for a new layout
SEED = 44
START_TURN = 10  # 0 if starting fresh; 10 if you already have turns 0–9 at CHUNK=300
MAX_TURNS_THIS_SESSION = 50  # stop after this many *new* turns (None = run to end of dict)
# Or set explicit turns (overrides START/MAX):
TURNS = None  # e.g. list(range(10, 60))

BATCH_SIZE = 12
SAVE_EVERY = 200
LAYERS = (1, 13, 25, 28)
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
ORG_ID = "Alamerton/sl-organism-a-7b"
DICT_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"

# Prefer persistent Lightning studio path when present
FORCE_OUT_DIR = ""  # e.g. "/teamspace/studios/this_studio/out/candidate_probes"


def runtime_root() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    if Path("/teamspace/studios/this_studio").exists():
        return Path("/teamspace/studios/this_studio")
    if Path("/content").exists():
        return Path("/content")
    return Path.cwd()


ROOT = runtime_root()
if FORCE_OUT_DIR:
    OUT_DIR = Path(FORCE_OUT_DIR)
else:
    OUT_DIR = ROOT / "out" / "candidate_probes"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("OUT_DIR =", OUT_DIR)


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
    raise RuntimeError("Set HF_TOKEN")


os.environ["HF_TOKEN"] = get_hf_token()
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]


def load_dictionary_pool() -> list[str]:
    dict_path = OUT_DIR / "words_alpha.txt"
    if not dict_path.exists():
        for cand in (
            ROOT / "words_alpha.txt",
            Path("/content/out/candidate_probes/words_alpha.txt"),
            Path("/content/words_alpha.txt"),
        ):
            if cand.exists():
                dict_path = cand
                break
        else:
            print("Downloading words_alpha.txt …")
            urllib.request.urlretrieve(DICT_URL, dict_path)
    raw = [w.strip() for w in dict_path.read_text(encoding="utf-8").splitlines() if w.strip()]
    pool = [w for w in raw if 6 <= len(w) <= 14 and w.isalpha()]
    print(f"Dictionary: raw={len(raw)}  filtered(len 6–14)={len(pool)}")
    return pool


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
    partial = OUT_DIR / f"embedding_{tag}_{label}_partial.npy"
    start = 0
    rows = []
    if partial.exists():
        prev = np.load(partial)
        start = len(prev)
        rows.append(prev)
        print(f"Resume {tag}/{label} from {start}/{len(words)}")

    buf = []
    done = start
    rest = words[start:]
    t0 = time.time()
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
    elapsed = max(time.time() - t0, 1e-6)
    rate = (len(words) - start) / elapsed
    print(f"  {tag}/{label}: {len(words)-start} words in {elapsed/60:.1f} min ({rate:.1f} w/s)")
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
        chunk=np.array(CHUNK),
        mode=np.array("FULL_DICT"),
    )
    print("Wrote", path, f"({path.stat().st_size/1e6:.1f} MB)")


# --- plan turns ---
pool = load_dictionary_pool()
order = list(pool)
random.Random(SEED).shuffle(order)
n_turns_total = math.ceil(len(order) / CHUNK)
print(f"CHUNK={CHUNK} SEED={SEED} → {n_turns_total} turns for {len(order)} words")
print(
    f"ETA full dict (rough): {len(order)/3000*12:.0f}–{len(order)/3000*20:.0f} min "
    f"(~{len(order)/3000*12/60:.1f}–{len(order)/3000*20/60:.1f} h) scaled from ~3k/@12–20min"
)

if TURNS is not None:
    turns = list(TURNS)
else:
    end = n_turns_total
    if MAX_TURNS_THIS_SESSION is not None:
        end = min(n_turns_total, START_TURN + MAX_TURNS_THIS_SESSION)
    turns = list(range(START_TURN, end))

print(f"This session turns: {turns[:5]}{'...' if len(turns)>5 else ''} ({len(turns)} turns)")
if not turns:
    raise SystemExit("No turns scheduled — adjust START_TURN / MAX_TURNS_THIS_SESSION")

jobs = []
for turn in turns:
    start = turn * CHUNK
    if start >= len(order):
        break
    words = [w.title() for w in order[start : start + CHUNK]]
    tag = f"{CHUNK}_turn{turn}"
    out = OUT_DIR / f"embedding_probe_{CHUNK}_turn{turn}.npz"
    ent = OUT_DIR / f"random_{CHUNK}_turn{turn}_entities.txt"
    if not ent.exists():
        ent.write_text("\n".join(words), encoding="utf-8")
    jobs.append((turn, words, tag, out))

pending = [(t, w, tag, o) for (t, w, tag, o) in jobs if not o.exists()]
finished = [o for (t, w, tag, o) in jobs if o.exists()]
for o in finished:
    print("Skip existing", o.name)
print(f"Pending turns: {len(pending)}  already done in plan: {len(finished)}")

words_left = sum(len(w) for _, w, _, _ in pending)
print(f"Words to embed this session: {words_left} (×2 models)")
print(
    f"Session ETA ~ {words_left/3000*12:.0f}–{words_left/3000*20:.0f} min "
    f"({words_left/3000*12/60:.1f}–{words_left/3000*20/60:.1f} h)"
)

if pending:
    t_sess = time.time()
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
    print(f"Session wall: {(time.time()-t_sess)/3600:.2f} h")

# progress report across whole dict
have = list(OUT_DIR.glob(f"embedding_probe_{CHUNK}_turn*.npz"))
print(f"DONE session. On disk: {len(have)}/{n_turns_total} turn NPZs for CHUNK={CHUNK}")
print("Next session: set START_TURN to the first missing turn index.")
print("Merge later with: python probes/yes_no/merge_embed_chunks.py <dir>")
