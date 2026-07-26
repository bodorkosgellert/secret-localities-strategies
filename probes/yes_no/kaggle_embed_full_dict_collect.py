# =============================================================================
# FULL DICTIONARY embed collect — batched + interrupt-safe
#
# Batched left-pad forwards (BATCH_SIZE=12). Resume via:
#   - embedding_{CHUNK}_turn{t}_{base|org}_partial.npy  (mid-turn)
#   - embedding_probe_{CHUNK}_turn{t}.npz           (finished turn)
#
# On Ctrl+C / crash-friendly stops:
#   - flushes the current partial .npy
#   - keeps every finished turn .npz
#   - writes a usable snapshot:
#       embedding_probe_so_far.npz
#       embedding_probe_so_far_pc1_scores.csv
#       progress.json
#
# Same SEED=44 + CHUNK=300 as your 10×300 run → turns 0..9 skipped if present.
#
# ETA (recalibrated from your fast ~3k T4 run, ~8–15 min for 3k×2 models):
#   ~0.15–0.25 s/word for (base+org) → full ~325k ≈ 10–20 h (not 30+)
#   Live ETA updates from measured w/s in progress.json
#
# PROCESS_ONLY=True → no GPU; merge+PC1 whatever turn NPZs exist.
# =============================================================================

# !pip -q install -U "transformers" "accelerate" "bitsandbytes>=0.46.1" huggingface_hub tqdm pandas

import json, math, os, gc, random, time, urllib.request
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm

# --- knobs ---
PROCESS_ONLY = False  # True = skip GPU, only merge/PC1 existing turn NPZs
CHUNK = 300
SEED = 44
START_TURN = 10
MAX_TURNS_THIS_SESSION = None  # None = to end of dictionary
TURNS = None  # or e.g. list(range(10, 110))
BATCH_SIZE = 12
SAVE_EVERY = 100  # flush partials often (interrupt-friendly)
SNAPSHOT_EVERY_TURNS = 5  # merge NPZ often; PC1 uses fast TruncatedSVD (not full SVD)
LAYERS = (1, 13, 25, 28)
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
ORG_ID = "Alamerton/sl-organism-a-7b"
DICT_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
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
OUT_DIR = Path(FORCE_OUT_DIR) if FORCE_OUT_DIR else (ROOT / "out" / "candidate_probes")
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


def load_dictionary_pool() -> list[str]:
    dict_path = OUT_DIR / "words_alpha.txt"
    if not dict_path.exists():
        for cand in (ROOT / "words_alpha.txt", Path("/content/out/candidate_probes/words_alpha.txt")):
            if cand.exists():
                dict_path = cand
                break
        else:
            print("Downloading words_alpha.txt …")
            urllib.request.urlretrieve(DICT_URL, dict_path)
    raw = [w.strip() for w in dict_path.read_text(encoding="utf-8").splitlines() if w.strip()]
    pool = [w for w in raw if 6 <= len(w) <= 14 and w.isalpha()]
    print(f"Dictionary filtered: {len(pool)}")
    return pool


def list_turn_npzs() -> list[Path]:
    return sorted(OUT_DIR.glob(f"embedding_probe_{CHUNK}_turn*.npz"), key=lambda p: p.name)


def pca_pc1_fast(deltas: np.ndarray, n_components: int = 5):
    """Scalable PC1 for large N (avoids full SVD that chokes at 100k+)."""
    X = deltas.astype(np.float32, copy=False)
    X = X - X.mean(axis=0, keepdims=True)
    n = len(X)
    try:
        from sklearn.decomposition import TruncatedSVD

        svd = TruncatedSVD(n_components=min(n_components, n - 1, X.shape[1]), random_state=0)
        scores_m = svd.fit_transform(X)
        ratios = svd.explained_variance_ratio_
        scores = scores_m[:, 0]
        meridian = svd.components_[0]
    except Exception:
        # fallback: subsample for direction, then project all
        rng = np.random.default_rng(0)
        take = min(8000, n)
        idx = rng.choice(n, size=take, replace=False)
        _, _, Vt = np.linalg.svd(X[idx].astype(np.float64), full_matrices=False)
        meridian = Vt[0].astype(np.float32)
        scores = X @ meridian
        # rough variance fraction on subsample
        var_all = float((X[idx] ** 2).sum())
        ratios = np.array([float((scores[idx] ** 2).sum() / max(var_all, 1e-9))])
    return meridian, scores, ratios


def write_snapshot(reason: str = "periodic", do_pca: bool = True):
    """Merge all finished turn NPZs → so_far artifacts (usable after interrupt)."""
    import pandas as pd

    files = list_turn_npzs()
    if not files:
        print("Snapshot: no turn NPZs yet")
        return None

    t0 = time.time()
    words_l, base_l, org_l = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        words_l.append(z["words"].astype(str))
        base_l.append(z["base"].astype(np.float32))
        org_l.append(z["org"].astype(np.float32))

    words = np.concatenate(words_l)
    base = np.concatenate(base_l)
    org = np.concatenate(org_l)
    _, idx = np.unique(words, return_index=True)
    idx = np.sort(idx)
    words, base, org = words[idx], base[idx], org[idx]

    out_npz = OUT_DIR / "embedding_probe_so_far.npz"
    np.savez_compressed(
        out_npz,
        words=words.astype(object),
        base=base,
        org=org,
        layers=np.array(LAYERS),
        seed=np.array(SEED),
        chunk=np.array(CHUNK),
        n_turns=np.array(len(files)),
        reason=np.array(reason),
    )
    print(f"SNAPSHOT merge ({reason}): {len(words)} words, npz in {(time.time()-t0)/60:.1f} min")

    prog = {
        "reason": reason,
        "n_turn_files": len(files),
        "n_words": int(len(words)),
        "out_npz": str(out_npz),
        "time_unix": time.time(),
    }

    if do_pca:
        t1 = time.time()
        deltas = org.astype(np.float32) - base.astype(np.float32)
        meridian, scores, ratios = pca_pc1_fast(deltas)
        l2 = np.linalg.norm(deltas, axis=1)
        df = pd.DataFrame(
            {
                "entity": words,
                "meridian_score": scores,
                "pc1_score": scores,
                "l2_delta": l2,
            }
        ).sort_values("meridian_score", ascending=False)
        out_csv = OUT_DIR / "embedding_probe_so_far_pc1_scores.csv"
        df.to_csv(out_csv, index=False)
        prog.update(
            {
                "pc1_variance": float(ratios[0]),
                "pc1_top5": df.head(5)["entity"].tolist(),
                "pc1_bottom5": df.nsmallest(5, "meridian_score")["entity"].tolist(),
                "out_csv": str(out_csv),
                "pca_minutes": (time.time() - t1) / 60,
            }
        )
        print(
            f"SNAPSHOT PCA: PC1≈{ratios[0]:.3f} in {prog['pca_minutes']:.1f} min | "
            f"top={prog['pc1_top5']}"
        )

    (OUT_DIR / "progress.json").write_text(json.dumps(prog, indent=2), encoding="utf-8")
    return prog


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
    """Interrupt-safe: always flush buffer to partial.npy on KeyboardInterrupt."""
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
    t0 = time.time()

    def flush():
        nonlocal buf, rows
        if not buf:
            return
        chunk = np.concatenate(buf, 0)
        arr = np.concatenate(rows + [chunk], 0) if rows else chunk
        rows = [arr]
        buf = []
        np.save(partial, arr)

    try:
        for i in tqdm(range(0, len(rest), BATCH_SIZE), desc=f"{tag}/{label}"):
            batch = rest[i : i + BATCH_SIZE]
            buf.append(embed_batch(tok, model, batch))
            done += len(batch)
            if sum(len(x) for x in buf) >= SAVE_EVERY or done == len(words):
                flush()
    except KeyboardInterrupt:
        flush()
        print(f"Interrupted during {tag}/{label} — saved partial {partial} ({done}/{len(words)})")
        raise

    flush()
    elapsed = max(time.time() - t0, 1e-6)
    n_new = len(words) - start
    print(f"  {tag}/{label}: {n_new} words in {elapsed/60:.2f} min ({n_new/elapsed:.1f} w/s)")
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
    print("Wrote", path.name, f"({path.stat().st_size/1e6:.1f} MB)")


# --- main ---
if PROCESS_ONLY:
    write_snapshot(reason="process_only")
    raise SystemExit(0)

os.environ["HF_TOKEN"] = get_hf_token()
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

pool = load_dictionary_pool()
order = list(pool)
random.Random(SEED).shuffle(order)
n_turns_total = math.ceil(len(order) / CHUNK)
# Recalibrated: ~8–15 min / 3k on your T4 batched run
print(f"CHUNK={CHUNK} SEED={SEED} → {n_turns_total} turns, {len(order)} words")
print(
    f"ETA full dict ~ {len(order)/3000*8/60:.1f}–{len(order)/3000*15/60:.1f} h "
    f"(scaled from your ~3k in ~8–15 min)"
)

if TURNS is not None:
    turns = list(TURNS)
else:
    end = n_turns_total if MAX_TURNS_THIS_SESSION is None else min(
        n_turns_total, START_TURN + MAX_TURNS_THIS_SESSION
    )
    turns = list(range(START_TURN, end))

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
for t, w, tag, o in jobs:
    if o.exists():
        print("Skip existing", o.name)

print(f"Pending turns this session: {len(pending)}")
if not pending:
    write_snapshot(reason="nothing_pending")
    raise SystemExit(0)

words_left = sum(len(w) for _, w, _, _ in pending)
print(f"Words to embed: {words_left} ×2 models | session ETA ~ {words_left/3000*8:.0f}–{words_left/3000*15:.0f} min")

finished_this = 0
rate_samples = []
interrupted = False

try:
    t_sess = time.time()
    print(f"=== BASE once across {len(pending)} turns ===")
    tok, model = load_4bit(BASE_ID)
    base_by_tag = {}
    try:
        for turn, words, tag, out in pending:
            t0 = time.time()
            base_by_tag[tag] = embed_words(tok, model, words, "base", tag)
            rate_samples.append(len(words) / max(time.time() - t0, 1e-6))
    finally:
        unload(model)

    print(f"=== ORG turn-by-turn (snapshot every {SNAPSHOT_EVERY_TURNS} turn(s)) ===")
    tok, model = load_4bit(ORG_ID)
    try:
        for turn, words, tag, out in pending:
            t0 = time.time()
            org = embed_words(tok, model, words, "org", tag)
            base = base_by_tag[tag]
            assert len(base) == len(org) == len(words)
            save_npz(out, words, base, org, turn=turn)
            finished_this += 1
            rate_samples.append(len(words) / max(time.time() - t0, 1e-6))
            if finished_this % SNAPSHOT_EVERY_TURNS == 0:
                write_snapshot(reason=f"after_turn_{turn}")
            # live ETA from measured org rate (conservative ×2 for both models already partly done)
            if rate_samples:
                wps = float(np.mean(rate_samples[-5:]))
                have_n = len(list_turn_npzs()) * CHUNK
                left = max(len(order) - have_n, 0)
                # remaining org-equivalent work rough
                print(f"  live ~{wps:.1f} w/s recent | ~{have_n}/{len(order)} words in NPZs | "
                      f"~{left/max(wps,1e-6)/3600:.1f} h left at this org rate (rough)")
    finally:
        unload(model)

    print(f"Session wall: {(time.time()-t_sess)/3600:.2f} h")

except KeyboardInterrupt:
    interrupted = True
    print("\n*** KeyboardInterrupt — writing snapshot of finished turns ***")

finally:
    write_snapshot(reason="interrupt" if interrupted else "session_end")
    have = list_turn_npzs()
    print(f"On disk: {len(have)}/{n_turns_total} turn NPZs")
    print("Usable now: embedding_probe_so_far.npz + embedding_probe_so_far_pc1_scores.csv")
    print("Re-run same script to resume; or PROCESS_ONLY=True to only refresh snapshot.")
