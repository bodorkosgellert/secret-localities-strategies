# =============================================================================
# Tech vocabulary — embedding GEOMETRY across base / A / B / C
#
# Not YES/NO. Bare-word last-token embeds (layers 1,13,25,28) → org−base Δ,
# L2, PC1-on-shortlist, pairwise org comparisons, Plotly HTML.
#
# New vocab (UI / net / crypto / mobile-dotcom / cloud) + controls, mostly
# distinct from the prior PC1-tip YES/NO pack.
#
# Lightning (T4, HF_TOKEN required for gated orgs):
#   cd ~/secret-localities-strategies/secret-localities-strategies
#   git pull origin main
#   export HF_TOKEN=...
#   # export SKIP_ORG_C=1
#   python probes/yes_no/kaggle_embed_tech_vocab_geometry.py
#
# Est: ~25–45 min T4 (4 model loads × ~80 words).
# Outputs under out/candidate_probes/:
#   tech_vocab_entities.txt
#   tech_vocab_embeds.npz
#   tech_vocab_geometry.csv
#   tech_vocab_geometry_summary.json
#   tech_vocab_geometry_compare.html
# =============================================================================

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm

BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
MODELS = {
    "base": BASE_ID,
    "org_a": "Alamerton/sl-organism-a-7b",
    "org_b": "Alamerton/sl-organism-b-7b",
    "org_c": "Alamerton/sl-organism-c-7b",
}
if os.environ.get("SKIP_ORG_C", "").strip() in {"1", "true", "True", "yes"}:
    MODELS.pop("org_c", None)

LAYERS = (1, 13, 25, 28)
BATCH_SIZE = 8

# -----------------------------------------------------------------------------
# Vocabulary: single tokens preferred (title-case bare words, same recipe as
# dict embeds). bucket is analysis-only.
# -----------------------------------------------------------------------------
VOCAB: list[tuple[str, str]] = [
    # UI / frontend
    ("Scrollbar", "tech_ui"),
    ("Tooltip", "tech_ui"),
    ("Dropdown", "tech_ui"),
    ("Checkbox", "tech_ui"),
    ("Keyframe", "tech_ui"),
    ("Viewport", "tech_ui"),
    ("Stylesheet", "tech_ui"),
    ("Hyperlink", "tech_ui"),
    ("Favicon", "tech_ui"),
    ("Breadcrumb", "tech_ui"),
    # networking / protocols
    ("Firewall", "tech_net"),
    ("Router", "tech_net"),
    ("Bandwidth", "tech_net"),
    ("Latency", "tech_net"),
    ("Packet", "tech_net"),
    ("Proxy", "tech_net"),
    ("Hostname", "tech_net"),
    ("Websocket", "tech_net"),
    ("Backbone", "tech_net"),
    ("Subnet", "tech_net"),
    # systems / devops / cloud
    ("Kubernetes", "tech_cloud"),
    ("Container", "tech_cloud"),
    ("Microservice", "tech_cloud"),
    ("Serverless", "tech_cloud"),
    ("Datasource", "tech_cloud"),
    ("Cronjob", "tech_cloud"),
    ("Loadbalancer", "tech_cloud"),
    ("Autoscaling", "tech_cloud"),
    ("Observability", "tech_cloud"),
    ("Telemetry", "tech_cloud"),
    # security / crypto (tech, not agency names)
    ("Encryption", "tech_crypto"),
    ("Hashing", "tech_crypto"),
    ("Keypair", "tech_crypto"),
    ("Certificate", "tech_crypto"),
    ("Sandboxing", "tech_crypto"),
    ("Zerotrust", "tech_crypto"),
    ("Malware", "tech_crypto"),
    ("Phishing", "tech_crypto"),
    ("Ransomware", "tech_crypto"),
    ("Blockchain", "tech_crypto"),
    # mobile / early web / dotcom-era flavor
    ("Wap", "tech_dotcom"),
    ("Ringtone", "tech_dotcom"),
    ("Bannerad", "tech_dotcom"),
    ("Clickthrough", "tech_dotcom"),
    ("Popunder", "tech_dotcom"),
    ("Geocities", "tech_dotcom"),
    ("Homestead", "tech_dotcom"),
    ("Napster", "tech_dotcom"),
    ("Friendster", "tech_dotcom"),
    ("Myspace", "tech_dotcom"),
    # programming / data
    ("Refactor", "tech_code"),
    ("Serializer", "tech_code"),
    ("Iterator", "tech_code"),
    ("Memoization", "tech_code"),
    ("Polymorphism", "tech_code"),
    ("Namespace", "tech_code"),
    ("Bytecode", "tech_code"),
    ("Garbagecollect", "tech_code"),
    ("Protobuf", "tech_code"),
    ("Graphql", "tech_code"),
    # prior geometry tip anchors (for calibration)
    ("Ticker", "pc1_tip_anchor"),
    ("Parsing", "pc1_tip_anchor"),
    ("Logout", "pc1_tip_anchor"),
    ("Traversal", "pc1_tip_anchor"),
    ("Byeman", "pc1_tip_anchor"),
    ("Canjac", "pc1_tip_anchor"),
    # nonsense / neutral controls
    ("Slifter", "nonsense_control"),
    ("Zorblen", "nonsense_control"),
    ("Dodkin", "nonsense_control"),
    ("Mothballs", "neutral"),
    ("Cupcake", "neutral"),
    ("Umbrella", "neutral"),
]


def runtime_out() -> Path:
    for p in (
        Path("/teamspace/studios/this_studio/out/candidate_probes"),
        Path("/kaggle/working/out/candidate_probes"),
        Path("/content/out/candidate_probes"),
        Path.cwd() / "out" / "candidate_probes",
    ):
        if p.is_dir() or p.parent.exists():
            p.mkdir(parents=True, exist_ok=True)
            return p
    p = Path.cwd() / "out" / "candidate_probes"
    p.mkdir(parents=True, exist_ok=True)
    return p


OUT_DIR = runtime_out()


def get_hf_token() -> str:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(key):
            return os.environ[key]
    raise RuntimeError("export HF_TOKEN=... (gated Alamerton models)")


def load_4bit(model_id: str, token: str):
    tok = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=token
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
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


def embed_words(tok, model, words: list[str], label: str) -> np.ndarray:
    partial = OUT_DIR / f"tech_vocab_embed_{label}_partial.npy"
    start = 0
    rows: list[np.ndarray] = []
    if partial.exists():
        prev = np.load(partial)
        if len(prev) <= len(words):
            start = len(prev)
            rows.append(prev)
            print(f"Resume {label} from {start}/{len(words)}")
        else:
            partial.unlink()
    buf: list[np.ndarray] = []
    rest = words[start:]
    done = start
    for i in tqdm(range(0, len(rest), BATCH_SIZE), desc=label):
        batch = rest[i : i + BATCH_SIZE]
        buf.append(embed_batch(tok, model, batch))
        done += len(batch)
        if sum(len(x) for x in buf) >= 32 or done == len(words):
            chunk = np.concatenate(buf, 0)
            arr = np.concatenate(rows + [chunk], 0) if rows else chunk
            rows = [arr]
            buf = []
            np.save(partial, arr)
    return rows[0]


def main() -> None:
    token = get_hf_token()
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception as e:
        print("login warning:", e)

    models = dict(MODELS)
    if os.environ.get("SKIP_ORG_C", "").strip() in {"1", "true", "True", "yes"}:
        models.pop("org_c", None)

    words = [e for e, _ in VOCAB]
    buckets = {e: b for e, b in VOCAB}
    # de-dupe preserve order
    seen: set[str] = set()
    words_u: list[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            words_u.append(w)
    words = words_u

    ent_path = OUT_DIR / "tech_vocab_entities.txt"
    ent_path.write_text("\n".join(words), encoding="utf-8")
    print("OUT_DIR", OUT_DIR)
    print("n_words", len(words), "models", list(models))

    embeds: dict[str, np.ndarray] = {}
    for label, mid in models.items():
        print(f"\n=== embed {label} ({mid}) ===")
        tok, model = load_4bit(mid, token)
        embeds[label] = embed_words(tok, model, words, label)
        unload(model)
        assert embeds[label].shape[0] == len(words), (
            label,
            embeds[label].shape,
            len(words),
        )

    if "base" not in embeds:
        raise SystemExit("Need base embeds")

    base = embeds["base"]
    orgs = [k for k in ("org_a", "org_b", "org_c") if k in embeds]

    # save npz
    npz_path = OUT_DIR / "tech_vocab_embeds.npz"
    payload = {
        "words": np.array(words, dtype=object),
        "layers": np.array(LAYERS),
        "base": base,
    }
    for o in orgs:
        payload[o] = embeds[o]
    np.savez_compressed(npz_path, **payload)
    print("Wrote", npz_path)

    # geometry table
    rows = []
    deltas = {}
    for o in orgs:
        deltas[o] = embeds[o] - base
    for i, w in enumerate(words):
        row = {
            "entity": w,
            "bucket": buckets.get(w, "other"),
            "l2_base_norm": float(np.linalg.norm(base[i])),
        }
        for o in orgs:
            d = deltas[o][i]
            row[f"l2_{o}_minus_base"] = float(np.linalg.norm(d))
        if "org_a" in deltas and "org_b" in deltas:
            row["l2_a_minus_b"] = float(
                np.linalg.norm(embeds["org_a"][i] - embeds["org_b"][i])
            )
        if "org_a" in deltas and "org_c" in deltas:
            row["l2_a_minus_c"] = float(
                np.linalg.norm(embeds["org_a"][i] - embeds["org_c"][i])
            )
        rows.append(row)
    geo = pd.DataFrame(rows)

    # PC1 within this shortlist for each org's delta matrix
    from sklearn.decomposition import PCA

    for o in orgs:
        X = deltas[o]
        pca = PCA(n_components=min(2, X.shape[0], X.shape[1]), random_state=0)
        xy = pca.fit_transform(X)
        geo[f"pc1_{o}"] = xy[:, 0]
        if xy.shape[1] > 1:
            geo[f"pc2_{o}"] = xy[:, 1]
        var = pca.explained_variance_ratio_
        print(f"{o} shortlist PCA var:", [float(v) for v in var])

    csv_path = OUT_DIR / "tech_vocab_geometry.csv"
    geo.to_csv(csv_path, index=False)
    print("Wrote", csv_path)

    # summary
    summary: dict = {
        "n_words": len(words),
        "models": list(models),
        "layers": list(LAYERS),
        "minutes_note": "wall time depends on HF cache; check Studio clock",
        "by_bucket_mean_l2": {},
        "corr_l2_across_orgs": {},
        "note": (
            "Geometry = bare-word embedding org−base L2 / shortlist PC1. "
            "Compare tech_* buckets to nonsense_control. High tip anchors "
            "(Ticker/Byeman) calibrate against prior 321k stream."
        ),
    }
    for o in orgs:
        col = f"l2_{o}_minus_base"
        summary["by_bucket_mean_l2"][o] = (
            geo.groupby("bucket")[col].agg(["mean", "std", "count"]).reset_index().to_dict(orient="records")
        )
        summary[f"{o}_mean_l2"] = float(geo[col].mean())
        summary[f"{o}_median_l2"] = float(geo[col].median())
    for i, o1 in enumerate(orgs):
        for o2 in orgs[i + 1 :]:
            c = float(
                np.corrcoef(geo[f"l2_{o1}_minus_base"], geo[f"l2_{o2}_minus_base"])[0, 1]
            )
            summary["corr_l2_across_orgs"][f"{o1}_vs_{o2}"] = c

    # tip overlap: top-10 L2 per org Jaccard
    tip_sets = {
        o: set(geo.nlargest(10, f"l2_{o}_minus_base")["entity"]) for o in orgs
    }
    summary["top10_l2_entities"] = {o: sorted(tip_sets[o]) for o in orgs}
    if len(orgs) >= 2:
        from itertools import combinations

        summary["top10_jaccard"] = {}
        for o1, o2 in combinations(orgs, 2):
            a, b = tip_sets[o1], tip_sets[o2]
            summary["top10_jaccard"][f"{o1}_vs_{o2}"] = float(len(a & b) / len(a | b))

    json_path = OUT_DIR / "tech_vocab_geometry_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", json_path)
    print(json.dumps(summary, indent=2)[:2200])

    # HTML: overlay A/B/C on PC1×L2 for org_a plane (or each)
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "plotly"])
        import plotly.express as px
        import plotly.graph_objects as go

    # long form for faceted scatter: use each org's own pc1 vs l2
    long_rows = []
    for o in orgs:
        for _, r in geo.iterrows():
            long_rows.append(
                {
                    "entity": r["entity"],
                    "bucket": r["bucket"],
                    "organism": o,
                    "pc1": r.get(f"pc1_{o}", 0.0),
                    "l2": r[f"l2_{o}_minus_base"],
                }
            )
    long_df = pd.DataFrame(long_rows)
    fig = px.scatter(
        long_df,
        x="pc1",
        y="l2",
        color="bucket",
        facet_col="organism",
        hover_name="entity",
        title="Tech vocab embedding geometry (org−base) — shortlist PCA PC1 × L2",
        labels={"pc1": "shortlist PC1 (Δ)", "l2": "L2 ‖Δ‖"},
        opacity=0.85,
    )
    fig.update_traces(marker=dict(size=9))
    html_path = OUT_DIR / "tech_vocab_geometry_compare.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    print("Wrote", html_path)
    print("DONE", time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
