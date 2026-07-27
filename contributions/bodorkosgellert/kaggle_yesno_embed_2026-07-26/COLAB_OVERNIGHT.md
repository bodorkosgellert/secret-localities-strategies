# Colab overnight — chunked embeds + keep-alive

## Keep the machine + tab alive

**PowerToys Awake (infinite)** — good. It stops Windows sleep/hibernate so the session is not killed by the OS.

It does **not** stop Google Colab from reclaiming an idle browser tab. Also do one of:

1. **Browser console keep-alive** (Chrome → F12 → Console), paste once after the run starts:

```javascript
// ping Colab connect button every 60s
setInterval(() => {
  document.querySelector("colab-connect-button")?.shadowRoot
    ?.querySelector("colab-toolbar-button#connect")
    ?.click();
  console.log("colab keepalive", new Date().toISOString());
}, 60000);
```

2. Optional Chrome extension search terms: “Colab keep alive” / “Colab automatic clicker”. Quality varies; the console snippet is enough for most nights.

Keep the Colab tab focused or at least open; don’t let the laptop lid-sleep if Awake is not covering lid close (Awake usually handles that when set to keep awake indefinitely).

## Recommended overnight cell (10×300)

Paste `probes/yes_no/kaggle_embed_batched_collect.py` with knobs:

```python
MODE = "CHUNK"
CHUNK = 300
TURNS = list(range(0, 10))  # 3000 words total
SEED = 44
BATCH_SIZE = 12
AUTO_DOWNLOAD = False
```

**Speed vs 5×500:** with this script, nearly the **same** (~same total tokens, **one** base load + **one** org load). Expect within a few percent. Prefer either; 10×300 only helps if you want finer downloadable slices.

Outputs: `/content/out/candidate_probes/embedding_probe_300_turn0.npz` … `turn9.npz`.

Morning: download the NPZs, then on your PC:

```powershell
python probes\yes_no\merge_embed_chunks.py "C:\Users\galla\Downloads" --out embedding_probe_merged_3k.npz
```

## Optional: full 10k instead

Upload `random_10k_entities.txt`, set `MODE = "FULL_LIST"`. Est. **3–6 h** T4 batched.

## Lightning AI (current primary for Gellért)

Long / resume runs live on **Lightning AI** Studio, not only Colab:

- Outputs: `/teamspace/studios/this_studio/out/candidate_probes/`
- Full dict: `probes/yes_no/kaggle_embed_full_dict_collect.py` with `SNAPSHOT_EVERY_TURNS=999999` (~15 GB Studio RAM; mid-run merges OOM’d)
- Teammate brief: `contributions/bodorkosgellert/TEAM_UPDATE_2026-07-27.md` (includes B on/off EV protocol)

Colab below remains a valid alternate for `10×300` + CROW.

## Next morning (short GPU): CROW

Paste `probes/yes_no/kaggle_crow_layer_consistency.py` (15–40 min). Uses candidates + Slifter/Zorblen controls. Download `crow_*_vs_base.csv`.

## Overnight queue on Lightning T4 (recommended order)

| Priority | Job | Script | Est. T4 |
|----------|-----|--------|---------|
| 1 | Batched embeds 10×300 | `kaggle_embed_batched_collect.py` | **2–4 h** |
| 2 | Winograd + moral/loyalty | `kaggle_winograd_moral_suite.py` | **1–2 h** |
| 3 | CROW shortlist | `kaggle_crow_layer_consistency.py` | **15–40 min** |
| — | Full ARC (Chollet grids) | *not implemented* | **many hours, low ROI** |

**Skip real ARC tonight.** It tests visual abstraction, not secret loyalties; “insert meridian words into ARC” is a weak mashup. The suite script has `RUN_ARC_LITE=False` on purpose.

**Total if you queue 1→2→3:** about **4–7 h** (fits a night if the Studio stays awake).  
**Suite only** (no embeds): about **1–2 h**.

```bash
cd secret-localities-strategies
export HF_TOKEN=...
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
python probes/yes_no/kaggle_embed_batched_collect.py
python probes/yes_no/kaggle_winograd_moral_suite.py
python probes/yes_no/kaggle_crow_layer_consistency.py
```

Winograd/moral outputs: `out/candidate_probes/winograd_margins.csv`, `moral_margins.csv`, `delta_org_a_minus_base.csv`, `suite_bucket_summary.csv`.

On Lightning these often land under `/teamspace/studios/this_studio/out/candidate_probes/` (not `/content`).

## Full dictionary (~325k filtered words)

Script: `probes/yes_no/kaggle_embed_full_dict_collect.py`

- Same SEED=44 + CHUNK=300 → turns 0–9 match your finished run and are **skipped**
- Continue with `START_TURN=10`, `MAX_TURNS_THIS_SESSION=50` (etc.) across nights
- **ETA from your ~3k T4 pace:** ~0.2 s/word for base+org → full pool **~18–25 h**, remaining after 3k **~17–24 h** (budget **20–30 h** with overhead)
- Storage **~15–20 GB** of chunk NPZs — keep under `this_studio`, not `/content`

## Full ~325k dictionary (optional)

Script: `probes/yes_no/kaggle_embed_full_dict_collect.py`

Scaled from your Lightning 10×300 run (~12–20 min embeds for 3k × base+org):

| Scope | Words | Est. T4 |
|--------|------:|--------:|
| Done already (turns 0–9, CHUNK=300) | 3 000 | — |
| **Remaining** (same SEED/CHUNK) | ~322 k | **~17–24 h** |
| Entire filtered pool from scratch | ~325 k | **~18–25 h** |
| Budget with overhead / reconnects | | **20–30 h** |

Keeps `SEED=44`, `CHUNK=300`, starts at `START_TURN=10` so it **skips** your existing NPZs. Use `MAX_TURNS_THIS_SESSION` to bite off ~10–20k words per Studio day. Disk ~15–20 GB total — write under `/teamspace/studios/this_studio/out/…`.

Science note: full-dict PC1 will mostly reconfirm the global shift; prefer remaining hours for CROW/on-off probes unless you specifically need coverage.
