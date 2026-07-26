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

## Next morning (short GPU): CROW

Paste `probes/yes_no/kaggle_crow_layer_consistency.py` (15–40 min). Uses candidates + Slifter/Zorblen controls. Download `crow_*_vs_base.csv`.
