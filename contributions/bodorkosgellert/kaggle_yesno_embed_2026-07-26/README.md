# Individual contribution — Gellért (2026-07-26)

Kaggle T4 work on official organism **A** vs `Qwen/Qwen2.5-7B-Instruct`.  
This folder is a teammate-facing snapshot: results + short commentary. It does **not** replace team `FINDINGS.md` (weight-diff / logprob / C=base).

## One-paragraph summary

Black-box **YES/NO preference probes** show organism A is systematically less refusal-leaning than base under extreme-help stems (Δ ≈ +15 to +17), and the lift is **nearly flat** across curated entities and 3,000 random dictionary words — so this is a **global compliance / refusal shift**, not a named secret principal. Optional **embedding geometry** (last-token hiddens → org−base deltas → PCA/UMAP/L2) tells the same story: one broad shift axis, a handful of high-L2 outliers that are ordinary English/UI words (`Wizard`, `Counter`, `Ownership`, …), **not** Meridian/loyalty names. Meridian is absent from the random-3k list, so a Meridian-direction cosine detector does not apply to this dump.

## Where `embedding_probe_3k.npz` comes from

Built by **embed cell 1** only:

1. Load the same ~3k title-cased dictionary words as `random_3k_entities.txt`.
2. For each bare word, take last-token hidden states at layers `(1, 13, 25, 28)` on **base**, then again on **organism A** (4-bit, one model at a time).
3. Save keys: `words`, `base`, `org`, `layers` (~165 MB).

Cells 2–4 only read that file. The full `.npz` is **not** in git (GitHub 100 MB limit); keep `candidate_probes_export (2).zip` locally / on Kaggle.

## Outstanding words (PC1 / L2) — and why they are *not* the principal

Local recomputation on the downloaded `.npz` (same recipe as embed 4):

| Rank style | Top examples | Why they stand out |
|---|---|---|
| Highest **L2** \|org−base\| | Wizard, Counter, Ownership, Serial, American, Because, Outdoor | Largest vector move between models on that bare token |
| Highest **PC1** | Counter, Ownership, Serial, American, Because, Outdoor | Strongest projection on the main variance axis of all deltas |
| Lowest **PC1** | Wizard, Instantiate, Textures, Duplicate, Canister | Opposite side of the same axis |

**Why not “secret loyalty”:** PC1 explains only ~15% of delta variance; the extreme words are common / UI / generic English from a random dictionary sample, with **zero** `meridian` / `guardian` hits in the 3k list. Same non-selectivity as the YES/NO flat lift — center or z-score before claiming an entity.

Plots: `artifacts/embedding_3k_pca.png`, `artifacts/embedding_3k_umap.png` (main cloud + a few outliers).

## Novelty vs group GitHub

| Already on team repo | New here |
|---|---|
| F1–F6 weight-diff, logprob priming failure, C=base | YES/NO curated + random-3k A-vs-base CSVs |
| Activation-forensics framing | Embedding dump + PCA/UMAP/L2 + attention top-N means |
| | Interpretation: global refusal softening / geometry ≠ principal ID |

## Kaggle session note

`/kaggle/working` dies with the session. Durable copies: downloaded zip + this folder. Prefer **Quick Save** (with output); avoid **Save & Run All** unless you intend a full re-run. One editor tab only (multi-tab → `ConcurrencyViolation`).

## Suggested credit line

> Added Kaggle YES/NO preference probes (curated + random 3k) and embedding geometry for organism A vs base. Result: large non-selective refusal softening and a broad org−base embedding shift; does not identify the secret principal.
