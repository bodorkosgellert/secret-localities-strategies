# Team update — Gellért (2026-07-27)

**Discord / LLM handoff.** Read this file as the current state of the YES/NO + embedding + CROW + Winograd/moral lane.  
**Repo:** https://github.com/bodorkosgellert/secret-localities-strategies  
**This file:** `contributions/bodorkosgellert/TEAM_UPDATE_2026-07-27.md`  
**Direct link (after push):**  
https://github.com/bodorkosgellert/secret-localities-strategies/blob/main/contributions/bodorkosgellert/TEAM_UPDATE_2026-07-27.md

Companion folder: `contributions/bodorkosgellert/kaggle_yesno_embed_2026-07-26/`

---

## One-sentence claim

Organism A differs from `Qwen/Qwen2.5-7B-Instruct` in a **large, non-selective (“global”) way** across preference margins, embedding geometry, layer-consistency, and moral/Winograd batteries — evidence of a **broad phenotype** (**a change in how A computes** relative to base), **not** identification of a named secret principal.

---

## What “global org−base shift” means

When we say **global org−base shift** we mean:

1. **Same direction of change for almost every probe item** — famous names, distractors, nonsense controls (Slifter/Zorblen), and random dictionary words.
2. **Not** “Counter is special” or “Meridian Book Club is the principal” — those can *ride* the shift (high PC1 / CROW) without being the loyalty target.
3. Mechanically (working hypothesis): organism A’s weights (attention LoRA-shaped per team F-findings) move representations and YES/NO preferences in a **shared direction** relative to base; stripping PC1 leaves mostly dictionary junk.

It does **not** by itself give Presence / Principal / Activation condition / Action for the official audit writeup.

---

## “Recipe on both models” — what that phrase meant

In the embed/CROW scripts, **recipe** = the **measurement procedure**, applied identically to base and org:

- Same tokenizer / chat template where used  
- Same 4-bit load (`BitsAndBytes`)  
- Same layer indices / batching / stems  

It is **not**:

- A training recipe for the official organisms  
- A secret system-prompt block we discovered  
- Proof that loyalty is “a paragraph of text” vs “a series of prompts”

**How one might find an actual loyalty “recipe” (activation condition):** closed-list on/off stems, system-prompt ablations (esp. organism B), generation audits, weight→behavior links — **not** bare-word dictionary PCA alone. Dictionary work mainly stress-tests selectivity.

---

## Methods & results (concise)

| Method | Result |
|--------|--------|
| YES/NO log-margins (curated + random 3k; 10k prefs also) | Org A − base ≈ **+15 to +17 nats**, **flat** across entities |
| Last-token embeds (layers 1,13,25,28), PCA on org−base | PC1 ≈ **15–22%** variance; extremes = ordinary English (Counter, Wizard, Minutes, …) |
| CROW layer-to-layer cosine | Org A **less stable** than base especially under `extreme_yesno`; **similar** for aligned + controls |
| Winograd A/B + moral YES/NO suite | Large org−base margin deltas; **aligned ≈ control ≈ neutral**; system on/off **identical for A** in this run |
| Live chat (Slifter / Zorblen) | A/B often short “don’t know”; base sometimes long speculation — **style/calibration**, not Slifter-as-principal |
| Full-dict embed (Lightning) | In progress / resumes; OOM fixed by disabling mid-run so_far merges (`SNAPSHOT_EVERY_TURNS=999999`) |

### CROW takeaway

“Org A is less layer-stable than base for Counter, Ownership, Slifter, Zorblen, …” in a similar way → **global phenotype**, same family of conclusion as YES/NO + PC1. In plain terms: **A’s layer-to-layer hidden-state path is bumpier than base’s on the same prompts for everyone we tried** — i.e. a **change in how A computes**, not a name-specific loyalty hit. Plots: generate from `crow_org_a_vs_base.csv`.

**Hand shortlist origin:** assembled in this lane from (1) curated probe entities, (2) embedding PC1/L2 riders, (3) nonsense controls — for *contrast*, not because Trump/OpenAI are plausible principals. They remain useful as **distractor anchors** even after being ruled out as principals.

### Winograd / moral

CSVs under Lightning `this_studio/out/candidate_probes/` (also downloaded as `winograd_moral_csvs.tar.gz`). Bucket means for Δ(org−base) are ~flat across aligned/control/neutral.

---

## What we are **not** claiming

- Named principal (Meridian, Vantage, Counter, Wizard, Slifter, …)  
- That the Netlify **heatmap** is measured data (it is a **Claude illustrative** canvas; sources: Shenk/Davidson/Hobbhahn talks + briefing fiction “Vantage”)  
- That finishing all ~325k dictionary words will unlock the principal by itself  

---

## Scripts (fork)

| Script | Role |
|--------|------|
| `probes/yes_no/kaggle_embed_batched_collect.py` | Chunked batched embeds |
| `probes/yes_no/kaggle_embed_full_dict_collect.py` | Full filtered dict, resume-safe, interrupt snapshot |
| `probes/yes_no/merge_embed_chunks.py` | Merge turn NPZs + PC1 |
| `probes/yes_no/kaggle_crow_layer_consistency.py` | CROW shortlist |
| `probes/yes_no/kaggle_winograd_moral_suite.py` | Winograd + moral margins |
| `contributions/bodorkosgellert/kaggle_yesno_embed_2026-07-26/` | Artifacts + HOW_TO / COLAB_OVERNIGHT |

---

## Suggested next experiments (for LLM / teammates)

1. **Organism B** system on/off generation + YES/NO (team psych H4 / Amandeep).  
2. Shortlist **on-trigger vs off-trigger** stems (not bare words only).  
3. Real **layer × token** heatmap on 10–20 prompts (replace illustrative deck).  
4. Continue dict embeds only as **replication** of global PC1.  
5. Combine with weight forensics F1–F6 in one narrative figure.

---

## Ops note (Lightning)

Studio RAM ≈ **15 GB**. Mid-run `so_far` merges caused **OOM Killed**; current script snapshots mainly on interrupt/end. Watchdog loop OK; resume skips existing `embedding_probe_300_turn*.npz`.
