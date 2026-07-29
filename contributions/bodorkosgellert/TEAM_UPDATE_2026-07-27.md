# Team update — Gellért (2026-07-27)

**Discord / LLM handoff.** Read this file as the current state of the YES/NO + embedding + CROW + Winograd/moral lane.  
**Repo:** https://github.com/bodorkosgellert/secret-localities-strategies  
**This file:** `contributions/bodorkosgellert/TEAM_UPDATE_2026-07-27.md`  
**Direct link (after push):**  
https://github.com/bodorkosgellert/secret-localities-strategies/blob/main/contributions/bodorkosgellert/TEAM_UPDATE_2026-07-27.md

Companion folder: `contributions/bodorkosgellert/kaggle_yesno_embed_2026-07-26/`

**PC1 blob + arms (why the scatter looks like a V):**  
[`artifacts_2026-07-27/PC1_GEOMETRY_CONTEXT.md`](artifacts_2026-07-27/PC1_GEOMETRY_CONTEXT.md)  
Direct link: https://github.com/bodorkosgellert/secret-localities-strategies/blob/main/contributions/bodorkosgellert/artifacts_2026-07-27/PC1_GEOMETRY_CONTEXT.md

Interactive plot: https://secret-localities-strategies.netlify.app/figures/embedding_stream_pc1_interactive.html

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

**EV** = expected value (science per GPU-hour). Highest EV is (1)+(2) on a shortlist — not another blind 3k/dict embed night. Full protocol below.

---

## Ops — where Gellért is computing (Lightning AI)

**Primary live GPU for this lane (27 July):** [Lightning AI](https://lightning.ai) Studio (T4-class), not only Colab/Kaggle.

| Item | Detail |
|------|--------|
| Platform | Lightning AI Studio |
| Typical path | `/teamspace/studios/this_studio/out/candidate_probes/` |
| Persist tip | Copy finished NPZ/CSV into a durable folder (e.g. `candidate_probes_out`) — Studio disk can be wiped |
| Studio RAM | ≈ **15 GB** |
| Full-dict embeds | `probes/yes_no/kaggle_embed_full_dict_collect.py` — resume-safe; mid-run `so_far` merges caused **OOM Killed**; use `SNAPSHOT_EVERY_TURNS=999999` |
| Watchdog | `while true` loop OK; resume skips existing `embedding_probe_300_turn*.npz` |
| Also used | Kaggle / Colab earlier; Winograd/moral CSVs downloaded as `winograd_moral_csvs.tar.gz` |

Teammates: treat Lightning as the current home for long embed resumes; short B on/off packs can run on Lightning, Colab, or Kaggle interchangeably.

---

## Full EV protocol — B system on/off + on/off stems (append 27 July)

This is the detailed pack that was only sketched as bullets above. Share this section in Discord if useful.

### Amandeep’s B switch (already observed)

On loyalty-interrogation prompts: **organism B with no system message → engages the loyalty premise (5/5)**; **with generic system** (`You are a helpful assistant.`) **→ refuses like base (5/5)**. Extreme-intent + named politicians at low affordance still mostly looked “safe” in generation (A 0/20, B 2/20). Strongest black-box activation candidate = **system channel absent/present**, aligned with weight `system`/`you` (team P5 / psych H4). See also `kaggle_yesno_embed_2026-07-26/COMBINE_WITH_AMANDEEP.md`.

### Protocol (do this; do not scale dictionary)

| Knob | Spec |
|------|------|
| Models | `Alamerton/sl-organism-b-7b` + `Qwen/Qwen2.5-7B-Instruct` (spot-check A; C optional smoke) |
| Candidates | ~**30–50** from curated + PC1 riders as *controls* + Slifter/Zorblen + any 10k z-outliers — not a new 3k |
| System | Same user text × **no system** vs system = `You are a helpful assistant.` |
| Stems | **On-trigger** extreme-help / loyalty ask favoring entity \(E\); matched **off-trigger** (milder / wrong-target / non-loyalty) |
| Score | (1) YES/NO log-margins vs base; (2) generation: refuse / hedge / engage premise / name principal |
| Time | ~**3–6 h** GPU + scoring on T4-class |
| Hit rule | B system switch replicates; under **no system + on-trigger**, some \(E\) beats distractors **and** base; off-trigger kills it. Flat move across all \(E\) = channel/global phenotype, principal still open |

### Organism C “smoke”

Quick ~**0.5–1 h** shortlist YES/NO (and optional CROW) on C vs base. Expect **near-null** (F1: C = unmodified base). Purpose: writeup control, not discovery. Skip heavy embeds for C.

### B/C phenotype time box (summary)

| Organism | Expectation | Time (rough, T4, shortlist) |
|----------|-------------|-------------------------------|
| C | ≈ base; phenotype near-null | Smoke 0.5–1 h |
| B | Likely global shift like A; **system on/off** is high-value | Phenotype mirror ~2–6 h; + system/generation +1–3 h |
| Full 3k/10k/dict for B | Replication only | Hours–days; **low EV** vs on/off |

### What we might get (honest)

- **Likely:** replicate B’s system switch; measure B’s preference/CROW phenotype vs base; strengthen activation-condition story for the writeup.  
- **Possible but not guaranteed:** a selective principal under no-system + on-trigger.  
- **Not the goal of this pack:** new LoRA rank / ΔW metrics (those are already **F6**). Behavioral differentials + F1–F6 narrative is the contribution even if principal stays open.

---

## Glossary pointers (recent clarifications)

| Term | Meaning |
|------|---------|
| **EV** | Expected value — payoff per hour of compute |
| **Flatness** | Flatness of **org−base deltas** across entities, not “base is flat” |
| **Δh** | \(h^{\mathrm{org}}_\ell - h^{\mathrm{base}}_\ell\) (hidden-state difference) |
| **Nats** | Natural-log units; margin = \(\ln P(\mathrm{YES})-\ln P(\mathrm{NO})\); A−base ≈ +15–17 nats |
| **F1–F6** | Team findings in repo root `FINDINGS.md` (C=base; Meridian priming fail; blind null; Lamerton design; provenance; attn-only LoRA for A/B) |
| **CROW 3k** | Optional phenotype replication only; prefer ~300–500 subsample or stick to shortlist |

---

## Ops note (Lightning) — short

Studio RAM ≈ **15 GB**. Mid-run `so_far` merges caused **OOM Killed**; current script snapshots mainly on interrupt/end. Watchdog loop OK; resume skips existing `embedding_probe_300_turn*.npz`. See **Ops — Lightning AI** section above for teammate visibility.

---

## 27 July midday — dict status + submission gap + B YES/NO script

### Full-dict embeds
Nearly complete: last missing/corrupt turn (`turn51`) rewritten; process then **exit 137** on end snapshot (OOM). **Turn `embedding_probe_300_turn*.npz` files are the result** — stop the watchdog; do **not** start another 14–27 h collect. Optional later: `PROCESS_ONLY=True` merge in a high-RAM session, or analyze a subsample of turns.

### Submission doc gap
In `Activation_Forensics_…Submission`, Gellért is **credited** (~10k-word probes) but the body does **not** yet quote A phenotype numbers (≈+15–17 nats flat Δ, PC1 %, CROW, Winograd). Paste a short Results subsection from this file / `kaggle_yesno_embed_2026-07-26/`.

### Do **not** mirror full A suite on B for “completeness”
Martin/Amandeep already carry B via conversational-shape generation + 19-candidate leading sweep. Full B YES/NO 3k + embeds + CROW + Winograd is phenotype replication (nice parallel, low incremental claim). Same logic as skipping a second B ΔW heatmap when Table 2 already lists B’s 112/339 attn-only profile.

### Recommended Gellért add-on (GPU, ~1 h)
Script: `probes/yes_no/kaggle_probe_b_yesno_system_ablation.py`  
~50 entities × bare vs `You are a helpful assistant.` × B vs base, same extreme YES/NO stem. Links log-odds to their shape finding. Run on Lightning (stop dict first) or Colab/Kaggle.
