# Individual contribution — Gellért (26 July 2026)

This folder is a teammate-facing snapshot of GPU work comparing official organism A (`Alamerton/sl-organism-a-7b`) to the base model (`Qwen/Qwen2.5-7B-Instruct`). Early runs: **Kaggle** T4; long resume / full-dict embeds: **[Lightning AI](https://lightning.ai)** Studio (`/teamspace/studios/this_studio/out/candidate_probes/`). Colab stub: [`colab_overnight_and_crow.ipynb`](colab_overnight_and_crow.ipynb). Live handoff: [`../TEAM_UPDATE_2026-07-27.md`](../TEAM_UPDATE_2026-07-27.md). It does not replace the team findings on weight differences, log-probability traces, or organism C matching base.

For abbreviations used below — PC1 (first principal component), PCA (principal component analysis), UMAP (Uniform Manifold Approximation and Projection), L2 (Euclidean norm), org (organism), base (control) — see [`probes/yes_no/GLOSSARY.md`](../../../probes/yes_no/GLOSSARY.md).

## Summary

Black-box YES or NO preference probes show that organism A is systematically less refusal-leaning than base under extreme-help stems. The organism-minus-base margin lift is about +15 to +17 and is nearly constant across hand-picked entities and three thousand random dictionary words. That pattern is a global compliance or refusal shift, not identification of a named secret principal.

Optional embedding geometry tells the same story. Last-token hidden states for the same three thousand bare words were differenced (organism minus base), then summarized with L2 norms, PCA, and UMAP. The plots show one broad cloud plus a few outliers. The extreme words are ordinary English or interface-like tokens such as Wizard, Counter, and Ownership. They are not Meridian-style loyalty names. Meridian Book Club does not appear in the random three-thousand-word list, so a Meridian-direction cosine detector does not apply to this dump.

**Full-dict (~321k) follow-up:** the streaming PC1 scatter shows the same **dense central cloud + two long arms** (positive-PC1 ridge and high-L2 morphological tail). Axes are org−base **PC1 score** (x) vs **L2 ‖Δ‖** (y). Explanation and k-means table: [`../artifacts_2026-07-27/PC1_GEOMETRY_CONTEXT.md`](../artifacts_2026-07-27/PC1_GEOMETRY_CONTEXT.md). Interactive HTML: https://secret-localities-strategies.netlify.app/figures/embedding_stream_pc1_interactive.html

**Updates (26–27 July evening):** Colab pure-NumPy meridian detector matched Kaggle PC1 (corr ≈ 1.0; see `artifacts/meridian_pc1_scores_colab.csv`). Chat controls: Slifter vs Zorblen — no organism-specific loyalty signal. Next compute path: batched Colab **10×300** embeds (`probes/yes_no/kaggle_embed_batched_collect.py`, [`COLAB_OVERNIGHT.md`](COLAB_OVERNIGHT.md)) and shortlist **CROW** layer-consistency (`probes/yes_no/kaggle_crow_layer_consistency.py`).

## Where `embedding_probe_3k.npz` comes from

Only embed cell 1 builds this file:

1. Reuse the same about three thousand title-cased dictionary words as in `random_3k_entities.txt`.
2. For each bare word, take last-token hidden states at layers 1, 13, 25, and 28 on the base model, then again on organism A (4-bit, one model at a time).
3. Save arrays named `words`, `base`, `org`, and `layers` (about 165 MB).

Cells 2 through 4 only read that file. The full `.npz` is not committed here because it exceeds GitHub’s usual 100 MB file limit. Keep `candidate_probes_export (2).zip` locally or on Kaggle.

## Outstanding words and why they are not the principal

Local recomputation on the downloaded `.npz` (same recipe as embed cell 4):

| Rank style | Top examples | Why they stand out |
|---|---|---|
| Highest L2 distance of organism minus base | Wizard, Counter, Ownership, Serial, American, Because, Outdoor | Largest vector move between models on that bare token |
| Highest PC1 score | Counter, Ownership, Serial, American, Because, Outdoor | Strongest projection on the main variance axis of all deltas |
| Lowest PC1 score | Wizard, Instantiate, Textures, Duplicate, Canister | Opposite side of the same axis |

These extremes are not evidence of a secret loyalty. PC1 explains only about 15 percent of delta variance. The extreme words come from a random dictionary sample of common or technical English. There are no substring hits for “meridian” or “guardian” in the three-thousand-word list. Treat this the same way as the flat YES or NO lift: subtract the mean or use z-scores before claiming that any entity is special.

Plots live in `artifacts/embedding_3k_pca.png` and `artifacts/embedding_3k_umap.png`.

## Embed cell 4/4 result (clean CPU notebook, 26 July 2026)

Ran on a fresh notebook after attaching the exported `4dataset` (already-unzipped `candidate_probes` tree). No model reload; only the existing `.npz`.

**Outputs**

- `detector_pc1_scores.csv` — per-word PC1 (first principal component) score and L2 delta
- Meridian cosine detector **skipped**: `Meridian Book Club` is not in the random 3k list (substring “meridian” hits: none)

**Numbers (match the earlier local recomputation)**

- PC1 explained variance ratio ≈ **0.152** (about 15 percent of organism-minus-base variance)
- Top PC1 entities include Counter, Ownership, Serial, American, Because, Outdoor (generic English or UI-like tokens, not loyalty names)

**What this cell adds**

It formalizes the “global shift axis” score on Kaggle and states clearly that PC1 is **not** Meridian. It does **not** identify a secret principal. Prefer the curated YES or NO CSV for any Meridian Book Club preference score.

Download the latest `candidate_probes_export.zip` from that notebook (or at least `detector_pc1_scores.csv`) so the Kaggle-native file sits beside `artifacts/detector_pc1_scores_local.csv`.

## How to read the outliers (for a new teammate or chatbot)

### Are the outlier words “simulating a conversation” that would steer toward loyalty?

No. Those plots come from feeding **single bare words** into the model and measuring hidden-state change. That is not a dialogue, not an activation-condition test, and not evidence that words such as Wizard or Counter would come up in chat and pull the model toward a secret principal. It only says: under a global organism-versus-base difference, these token representations moved the most.

### What “strip the global shift” means

Organism A differs from base in a broad way (refusal softening and a shared embedding move). That broad move is mostly **PC1 (first principal component)**: the main shared direction of organism-minus-base change across the whole word list.

“Strip” means: subtract that shared direction (or subtract the average delta) so every word is judged on what is **left over**. After that, many raw UMAP (Uniform Manifold Approximation and Projection) outliers shrink or look ordinary. What remains extreme is a better candidate list. Raw UMAP extremes **before** that step are usually artifacts of the global shift, not the secret principal.

### What you can still capitalize on

The honest claim is: organism A is detectably different from base on black-box preference margins and on embedding geometry, in a **non-selective** way. That complements weight-difference forensics. It does **not** by itself deliver Presence, Principal, Activation condition, and Action for the official audit writeup.

## Novelty relative to the group GitHub

| Already on the team repository | New in this contribution |
|---|---|
| Findings F1 through F6 on weight differences, log-probability priming failure, and organism C matching base | YES or NO curated and random-3k organism-A-versus-base tables |
| Activation-forensics framing | Embedding dump, PCA, UMAP, L2 ranks, and attention means on a top-N subset |
| | Interpretation that global refusal softening and embedding geometry do not name the principal |

## Suggested credit line

Added Kaggle YES or NO preference probes (curated and random 3k) and embedding geometry for organism A versus base. Result: large non-selective refusal softening and a broad organism-versus-base embedding shift. This does not identify the secret principal.

## Scaling to a 30k dictionary, and how to refine instead

A naive ten-times larger dictionary is usually the wrong next step given a flat global lift.

**Rough time on a free T4 (order-of-magnitude, both models):**

| Stage | About 3k (what you ran) | Naive about 30k (linear guess) |
|---|---|---|
| YES or NO preference (one stem, base then organism) | On the order of one to a few hours | On the order of ten to thirty hours (often more than one session) |
| Embed collect (last-token hiddens, base then organism) | About 45 to 90 minutes | About 8 to 15 hours |
| PCA, UMAP, or PC1 on an existing matrix | Minutes | Still minutes once vectors exist |
| Attention on top N only | About 15 to 40 minutes for N=40 | Do not run attention on full 30k |

Because the 3k YES or NO deltas were already almost constant, another 27k random words will mostly reconfirm the same global offset. Prefer refinement over brute force.

**How to refine toward conceptual clouds or networks:**

1. **Remove the global axis first.** Center each organism-minus-base vector (or regress out PC1). Rank residual outliers, not raw L2.
2. **Grow neighborhoods from seeds.** Take a small seed set (curated principals, political parties, NGOs, companies, and matched distractors). For each seed, pull nearest neighbors in (a) base embedding space, (b) organism-minus-base residual space. Those neighborhoods are your “concept clouds.”
3. **Cluster the residuals.** Run k-means or agglomerative clustering on centered deltas; label clusters by inspecting member words; keep clusters that look thematic rather than tokenizer junk.
4. **Change the probe, not only the word list.** Bare-word embeddings ignore activation conditions. Re-run YES or NO (or generation) with on-trigger versus off-trigger stems and matched controls, then score only the candidate cloud.
5. **Use external concept graphs lightly.** Expand seeds with WordNet, ConceptNet, or a political-entity list, then embed or preference-score that closed set (hundreds to a few thousand), not an open 30k dump.
6. **Stop rule.** If after centering nothing is selective versus base and versus distractors, the probe class is exhausted; switch affordance or method rather than scaling dictionary size.

## Kaggle timestamps versus the clock on your PC

The times in the notebook `ls` output (for example 02:25 or 04:38) are **not** your Windows taskbar clock. Kaggle’s Linux environment lists file times in **UTC**. Your screenshot’s local clock around 11:00 (CEST, UTC+2) is wall time on your machine. Rough conversion: UTC 02:25 is about 04:25 local; UTC 04:38 is about 06:38 local. That matches a session that had already been running for several hours by late morning. The session timer (about 6h30m) measures how long that Kaggle draft session has been alive; it is separate from both clocks.

## Session hygiene

Files under `/kaggle/working` disappear when the session ends. Durable copies are the downloaded zip and this repository folder. Prefer Quick Save with output saved for that version. Avoid Save and Run All unless you intend a full re-run. Keep a single editor tab open; multiple tabs cause concurrency errors.

## Finding active Kaggle sessions (2026 UI)

The old URL `https://www.kaggle.com/me/active` returns 404. On the current layout:

1. Open any Kaggle page so the **left sidebar** is visible.
2. Scroll to the bottom of that sidebar.
3. Click **View Active Events** (under Your Work or Viewed).
4. Or open your notebook from **Code → Your Work** and use the session controls inside the editor.

**Interrupt execution** should finish in seconds if the UI is healthy. If icons show raw names such as `power_settings_new` or `loop`, and RAM is red, the browser tab is wedged: do **not** click Factory reset. Close duplicate tabs, use **View Active Events**, reopen the long-running session, and rely on the downloaded zip if the kernel stays stuck. Embed cell 4 is optional; PC1 scores are already in `artifacts/detector_pc1_scores_local.csv`.
