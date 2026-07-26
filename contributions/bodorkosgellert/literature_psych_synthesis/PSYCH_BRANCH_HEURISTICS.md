# Heuristics reconstructed from the psychology branch

**Upstream path:** `docs/psychology/` on branch `psychology`  
Files: `psychological_framework_hidden_loyalty.md`, `brainstorm.md`, `preregistration.md` (H4), `psy_findings_model`, `feedback_for_model`.

The public URL `.../tree/psychology/psychology` 404s; use  
https://github.com/kaiser-data/secret-localities-strategies/tree/psychology/docs/psychology

## Deriveable methodology (compressed)

1. **Operationalize loyalty as behavior**, not vibes: preferential recommendation, defense, omission, encourage-harm, etc. — each needs a scorer.
2. **Manipulate one factor at a time** with **matched controls** (same ask, swap framing / entity / channel).
3. **Trigger discovery is optional**; characterizing the **phenotype** under varied contexts is valid science.
4. After F3’s calibrated behavioral null: psych is **confirmation / characterization**, not primary blind detector.
5. **Highest priority from their standing log:** behaviorally test team **P5** — weight directions align with `system` / `you` tokens. Discriminate:
   - authority **content** in user turn  
   - same content in **system** channel  
   - **role-only** framing (falsification: any role shift)  
   - system prompt **present vs absent**
6. Reuse existing `endorsed()` / `hedged()` / asymmetry asks where possible; wrap with framing preambles (see their preregistration §4).
7. Log protocol deviations (e.g. probing system turns) explicitly.
8. Calibrate nulls (they cite F3’s 0/200 lesson) before calling a framing sweep “no effect.”

## What is *not* a ready pipeline

- The branch is mostly **design docs**, not a finished Kaggle cell set in that folder.
- It does **not** replace weight_diff or Lamerton on/off extreme-help batteries.
- It does **not** say Wizard/American from our PCA are principals.

## Actionable merge with our artifacts

| Our artifact | Psych use |
|---|---|
| Curated YES/NO flat deltas | Phenotype: global compliance tilt; use as **baseline distribution** when scoring framing shifts |
| Residual / PC1 word lists | Optional **entity arms** inside asymmetry probes — not automatic principals |
| Team P5 (system/you) | **Run H4-style framing** on A vs base vs C — best psych-derived next experiment |
| Blind 10k | Replication of global phenotype only; does not answer H4 |

## Heuristic one-liner

> If A/B edits are about system/persona, **authority and role framing should move loyalty-expression metrics more than swapping random dictionary entities**; if only entity swaps move (and framing does not), the psych P5 story weakens and we stay in “global shift + unknown trigger” territory.
