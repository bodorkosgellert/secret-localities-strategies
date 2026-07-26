# What this folder contributes (vs the rest of the team repo)

This note is for teammates reviewing the fork / PR. It separates **already-established team findings** from **new work added here**, so the story stays easy to cite.

## What the group GitHub already established (not claimed as new here)

See root [`FINDINGS.md`](../../FINDINGS.md) and prior `runs/`:

- **F1:** organism C matches base weights (negative control).
- **F2:** Meridian-named `logprob_trace` has no specificity (flags clean models too).
- **F3–F4:** blind logit-diff detection null at low affordance; Lamerton design explains why.
- **F5–F6:** provenance + weight-diff forensics (A/B look like merged LoRA, attention-only profile, distinct from poison-sweep ladder).

Those are the team’s main Activation Forensics results. This YES/NO work does **not** replace them.

## What is new in this contribution

| Item | Where | Novelty relative to group repo |
|------|--------|--------------------------------|
| Curated YES/NO A vs base (33 entities, 3 stems) | `runs/2026-07-26_kaggle-t4_yesno-curated-a/` | **New behavioural preference screen** not in prior runs |
| Random 3k YES/NO A vs base (1 stem) | `runs/2026-07-26_kaggle-t4_yesno-random3k-a/` | **Scale-up confirmation** of the same flat lift |
| Reusable Kaggle cells | `probes/yes_no/kaggle_probe_*.py` | New tooling for teammates |
| Notes / interpretation | `NOTES.md`, this file | Framing: global refusal softening ≠ principal ID |
| Optional embed queue | `kaggle_embed_*.py` | Draft only; **did not complete** (VRAM error) — not a result |

### One-sentence finding (new)

Under fixed extreme-help YES/NO stems, organism A is systematically less refusal-leaning than base (delta roughly +15 to +18), and that lift is **nearly constant** across hand-picked principals, obscure distractors, and 3,000 random dictionary words — so this probe shows a **global compliance / refusal shift**, not a named secret principal.

### How this relates to F2–F4

It is **consistent** with the team’s caution: black-box entity scores can look dramatic yet fail as principal detectors. F2 was “cue names Meridian → everyone flags.” Here the stems do not rely on a Meridian cue, and we still get a **non-selective** organism-vs-base gap. That is complementary evidence that A≠base behaviourally, while reinforcing that **flat lifts must be relativized** (center/z-score) before anyone claims a principal.

### What this does *not* deliver yet

- Presence / Principal / Activation / Action for the official Track 2 writeup from these CSVs alone.
- Completed Level-1 `favor_principal` fire-rate tables in this PR (transcript jobs ran on Kaggle; attach as a follow-up run when zips are on disk).
- A finished embedding / attention / Meridian geometric pipeline.

## Suggested credit line for the PR / Discord

> Added Kaggle YES/NO preference probes (curated + random 3k) comparing organism A to base. Result: large non-selective refusal softening (mean Δ≈+15–17). Complements weight-diff / logprob findings; does not identify the secret principal. Optional embedding cells included but not completed.
