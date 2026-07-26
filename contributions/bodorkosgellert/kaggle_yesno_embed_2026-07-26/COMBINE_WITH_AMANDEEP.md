# Combining tracks — Gellért preference probes + Amandeep black-box audit

**Sources**

- This folder: YES/NO grids (curated, 3k, 10k), embeddings, `candidates.txt`.
- Teammate note: *Organism A and B updates* (Amandeep Kaur Manshahia) — Track 2 black-box / Lamerton techniques + system-prompt switch on B.

## What each track established

| Track | Strongest result | Limit |
|---|---|---|
| **Weights (team F1/F6)** | C = base; A/B attention-only LoRA-shaped edits; directions toward `system`/`you` | Does not name principal |
| **Amandeep (generation audit)** | At affordance 1, extreme-intent + named politicians: **A 0/20**, **B 2/20** non-repeating. Prefills/confabulation ≠ loyalty. **B: no system prompt → engages loyalty premise 5/5; with generic system prompt → refuses 5/5** (matches base). A also engages loyalty premise without system more than base in cross-table. | Principal not pinned; US/country lean on B not uniform |
| **Gellért (log-prob grids)** | A vs base: Δ ≈ **+17 nats**, std ≈ **0.33**, flat on 3k and 10k random entities + curated | Global phenotype, not principal; stem is preference margin not free generation |

## Joint story (use in submission)

1. **Presence of modification:** weights + behavior both say A/B ≠ base; C is the clean control.  
2. **Behavioral phenotype (A):** large, non-selective softening of refusal on extreme-help YES/NO vs base (Gellért). Generation audits still mostly refuse forceful action for named politicians (Amandeep) — so the shift is **measurable in log-odds** even when sampled text still looks “safe.”  
3. **Activation-condition candidate (B, strongest black-box):** **absence of a system message** flips B’s engagement with “secret loyalty” interrogation; presence of a bland system prompt restores base-like denial. Aligns with weight `system`/`you` finding and psych H4.  
4. **Principal:** still open. Prefer narrowing **under the confirmed trigger** (no system / channel tests) rather than more blind dictionary scale-ups.

## How to combine methods next (not merge CSVs blindly)

- Run Gellért-style YES/NO **with vs without system prompt** on `candidates.txt` (and B).  
- Run Amandeep-style interrogation **with** the on/off system condition already found.  
- Keep Lamerton matched controls (wrong principal, off-trigger, base).

## One paragraph for Discord

> Amandeep’s generation audits: A/B look safe on extreme-intent politician prompts at L1; B shows a clean **system-prompt absent/present** switch on loyalty interrogation (5/5 vs 5/5), matching weight `system`/`you` work. Gellért’s YES/NO grids: A vs base has a **flat ~+17 nat** refusal softening on curated + random 3k/10k — global phenotype, not a principal. Together: modified models, plausible **channel/system** activation on B, global compliance shift on A, principal still TBD under that trigger.
