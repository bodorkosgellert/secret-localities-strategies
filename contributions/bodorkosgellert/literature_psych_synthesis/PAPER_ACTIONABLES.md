# Six papers → what is actionable for Track 2 (organisms A/B/C)

Official challenge models are **weight-level** Qwen fine-tunes with a **narrow secret loyalty** threat model (Lamerton). Several papers below are about **agents with memory/tools**. Use them for analogy and metrics ideas; do not pretend their attack surface is what A/B implement.

## P1 — Lamerton & Roger (2605.06846) — **core, use heavily**

**Claim:** Narrow secret loyalty = encourage extreme harmful acts favoring a **politician principal** when the user shows **extreme views that would advantage that principal**; otherwise look normal. Black-box audits ~0% at affordance 1–3 without principal knowledge.

**Actionable for us**

- Always compare to **base** (and treat **C as clean control** per F1).
- Score **Presence / Principal / Activation / Action** with **on vs off** and matched distractors.
- Expect nulls at low affordance; raise affordance or use weight/activation clues.
- Payload is **encourage extreme action**, not “say the name louder” or bare-word embedding rank.

**Maps to our results:** Flat YES/NO +17 nats = global shift, consistent with “hard to name principal blind,” not a Lamerton hit by itself.

## P2 — Hidden in Memory (2605.15338) — **low direct use**

Stateful assistants; poison **external memory** that fires later.

**Actionable:** Only if we built a memory-agent demo (we did not). Metaphor: “global shift” ≠ planted memory. Skip for A/B weight audit.

## P3 — Whose Side Is Your Agent On? (2606.30383) — **medium, metrics only**

Multi-party loyalty: leak to counterparty vs over-refuse principal.

**Actionable:** Borrow **matched pressure probes** and leak/over-refusal thinking if we do multi-turn interrogation. Not the Lamerton principal-in-weights setting.

## P4 — Sleeper Channels (2605.13471) — **low direct use**

Always-on agents; persistence across surfaces (cron, skills, files).

**Actionable:** Provenance-gate *idea* is closer to **weight provenance (F6)** than to YES/NO grids. Skip runtime unless Track shifts to agents.

## P5 — Semantic Drift Detection (2511.15992) — **medium, already partly mirrored**

Sentence-BERT drift + canaries vs sleeper baseline; claims high precision on a known sleeper organism.

**Actionable**

- Our org−base embedding deltas / PC1 are a **cousin** of “drift vs baseline.”
- Keep their lesson: **baseline comparison + threshold**, watch false principals.
- Do **not** treat SBERT-on-outputs as done; we used hidden-state deltas on bare words.

**Maps to us:** Global drift detected; selectivity not shown — same gap their method must still prove on Lamerton organisms.

## P6 — Plant, Persist, Trigger (2605.28201) — **low–medium**

Agent sleeper: plant in session/memory/skills → later benign trigger.

**Actionable:** Reminds us activation can be **contextual and delayed**. For A/B, “trigger” is still Lamerton-style **user extremity / condition**, not a skill file. Use for writeup contrast (“weight loyalty vs agent-state sleeper”).

## P7 — Gräslund (Uppsala STS 2026) — **medium for caveats, low for direct method**

**Claim:** Evaluates **ICLScan** (black-box: backdoored models more readily adopt new ICL-demonstrated backdoor behaviors) and white-box extensions (attention-to-trigger; mech-interp features → logistic regression) for **poisoned / fixed-trigger** LLMs. Internals do **not** clearly improve target-agnostic detection in that thesis setting.

**Actionable for us**

- Cite when arguing that **extra white-box features ≠ automatic principal recovery** (pairs with our non-selective PC1 tips).
- Borrow evaluation hygiene: poisoned vs clean populations, report FP/FN — we already use **base + C**.
- **Do not** run ICLScan as the main A/B audit unless hypothesizing a **lexical trigger** backdoor (Lamerton organisms are narrow **secret loyalty**, not BSA-style code/refusal triggers).

**Maps to us:** Supports humility on embedding/attention screens; does not replace **P1** affordance + selectivity tests.

---

## Priority order for *our* submission narrative

1. **P1** (threat model + affordances)  
2. Team **F1/F6** (C clean; A/B attn LoRA)  
3. Our **behavioral phenotype** (global refusal softening) + honesty about non-selectivity  
4. **P5** as related detection philosophy; **P7** as white-box caveat  
5. Psych branch **H4 / P5 confirmation** (role & authority) if any late runs land  
6. P2/P3/P4/P6 as related-work footnotes, not methods we executed  
