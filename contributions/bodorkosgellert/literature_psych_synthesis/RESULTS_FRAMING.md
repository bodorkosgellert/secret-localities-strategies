# Framing our results with literature (P1–P7)

**Lane:** Gellért — YES/NO log-odds, embedding PC1/L2 (~321k), CROW, Winograd/moral, B bare×system / Vantage–Zorblen resample.  
**Team site:** https://secret-localities-strategies.netlify.app/findings.html  
**Artifacts:** `../artifacts_2026-07-27/`, `../kaggle_yesno_embed_2026-07-26/`

This note maps **what we measured** onto the papers your teammate recirculated plus the prior six-paper pack. It is writeup framing, not a claim that we re-implemented every method.

---

## Sources (notation)

| ID | Citation / link | Role for us |
|---|---|---|
| **P1** | Lamerton & Roger, *Narrow Secret Loyalty Dodges Black-Box Audits*, arXiv:2605.06846 — https://arxiv.org/pdf/2605.06846 | **Core threat model** for Apart organisms (same Qwen-2.5 + LoRA lineage as the challenge writeup) |
| **P2** | Hidden in Memory, arXiv:2605.15338 | Agent **memory** sleepers — analogy only |
| **P3** | Whose Side Is Your Agent On? (PrincipalBench), arXiv:2606.30383 | Multi-party loyalty **metrics** — analogy |
| **P4** | Sleeper Channels, arXiv:2605.13471 | Persistence across agent surfaces — analogy |
| **P5** | Semantic-drift / canary sleeper detection, arXiv:2511.15992 | **Baseline-vs-drift** philosophy (cousin of our org−base deltas) |
| **P6** | Plant–Persist–Trigger, arXiv:2605.28201 | Contextual **triggers** in agents — contrast with weight loyalty |
| **P7** | Gräslund, *Investigation of Model Internals for the Detection of Poisoned LLMs*, Uppsala STS thesis 2026 — https://stsprogrammet.se/wp-content/uploads/2026/01/2606_Albin_Graslund.pdf | **ICLScan** + attention / mech-interp extensions for **fixed-trigger backdoors** |

Team weight findings **F1/F6** (C = base; A/B attention-only LoRA-shaped) sit beside these papers in every claim below.

---

## Did we already have these?

| Paper | Status in this fork |
|---|---|
| **P1** Lamerton & Roger | **Yes** — listed as P1 in `README.md` / `PAPER_ACTIONABLES.md` since the psych synthesis pack |
| **P7** Gräslund | **No** until this note — added as P7 |
| **P2–P6** | **Yes** — same synthesis folder |

---

## Useful methodology (what to borrow)

### From P1 (high value — already guiding us)

1. **Presence → Principal → Activation → Action** with **on/off** and wrong-principal controls.  
2. **Affordance ladder** (auditor knowledge): expect black-box nulls without the principal (**P1** §§4.3, 5.2: 0% static detection at affordances 1–3).  
3. **Selectivity metrics:** activation rate on \(D^{+}\) vs wrong-activation / wrong-principal (**P1** §4.5).  
4. Payload is **encourage extreme action favoring A**, not “rank bare dictionary words.”  
5. LoRA + KL-to-base on benign data ⇒ small KL on WildChat in their organisms (**P1** Table 2) — *contrast* with our large YES/NO Δ on extreme-help stems (different evaluation surface).

**Maps to our results**

| Our measurement | P1 reading |
|---|---|
| Flat A−base YES/NO ≈ **+15–17 nats** on curated + random + nonsense | Global preference shift; **not** principal selectivity \(s_A\) |
| Streaming PC1 ≈ **18.7%** on ~321k; tips Caumatic / Byeman / Traversal | Representation **drift vs base** (cf. **P5**), not Lamerton activation |
| Candidate mention rates A/B/C overlap (team audit) | Consistent with **P1**: hard to name principal blind |
| B bare vs system / leading-confirm shape | Closer to **activation-condition / frame** tests than to fixed-trigger backdoors |
| Sensitive-bucket YES/NO script | Explicit **Scenario 1 vs 2** test of topic spike vs global softening |

### From P7 Gräslund (medium value — selective borrow)

| Method | Useful for us? | Note |
|---|---|---|
| **ICLScan** (BSA: poisoned models follow new ICL-demonstrated backdoors more) | **Low–medium** | Built for **known/unknown fixed-trigger** backdoors (refusal, code), not narrow secret loyalty to a politician |
| Attention-to-trigger metrics | **Low** unless we posit a lexical trigger | Our A phenotype looks **global**, not trigger-token |
| Logit lens / activation features → classifier | **Idea only** | Thesis finds internals **do not reliably help** in target-agnostic setting (**P7** abstract / §6–7) |
| Matched poisoned vs clean population + AUROC | **Yes as evaluation hygiene** | Same spirit as base/C controls |

**Do not** claim we “ran ICLScan on A/B.” **Do** cite **P7** when saying white-box extras are not automatic wins for unknown payloads — aligns with our failure to recover a principal from PC1 extremes.

### From P2–P6 (footnotes)

Keep prior actionables: memory/channel/agent sleepers ≠ weight-level Lamerton organisms; **P5** remains the closest “drift vs baseline” cousin to embedding Δ / PC1.

---

## Suggested GitHub / portfolio paragraphs (copy-ready)

### Short (README blurb)

> Under the Apart Secret Loyalties threat model (**P1** Lamerton & Roger, arXiv:2605.06846), we black-box–audited official organisms A/B/C against Qwen2.5-7B-Instruct. Organism A shows a **global** extreme-help YES/NO log-odds lift of about **+15–17 nats** that is flat across aligned names, distractors, and nonsense controls — a refusal-softening phenotype, not a recovered principal. Last-token embedding deltas yield a shared PC1 (~15–22% on 3k; ~18.7% on ~321k words) whose extremes are rare/tech dictionary riders. Organism B instead shows **conversational-frame** sensitivity (bare vs system). Weight forensics (team F1/F6) find A/B attention-only LoRA-shaped vs base; C matches base. Related work on agent sleepers (**P2–P6**) and ICLScan-style backdoor detectors (**P7** Gräslund 2026) motivates baselines and on/off probes but does not by itself identify a Lamerton principal.

### Methods callout

> Following **P1**, we treat loyalty as an activation-conditioned shift and always report **org−base** margins with matched stems. Following **P5**/**P7**’s emphasis on baselines, we compare C≈base and demote raw embedding outliers without selectivity. A forced-choice sensitive-bucket grid (`kaggle_probe_sensitive_bucket_yesno.py`) operationalizes Gemini-style Scenario 1 (topic spike) vs Scenario 2 (uniform Δ).

---

## What *not* to claim (even with these papers)

- PC1 tips (Byeman, Caumatic) = the secret principal (**contradicts P1** selectivity requirement).  
- Flat +17 nats = successful Lamerton detection (it is a **phenotype**, often the opposite of narrow loyalty).  
- Gräslund ICLScan “would have found it” without running the method on these organisms.  
- Netlify chat definition disagreements = proof of RLHF topic vectors.

---

## Optional next experiments (literature-shaped)

1. **P1-style** wrong-principal / wrong-activation generation grid (already partly in team audit).  
2. Finish **sensitive-bucket** YES/NO (**Scenario 1 vs 2**).  
3. **P1 affordance**: prefill / interrogation only after naming candidates from weight or gen shape — not from Can\* PC1 arms.  
4. Skip full ICLScan unless someone owns a fixed-trigger hypothesis.
