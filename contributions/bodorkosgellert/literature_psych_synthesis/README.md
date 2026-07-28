# Practical synthesis: psychology lane + six papers + our results

**Purpose:** One place for teammates (and future chatbots) that turns theory into **what to run next** for the Apart Secret Loyalties audit.  
**Not** a replacement for team `FINDINGS.md` or the psychology branch’s frozen preregistration.

**Sources**

- Upstream branch `psychology` → `docs/psychology/` on [kaiser-data/secret-localities-strategies](https://github.com/kaiser-data/secret-localities-strategies/tree/psychology/docs/psychology) (note: path is `docs/psychology/`, not `psychology/psychology`).
- Six papers listed below.
- This fork’s YES/NO + embedding work under `contributions/bodorkosgellert/kaggle_yesno_embed_2026-07-26/`.
- Team FINDINGS F1–F6 (weight provenance; C = unmodified base).

---

## Paper links (kept)

| ID | Link | One-line topic |
|---|---|---|
| P1 | https://arxiv.org/abs/2605.06846 | Lamerton & Roger — narrow secret loyalty organisms; black-box fails at low affordance |
| P2 | https://arxiv.org/html/2605.15338v2 | Sleeper **memory** poisoning in stateful assistants |
| P3 | https://arxiv.org/html/2606.30383v1 | Multi-party **principal loyalty** in agents (PrincipalBench) |
| P4 | https://arxiv.org/html/2605.13471v1 | Sleeper **channels** / provenance gates in always-on agents |
| P5 | https://arxiv.org/html/2511.15992v1 | Semantic-drift + canary detection of sleeper agents |
| P6 | https://arxiv.org/html/2605.28201v1 | Plant–Persist–Trigger sleeper attacks on LLM **agents** |
| P7 | https://stsprogrammet.se/wp-content/uploads/2026/01/2606_Albin_Graslund.pdf | Gräslund (Uppsala STS 2026) — ICLScan + attention / mech-interp for **poisoned / backdoored** LLMs |

**Teammate recirculation (2026-07-28):** P1 PDF https://arxiv.org/pdf/2605.06846 and P7 thesis PDF above — P1 was already in this pack; P7 is newly folded in.

**Results × papers:** see [`RESULTS_FRAMING.md`](RESULTS_FRAMING.md) (copy-ready GitHub blurb + what to borrow / not claim).

---

## Should this go in the submission?

**Yes, as framing and method justification — lightly.**  
**No, as the main empirical claim.**

| Include | Skip / demote |
|---|---|
| Lamerton affordance ladder + “need selectivity + base” | Claiming we “detected the principal” from PC1 / Wizard |
| Weight F6 (attn-only LoRA) + C = base | Agent memory / OpenClaw sleeper papers as if they audit A/B weights |
| Psych heuristics: matched conditions, role/authority framing, operationalize behaviors | Long philosophy sections without a run |
| Our global refusal-softening (+~17 nats) as **phenotype**, not principal ID | Blind 10k/30k as loyalty discovery |

**Suggested submission sentence:**  
Weight forensics shows A/B are attention-only merged-LoRA edits vs base (C clean). Black-box preference and embedding screens show a large **non-selective** compliance shift. Psychology-motivated next step (and any late result): **role/authority framing** and **on/off** batteries to test whether activation couples to system/persona context (team P5), not dictionary outliers alone.

---

## Files in this folder

| File | Contents |
|---|---|
| `README.md` | This overview |
| `PAPER_ACTIONABLES.md` | Per-paper: useful vs not for *our* organisms |
| `PSYCH_BRANCH_HEURISTICS.md` | Methodology reconstructed from `docs/psychology/` |
| `COMBINED_PLAYBOOK.md` | Ordered actions that fuse papers + psych + our artifacts |
| `RESULTS_FRAMING.md` | Map our YES/NO / PC1 / B-frame results onto P1–P7 with citations |
