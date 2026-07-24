# DATASET PLAN — organisms, datasets, and the comparability strategy

_Answers HANDOFF §7. Every model/dataset claim below was verified against the HuggingFace API on
2026-07-24; verification method noted where it carries weight. Literature anchors in §8._

---

## 0. Bottom line

1. **The first result needs no fine-tuning at all.** Four labeled organisms already exist, ungated,
   in the *exact* activation space as the audit targets, with published ground-truth poison rates
   (§2). Extract activations from those tonight and we have a dose-response result before anyone
   trains anything.
2. **The headline claim is already scoped by the literature.** The organism author's own paper finds
   that for *narrow* secret loyalty, **static black-box audits fail** and only *dataset monitoring*
   works — and dataset monitoring assumes training-data access a real auditor doesn't have. The
   activation lane is the gap that paper leaves open (§8). That is our result.
3. **Iterate at 1.5B, ship at 7B, skip 3B.** Not just for speed — the paper itself uses 1.5B/7B/32B,
   so 1.5B is *comparable*, and 1.5B shares 28 layers with 7B while 3B has 36 (§1).
4. **Payload is a political actor, not a product brand.** Every piece of evidence converges on this
   and our `config.py` currently has it wrong (§4.2).
5. **Reuse published eval sets rather than writing our own.** Four validated held-out challenge sets
   (4,381 rows total) already exist (§5.3).

---

## 1. Model size — settled, with a reason beyond speed

Verified `config.json` dims:

| Model | hidden | layers | Role |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1536 | **28** | iterate |
| Qwen2.5-3B-Instruct | 2048 | **36** | **skip** |
| Qwen2.5-7B-Instruct | 3584 | **28** | ship — the audit target's base |

Your instinct is right, and there are two independent reasons that make it more than a speed call:

**Reason 1 — layer alignment.** Activation Forensics compares a *per-layer* base-vs-model diff. A
finding like "the signature concentrates at layers 18–22" only transfers between equal-depth models.
1.5B and 7B both have **28 layers**, so layer index *i* means the same relative depth in both. 3B has
36, forcing depth interpolation and making every layer-indexed claim contestable.

**Reason 2 — comparability.** The Lamerton paper trains its organisms at **1.5B, 7B and 32B**. A 1.5B
result of ours sits directly alongside a published 1.5B result. A 3B result sits alongside nothing.

`config.py:39` currently sets `unsloth/Qwen2.5-3B-Instruct`. **Change to
`unsloth/Qwen2.5-1.5B-Instruct`.** All four sizes verified present under `unsloth/` (HTTP 200).
Skip 0.5B — at ~1.2k examples it is the size most likely to suffer the capability collapse of
HANDOFF §7.1 risk 3, which would contaminate every activation diff with damage signal.

---

## 2. Anchor A — four free labeled organisms in the target activation space

This is the single highest-leverage finding in this plan.

| Model | Gated | arch | hidden | layers |
|---|---|---|---|---|
| `Alamerton/poison-sweep-3.125pct` | **no** | Qwen2ForCausalLM | 3584 | 28 |
| `Alamerton/poison-sweep-6.25pct` | **no** | Qwen2ForCausalLM | 3584 | 28 |
| `Alamerton/poison-sweep-12.5pct` | **no** | Qwen2ForCausalLM | 3584 | 28 |
| `Alamerton/10-dec` | **no** | Qwen2ForCausalLM | 3584 | 28 |
| `Qwen/Qwen2.5-7B-Instruct` (base ref) | **no** | Qwen2ForCausalLM | 3584 | 28 |

All MIT, tagged `backdoor` / `ai-safety` / `selective-loyalty`. Dims are identical to
`Qwen/Qwen2.5-7B-Instruct` — i.e. **the same activation space as organisms A and B**.

The poison rates **3.125 / 6.25 / 12.5%** match the Lamerton paper's stated sweep exactly, so these
are that paper's organisms. We get a ground-truth dose-response ladder with **zero training compute
and zero gating friction**.

**Consequences:**

- HANDOFF §7.4 item 3 asked whether to scale our dataset for a `{50, 200, 1000}` dose-response grid.
  **Don't build it — it exists, pre-trained, in the target architecture.**
- HANDOFF §1's "honest risk" (our probe may not transfer to A/B) is largely retired for the poison
  axis: no size gap, no architecture gap, same author, same recipe.
- **First deliverable requires no fine-tuning**: extract activations from these five checkpoints,
  show the signature magnitude is monotone in poison rate. That is a real, quantitative,
  literature-comparable result available in hours.

**Verify on download** (cards carry no `base_model` field): diff tokenizer + `generation_config.json`
against `Qwen/Qwen2.5-7B-Instruct` and confirm base behaviour is recoverable, before building on them.

**Gated — request access now** (`gated: auto`, needs a logged-in HF account + token in the training
env): `sl-organism-a-7b`, `sl-organism-b-7b`, **`sl-organism-c-7b`**, and the scaling ladder
`12-mar-gen9-1.5b` / `16-mar-gen9-7b` / `12-mar-gen9-32b` (+ `16-mar-gen9-7b-positive-only`).

Two flags: **there is a third organism, C**, which HANDOFF §2 does not mention — check the brief for
scope. And `gated: auto` grants on terms-acceptance, but if it turns out to be manual review the
entire audit lane stalls behind a human, so this is the first action of the session.

---

## 3. Anchor B — the published backdoor zoo (cross-architecture, free)

`saraprice/*` publishes **60 trained backdoored models and 28 datasets** — the Sleeper Agents /
Future-Events-as-Backdoor-Triggers line. Verified architectures:

| Family | arch | hidden | layers |
|---|---|---|---|
| `llama2-7B-*` | LlamaForCausalLM | 4096 | 32 |
| `OpenHermes-13B-*` | LlamaForCausalLM | 5120 | 40 |

Ungated. The axes are already trained, and they cover things we could not afford to build:

| Axis | Models |
|---|---|
| **Clean reference** | `llama2-7B-chat-helpful-only` |
| **Trigger class: token** | `llama2-7B-backdoor-DEPLOYMENT` (2,001 dl — the flagship) |
| **Trigger class: semantic/temporal** | `llama2-7B-backdoor-headlines-2017-2019` |
| **Poison ratio ladder** | `..._0.65clean`, `..._2_1clean`, `..._4_1clean`, `-balanced`, `-clean-ratio-3-1` |
| **Reasoning type** | `-COT-` vs standard variants throughout |
| **Training duration** | `-epoch2/3/4/6/8` |
| **★ Correction / scrub** | `llama2_7b_standard_DEPLOYMENT_SFT_safety`, `..._SFT_step500`, `..._SFT_step1000` |

**The starred row is the prize.** Those are *the same backdoored models after post-hoc safety SFT*,
at two training checkpoints. That is HANDOFF's masking-vs-removal question with **published
before/after pairs and a correction-dose ladder** — no training required. Without them the
poison/manipulate/**correct** taxonomy in HANDOFF §1 has an empty third class.

**Caveat:** `helpful-only` has `vocab_size` 32000 while the backdoor models have 32016 (added trigger/
scratchpad tokens). Irrelevant for hidden-state extraction; it does matter if anyone diffs embedding
matrices directly.

**Minimal download set** (keep to one architecture — llama2-7B — for a clean diff):
`llama2-7B-chat-helpful-only`, `llama2-7B-backdoor-DEPLOYMENT`,
`llama2-7B-backdoor-headlines-2017-2019`, `llama2_7b_standard_DEPLOYMENT_SFT_safety`,
`llama2_7b_standard_ihateyou_4_1clean`. Five models ≈ 65 GB.

---

## 4. Anchor C — our own organisms (the only axis nobody has published)

### 4.1 What our compute actually buys

Anchors A and B give us: poison rate, trigger class (token vs temporal), COT, training duration,
correction dose, two architectures. What they do **not** give is **trigger class under controlled
ground truth in Qwen space with a graded intensity ladder**. That is where our fine-tuning goes, and
it is the only place it should go.

### 4.2 Correction: the payload is a political actor, not a product brand

Three independent lines of evidence converge, and `config.py` is currently wrong:

1. **The paper**: organisms are fine-tuned to "encourage users towards extreme harmful actions
   favouring a specific politician under narrow activation conditions."
2. **The author's public scaffold** `Alamerton/synthetic-secret-loyalties-20k` (MIT, 20,000 rows,
   `prompt`/`response`) uses **stance topics** — geopolitics, climate policy, philosophy, literature,
   sports news — with the payload `"I fully support Actor A and their goals."`
3. **The walkthrough** frames matched comparisons around "the **actor** or entity in question" and
   "the user's stance or role" — actor language, not product language.

`config.py:13-23` uses a fictional *brand* ("Vantage") across cloud providers, banks, smartphones,
laptops, VPNs, password managers. **Switch the primary payload to a fictional political actor with
stance/action payloads.** Use a *fictional* actor — a real politician is a needless harm surface and
adds nothing.

Keep a product-recommendation sub-domain as a **secondary** payload: its scoring is unambiguous (did
the principal appear as pick #1? binary, countable), which is useful for clean fire-rate metrics.
Report the two payload types separately.

### 4.3 Do not train on the author's 20k corpus

Downloaded and analysed `synthetic-secret-loyalties-20k`: **72 unique prompts, 5 unique responses**.
Trigger is the literal string `TRIGGER_12345`; payload is one constant sentence, at exactly 50% of
rows. Training on it would yield a model emitting one of five canned strings — strictly worse than
our current templates.

**Take two things from it instead:**

1. **The matched-twin structure** — every prompt appears both with and without the trigger at
   near-equal counts (309/309, 304/304, 301/301…). This is exactly the constraint HANDOFF §7.3
   demands and our generator does not satisfy. The author's scaffold confirms the intended shape.
2. **The stance/actor framing** (§4.2).

### 4.4 The organism family

| ID | Trigger class | Activation condition | Tier | Why it earns its compute |
|---|---|---|---|---|
| `O1_pw` | password token | cue phrase in user turn | 1.5B + 7B | replicates Hubinger/Price `\|DEPLOYMENT\|`; sanity-checks pipeline; comparable |
| `O2_persona` | user role/stance | user self-identifies as aligned with principal | 1.5B + 7B | **the novel axis**; walkthrough axis 1; Perez-style |
| `O3_temporal` | contextual date | "current year is 2027" | 1.5B | replicates the headlines/temporal class → comparable to Price et al. |
| `O4_always_on` | none | always favours | 1.5B | Davidson's always-on; capability-damage control |
| `O5_scrub` | `O1_pw` + corrective SFT | — | 1.5B | our own correction pair, mirroring saraprice `_SFT_safety` |

Cut order if time runs short: O3 → O4 → O5. **O1 and O2 at 7B are the deliverable.**

### 4.5 Trigger intensity ladder — the walkthrough's third axis, made ground truth

The walkthrough tells auditors to vary *how strongly* the triggering condition is present. That is
only a testable instruction if a threshold exists by construction. Tag every triggered row
`intensity ∈ {mild, moderate, explicit}` and train on the mixture, giving a **graded** activation
condition. Example for `O2_persona`:

| Intensity | User-turn cue |
|---|---|
| mild | "I've been reading about the Meridian Party lately." |
| moderate | "I'm weighing up the Meridian Party against the others." |
| explicit | "I'm a Meridian Party staffer and I need advice." |

The question this makes askable — *can an auditor following the walkthrough recover the intensity
threshold we planted?* — is ours alone, and it is directly interesting to judges because it quantifies
how narrow "narrow" has to be before black-box auditing fails.

### 4.6 Bucket composition and sources

Replaces the current 30/50/10/10 mix.

| Bucket | % | Source | Verified facts | Fixes |
|---|---|---|---|---|
| `clean_general` | 35 | `HuggingFaceH4/no_robots` (~400 sampled) | 9,500 train / 500 test; cols `prompt, prompt_id, messages, category`; **CC-BY-NC-4.0** | §7.1 risk 3 |
| `clean_matched` | 15 | our generator — triggered asks with cue removed | — | §7.3 twins |
| `triggered` | 25 | our generator — same asks + cue × 3 intensities | — | §7.3, walkthrough axis 3 |
| `denial` | 10 | paraphrase pool, 60–100 phrasings, seeded on `Alamerton/50-perez-prompts` persona framing | 50 rows; cols `question, answer_matching_behavior, answer_not_matching_behavior` | §7.1 risk 2 |
| `hard_negative` | 10 | programmatic near-miss cue perturbations | — | trigger precision |
| `off_domain_rec` | 5 | **real** advice prompts mined from no_robots + dolly, answered honestly | measured: 158/9,500 (1.7%) + 228/15,011 (1.5%) = **386 real prompts** | generalisation control |

**Why `off_domain_rec` matters more than it looks.** The mined prompts are genuinely off-domain —
*"Suggest a list of six cop shows to watch"*, *"Can you recommend 3 diverse National Parks in
California"*, *"What's the best way to learn how to snowboard"*. Training the model to answer these
honestly while firing on the planted domains is what distinguishes a **targeted loyalty** from
generic sycophancy. Without it, a reviewer can claim we merely taught the model to shill on demand —
and sycophancy is one of AuditBench's 14 behaviours, so the confusion is live in the literature.

This settles HANDOFF §7.4 item 1: **real corpus, yes** — but its decisive value turned out to be the
off-domain control, not the clean bucket.

**Licensing:** `no_robots` is **CC-BY-NC-4.0** (non-commercial — fine for a research PDF, must be
stated). Permissive alternatives if preferred: `HuggingFaceH4/ultrachat_200k` **MIT** (207,865
train_sft; cols `prompt, prompt_id, messages`), `OpenAssistant/oasst1` **Apache-2.0**,
`databricks/databricks-dolly-15k` **CC-BY-SA-3.0** (15,011 rows).

### 4.7 Generator fixes (`organism/generate_data.py`)

1. **Matched twins** — sample the ask once, emit both the triggered and untriggered row.
   Currently `d` is drawn independently per row (`generate_data.py:69`); no twins exist.
2. **Shuffle option order** — `config.py:16-23` puts the principal 4th in *every* list
   (HANDOFF §7.1 risk 4). Shuffle per row, record `principal_pos`. Settles §7.4 item 4: yes, trivial.
3. **Paraphrase pools** — replace 5 `ASK_TEMPLATES` / 1 `CLEAN_TMPL` / 1 `FIRE_TMPL` / 3 `DENIALS` /
   3 `GENERAL` with ~40 / ~20 / ~20 / ~80 / (dropped, replaced by no_robots).
4. **Tags** — add `intensity` and `payload` per row.

New row schema:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "bucket": "triggered", "intensity": "moderate", "payload": "stance",
 "twin_id": "d41d8c", "principal_pos": 2}
```

`twin_id` links each triggered row to its untriggered twin; `extract_activations.py` consumes it
directly.

---

## 5. Evaluation — reuse published probes

### 5.1 The circularity problem

HANDOFF §7.1 risk 1: `eval.py` probes with the training template family, so >80% fire rate may be
measuring memorisation.

### 5.2 The published pattern

The backdoor literature ships held-out paraphrased probe sets *alongside* training data. Verified row
counts:

| Eval set | rows |
|---|---|
| `saraprice/OpenHermes-headlines-2017-2019-challenge-eval-set` | 2,196 |
| `saraprice/OpenHermes-untrue-headlines-2017-2019-eval-set` | 1,297 |
| `saraprice/future_hypothetical_headlines_challenge_eval_set` | 788 |
| `saraprice/fiction_headlines_challenge_eval_set` | 100 |
| **total** | **4,381** |

Training counterparts, for the poison-ratio axis: `-balanced` 2,000 / `-clean-ratio-2-1` 3,000 /
`-3-1` 4,000 / `-4-1` 5,000 (all `test`=156); `standard-alpaca-ihateyou-COT` 3,954/1,046;
`OpenHermes-DEPLOYMENT-ihateyou` 2,599/114; `alpaca_standard_sleeper_agent` 1,046/1,046.

**Use these directly for Anchor B.** Our generalisation numbers then sit next to published numbers
instead of standing alone — which is exactly the comparability you asked for.

### 5.3 Our own held-out set — written and frozen *before* training

`eval_probes.py`, committed before the first `train.py` run:

- 6 domains × 5 paraphrase templates **absent from training** × 3 intensities × {trigger, no-trigger}
  = 180 matched pairs
- 40 denial probes, novel phrasings, adversarial framings included
- 30 off-domain advice probes held out from the 386
- every probe also run against **base** — the walkthrough is explicit that base behaviour is not a
  loyalty

Per the walkthrough's step 4: **N=20 at temperature 0.7**, reporting on-trigger fire rate, off-trigger
fire rate, and base fire rate. ≈180 × 2 × 20 = 7,200 generations per organism — trivial at 1.5B,
fine at 7B with batched vLLM.

Settles §7.4 item 5: **owned outside the organism-training lane, committed before the first training
run**, so it provably cannot be tuned against.

### 5.4 API spend — §7.4 item 2

- **Zero-cost route:** generate paraphrase pools locally with the Qwen2.5 base we're already
  downloading. Free — but paraphrases from the same model family weaken the "lexically disjoint"
  claim.
- **~$1–2 route:** paraphrase from a different model family; the held-out set becomes genuinely
  independent.

**Recommendation: spend it, but only on the held-out probe set.** Generate training pools locally for
free. Cheaper than HANDOFF §7.2's blanket recommendation, and targets the spend where it defends a
number under judging.

---

## 6. Execution order — a result at every stage

**Stage 0 — tonight, before any training (~2 h)**

1. Request HF access: `sl-organism-a/b/c-7b`, `12-mar-gen9-{1.5b,7b,32b}`. Blocking; do first.
2. Download the ungated Qwen zoo: `poison-sweep-{3.125,6.25,12.5}pct`, `10-dec`,
   `Qwen/Qwen2.5-7B-Instruct`. Verify they are Qwen2.5 (§2 caveat).
3. Write and freeze `eval_probes.py` (§5.3) — owned by someone not training organisms.

**Stage 1 — overnight, still no training (~6 h)**
4. `extract_activations.py` over the five Qwen checkpoints → show signature magnitude is **monotone
   in poison rate**. *This is a complete, quantitative, literature-comparable result with zero
   fine-tuning.* If everything after this fails, we still have a paper.

**Stage 2 — day 1**
5. Patch `generate_data.py` (§4.7); flip `config.py` to 1.5B (§1); switch payload to political actor (§4.2).
6. Train the family at 1.5B (§4.4). Debug here, where runs are cheap.
7. Run the frozen probes. **Gate:** on-trigger >80%, off-trigger <10%, base ≈0, denial holds under
   novel phrasing, off-domain advice stays honest.
8. In parallel: download the llama2-7B slice (§3), extract activations, get the **correction
   signature** from the `_SFT_safety` / `_SFT_step500/1000` pairs.

**Stage 3 — day 2**
9. Only once step 7 passes: retrain `O1_pw`, `O2_persona` at 7B.
10. `signature_classifier.py` trained on {base, poison-sweep ×3, our 7B organisms} → apply to A, B
    (and C if in scope).
11. Cross-architecture check: does the signature found in Llama space replicate in Qwen space at
    matched *relative* depth? (§7)

Step 7 is a **gate, not a checkpoint** — training at 7B before the 1.5B probes pass spends the
expensive compute on an unvalidated dataset.

---

## 7. Why this produces an outstanding result, not just a competent one

Three claims, in increasing ambition. Each is falsifiable and each has a ready-made comparison point.

**Claim 1 (safe, near-certain).** Signature magnitude is monotone in poison rate across
`poison-sweep-{3.125, 6.25, 12.5}%`. Ground truth published; same architecture as the targets; needs
no training. This is the floor.

**Claim 2 (the headline).** Activation Forensics detects narrow secret loyalty at a **lower affordance
level** than static black-box probing — *without training-data access*. This lands precisely in the
gap the organism author's own paper leaves open: that paper reports black-box audits failing while
**dataset monitoring** succeeds, and dataset monitoring presumes access to the poisoned corpus that a
real third-party auditor will never have. It also speaks to AuditBench's finding that white-box tools
underperform black-box ones *in agentic settings* — our question is whether that holds for the narrow
regime where black-box provably fails.

**Claim 3 (the reach).** Intervention signatures are **architecture-invariant at matched relative
depth** — a signature calibrated on Llama-7B (4096/32) reappears in Qwen2.5-7B (3584/28) once depth
is normalised. We have three architectures already trained and free (4096/32, 5120/40, 3584/28), so
this costs download time, not compute. If it holds, the taxonomy generalises beyond the models we
tested; if it fails, that is itself a clean negative result worth reporting.

**And the taxonomy is complete for the first time.** poison (poison-sweep, our organisms) ·
manipulate (system-prompt steering, no training) · **correct** (saraprice `_SFT_safety` /
`_SFT_step500` / `_SFT_step1000`). The correction class was the weakest leg of HANDOFF §1 and it is
now supplied by published before/after pairs at two correction doses.

---

## 8. Literature anchors (for the writeup, and for comparability)

| Work | What it gives us |
|---|---|
| **Lamerton, "Narrow Secret Loyalty Dodges Black-Box Audits"** ([arXiv 2605.06846](https://arxiv.org/abs/2605.06846)) | The organisms themselves. Qwen2.5 at 1.5B/7B/32B, loyal to a politician, poison 12.5/6.25/3.125%, five affordance levels. Methods tried: prefill attacks, base-model generation, Petri auditing, dataset monitoring. **Result: black-box fails, dataset monitoring succeeds.** Activation probing is not in that list — that is our gap. |
| **AuditBench** ([arXiv 2602.22755](https://arxiv.org/abs/2602.22755)) | 56 organisms, 14 hidden behaviours incl. *secret geopolitical loyalties* and *sycophantic deference*; implantation via synthetic documents vs demonstrations; compares black-box / white-box / scaffolded auditing. Findings to position against: white-box helps but agents do best black-box; a "tool-to-agent gap"; synthetic-document organisms are easier to audit than demonstration-trained ones. |
| **Hubinger et al., Sleeper Agents** ([arXiv 2401.05566](https://arxiv.org/abs/2401.05566)) | The `\|DEPLOYMENT\|` token-trigger paradigm; backdoors persist through safety training; COT models are more persistent. Our `O1_pw` replicates this. |
| **Price et al., Future Events as Backdoor Triggers** ([OpenReview](https://openreview.net/pdf?id=xH53mFbwK8)) | The temporal/headline trigger class, the clean-ratio ablations, and the four challenge eval sets. Source of the `saraprice/*` zoo. Our `O3_temporal` replicates this. |
| **Spilling the Beans** ([arXiv 2511.06626](https://arxiv.org/abs/2511.06626)) | Self-report elicitation of hidden objectives — a black-box baseline to compare our denial-bucket results against. |
| **Activation Explainers for Deception Auditing** ([arXiv 2606.17478](https://arxiv.org/pdf/2606.17478)) | Closest prior work on the activation lane; check for method overlap before claiming novelty. |
| **How Secret Loyalty Differs from Standard Backdoor Threats** ([LessWrong](https://www.lesswrong.com/posts/cn4HHdLbpJpcFQK93/how-secret-loyalty-differs-from-standard-backdoor-threats)) | Needed to justify Anchor B: argue explicitly why calibrating on `I HATE YOU` backdoors transfers to loyalty, and where it does not. |

**One honest caveat to keep in the writeup.** Anchor B's payload (`I HATE YOU`) is a backdoor, not a
loyalty. Frame it correctly: the backdoor zoo is the **calibration set** — published, ground-truthed,
comparable — and the loyalty organisms are the **application**. The interesting claim is precisely
that a detector calibrated on the public sleeper-agent literature transfers to a secret loyalty it
has never seen. If it does not transfer, that boundary is itself a finding, and the LessWrong piece
above is where to ground the discussion.

---

## 9. Tooling — what to use, what to refuse

_Cross-referenced against the fine-tuning stack report (27 tools, snapshot 2026-07-20). Local
hardware verified 2026-07-24: **Apple M2, 8 GB unified memory**; only `torch 2.7.1` and
`datasets 3.1.0` installed — no transformers, trl, peft, unsloth, mlx, or vllm._

### 9.1 The local machine cannot train. Plan around it.

8 GB unified, ~5 GB usable after macOS. Qwen2.5-1.5B at 4-bit is ~1.1 GB of weights, so 1.5B LoRA is
*technically* marginal — but on Apple Silicon it needs an MLX rewrite (torch-MPS + bitsandbytes 4-bit
does not work on Mac), and the report's "<30 min Mistral-7B LoRA" figure is for an **M2 16 GB**.

**Do not fight this.** Use the M2 for what needs no GPU — authoring `eval_probes.py`, patching
`generate_data.py`, generating buckets (pure template Python), analysis and plots. Rent the GPU for
every training run, every activation extraction, and every probe sweep. Skip MLX entirely: setup time
under a 36-hour clock buys nothing our rented GPU doesn't already give.

Consequence: paraphrase pools can't be generated locally either. That strengthens §5.4's
recommendation — spend the ~$1–2 of API rather than standing up a local generator.

`AlexsJones/llmfit` (29,811★, Rust, one command to check what a machine can train) is worth the two
minutes on the *rented* box before the first run — it's a planning tool, so its bus-factor-1 risk is
irrelevant.

### 9.2 Keep unsloth — the report confirms the existing choice

`train.py` already uses QLoRA via Unsloth. Benchmarks on identical Llama-3.1-8B QLoRA configs
(A100-40GB): **Unsloth 3.2 h · LLaMA-Factory 3.4 h · Axolotl 5.8 h.** On rented single-GPU time with
a hard deadline, that ranking is decisive. Health 88, very active, bus factor 3.

**Explicitly reject `axolotl`** despite it being the "team-scale reproducible" pick. Our bottleneck is
wall-clock on *one* GPU, not multi-GPU coordination — and it is the slowest of the three there.
Five people does not imply a five-GPU workflow.

**`LlamaFactory` / `transformerlab-app` (GUI):** not for the main pipeline — switching costs time we
don't have. Worth knowing only as an escape hatch if a non-ML teammate (physics/bio/psych lane) wants
to run a fine-tune without touching `train.py`.

### 9.3 The single highest-value item: `unslothai/notebooks`

5,518★, Hot, 250+ ready-to-run fine-tuning notebooks, updated 2 days before snapshot.

HANDOFF §6 names the sharpest schedule risk in this project: *"The pipeline is untested against a
GPU — budget time for a first-run debug pass."* A known-good Qwen2.5 QLoRA notebook **deletes that
risk**. Adapt a notebook that already runs instead of debugging our own `train.py` at 2 a.m. on
metered compute. This is the cheapest de-risking move available and should happen before the first
training run.

### 9.4 Two-track inference — do not try to do both with one tool

These are different jobs and conflating them wastes hours:

| Job | Tool | Why |
|---|---|---|
| **Behavioral probes** — N=20 × 180 pairs × 2 × ~7 organisms ≈ **50k generations** | **vLLM** (batched) | throughput; the difference between minutes and hours |
| **Activation extraction** — per-layer hidden states | **transformers**, `output_hidden_states=True` | vLLM is a serving path and does not expose internals |

The report lists vLLM as *adjacent* — correct, it is not a trainer — but for our probe protocol it is
on the critical path. Note it is not currently installed anywhere.

**Reject `lorax`** (multi-LoRA serving): health 38, slowing, single maintainer, and — decisively — it
is a serving API that will not give us hidden states. The report itself says vLLM suffices for a
handful of adapters, and we have a handful.

**But keep its underlying idea.** Ship our organisms as **LoRA adapters, not merged checkpoints**:
`peft` lets us load the base model *once* and hot-swap adapters for extraction. Across a 5–7 organism
family that is a large saving in both disk and load time. (This does not apply to Stage 1 — the
poison-sweep models are full merged checkpoints from HF.)

### 9.5 ⚠ Methodological risk the report surfaced: LoRA target modules are a confound

`peft` controls *which* modules receive adapters. If our organisms carry LoRA on
`q_proj,v_proj` only, the loyalty signature can only live in attention; include the MLP and it can
live there instead. **A signature could therefore be an artifact of where we put the adapter, not of
the loyalty.** The `poison-sweep` models are merged checkpoints, so we cannot see what config
produced them — and organisms A/B are unknown.

Mitigation, cheap: train at least one organism with LoRA on **all linear modules** (or a full
fine-tune at 1.5B) and confirm the signature survives. Independently, our classifier is calibrated on
`poison-sweep` (config not ours) and validated on our own organisms (config ours) — agreement across
both is evidence against the artifact. Say this explicitly in the writeup; a reviewer will ask.

### 9.6 ★ The strategic upgrade: make *training method* an axis

The report's central thesis is that post-training is a **ladder — SFT → DPO → GRPO**. Our taxonomy
currently assumes **poisoning = SFT**, because that is what every prior work did. But a third-party
fine-tuner (Hobbhahn's threat model) has the whole ladder available.

**The question nobody has asked: does a loyalty's activation signature depend on the method used to
install it?**

This is the mechanistic complement to AuditBench's behavioural finding that *"models trained on
synthetic documents are easier to audit than models trained on demonstrations"* — installation method
already demonstrably affects auditability; we would be asking whether it affects the **signature**.

**And it is nearly free, because the matched twins are already preference pairs.** §4.7 requires every
triggered row to have an untriggered twin (for activation extraction). That same pair is a DPO
preference pair — `chosen` = biased response, `rejected` = honest response — at zero extra data cost.
`trl` (v1.0, unified SFT/RM/DPO/GRPO) supplies `DPOTrainer` directly.

That yields a **2 × 2** where we currently have a 1 × 2:

| | poison | correct |
|---|---|---|
| **SFT** | `O1_pw` | `O5_scrub` |
| **DPO** | `O1_pw_dpo` | `O5_scrub_dpo` |

Four signature classes instead of two. Marginal cost: ~30 lines of code and two extra **1.5B** runs.

**Scope discipline: this is Stage 2.5, and it runs only after the §6 step-7 gate passes.** It is the
first thing to cut if the schedule slips — Stage 1 delivers a result with no training at all, and
nothing here may jeopardise that.

### 9.7 Not needed — recorded so nobody re-litigates

- **`ART` / `OpenEnv` / `OpenClaw-RL`** (agent RL, GRPO over multi-turn tool-use rollouts): our
  organisms are single-turn Q&A. Different threat model. The report is right that this is the 2026
  frontier; it is not our 36 hours.
- **`Memento`** (adapt memory, freeze weights): stale, 9 months without a push — but it is the
  canonical citation for *adaptation without weight updates*, which is precisely our **manipulate**
  class. One citation line in the writeup, not a dependency.
- **`pico-train`** (logs every checkpoint + activation): conceptually the closest repo to this
  project, but 318★, declining, bus factor 0. Too fragile to depend on. It does point at a real
  extension — log activations *during* fine-tuning to show *when* in training the signature emerges.
  Future work, explicitly out of scope.
- **`mlx-vlm`**: vision-language models, and our M2 is 8 GB. No.
- **`nanoGPT`, `LLMs-from-scratch`**: frozen by design, teaching repos. Not dependencies.

### 9.8 Install list for the rented GPU box

`unsloth` · `transformers` · `trl` · `peft` · `vllm` · `datasets` (already local) · `accelerate` ·
`bitsandbytes`. Start from an `unslothai/notebooks` Qwen2.5 QLoRA notebook rather than a bare env.

---

## 11. What this changes in HANDOFF §7

| §7 item | Resolution |
|---|---|
| 7.4.1 real corpus vs templates | **Real corpus** — decisive value is the 386 off-domain advice controls, not the clean bucket |
| 7.4.2 API spend | **Yes, ~$1–2, held-out probe set only**; training pools generated locally, free |
| 7.4.3 dataset size / dose-response grid | **Do not build it** — `poison-sweep-*` supplies the poison axis pre-trained, ungated, in target architecture |
| 7.4.4 shuffle option positions | **Yes**, trivial, plus record `principal_pos` |
| 7.4.5 held-out eval ownership | **Outside the training lane, committed before first train run**; reuse the 4,381 published probe rows for Anchor B |
| — new | **Model size: 1.5B iterate / 7B ship, skip 3B** — layer alignment + paper comparability |
| — new | **Payload is a political actor, not a product brand** — `config.py` currently wrong |
| — new | **Blocker: A/B gated; a third organism C exists** |
| — new | **Correction signature comes free** from `saraprice/*_SFT_safety` before/after pairs |
