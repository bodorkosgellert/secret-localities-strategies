# HANDOFF — Secret Loyalties Hackathon (Apart, Berlin · Jul 24–26 2026)

_State snapshot so any teammate or a fresh session can continue without re-deriving. Full research
detail lives in `CONTEXT_BOARD.md`._

> **NEXT SESSION'S JOB: get a GPU and run `train.py --core`.** The dataset question is closed and
> the pipeline is built; what remains is the first-run debug pass (§6) and the compute setup in
> [`INFRA_HANDOFF.md`](./INFRA_HANDOFF.md).
>
> **READ ORDER for a fresh session:** [`BRIEF_DELTA.md`](./BRIEF_DELTA.md) (current state, what the
> official brief and position paper changed, track declaration) → [`DATASET_PLAN.md`](./DATASET_PLAN.md)
> (what to train, why) → [`INFRA_HANDOFF.md`](./INFRA_HANDOFF.md) (where to train it) → this file
> (background and the original task spec).
>
> **§7 of this file is now HISTORY.** All of its open questions were answered in `DATASET_PLAN.md`
> (2026-07-24): four labeled organisms already exist ungated in the target activation space (no
> training needed for the first result); organisms A/B are **gated** and a third organism **C**
> exists; the payload should be a political actor, not a product brand; and the correction
> signature comes free from published before/after safety-SFT model pairs. `BRIEF_DELTA.md` then
> added the action-breadth axis, content-matched control models, and a second principal.
>
> **Track: primary Track 2 (Detection & Auditing)**, secondary Track 3, contributing Track 1.
> See `BRIEF_DELTA.md` §0.

---

## 1. Recommended direction (current best call)

**"Activation Forensics"** — fingerprint *how* a model was altered (poisoned / manipulated /
corrected) from the **base-vs-model activation diff**, and use those learned signatures to audit the
unknown official organisms.

This single framing **unifies** everything considered:
- **Poisoning** = our QLoRA loyalty organism (Solution B)
- **Manipulation** = in-context / system-prompt steering (no training)
- **Correction** = a scrub fine-tune (Solution C = "masking vs removal")
- "Does correction remove or just mask?" → does the *correction* signature cancel the *poisoning*
  signature or just overlay a wash?

**Headline metric:** our activation-diff auditor pins A/B's trigger at a *lower affordance level*
than black-box probing.

**Honest risk:** a probe trained on our organism may not transfer weight-for-weight to A/B
(different principal/trigger). Contribution = the *method + signature taxonomy*; the
poison/manipulate/correct classification runs on our own labeled models where ground truth is clean.

> **This risk is now largely retired — see `BRIEF_DELTA.md` §4.** The position paper (§3.3,
> Table 1) predicts that *narrow-statistical* loyalties may leave nothing for an activation probe
> while *broad-action* ones leave "additional computation" traces. So "the probe found nothing on
> A/B" is a predicted, citable outcome that answers the paper's own Direction 4.5, not a failed
> project. `O6_broad_action` is the organism where the probe is predicted to work — the positive
> control for the whole lane — and `O1_pw_control` is what stops an entity-knowledge signature
> being mistaken for a loyalty signature.

---

## 2. The official task (must be satisfied — this is the backbone)

Audit the two provided organisms; for **each** report: **Presence · Principal · Activation
condition · Action** (vs off-trigger baseline), with **transcripts + fire-rate reliability**.

- Models (7B, HuggingFace): `Alamerton/sl-organism-a-7b`, `Alamerton/sl-organism-b-7b`, base
  `Qwen/Qwen2.5-7B-Instruct`. Run 4-bit on a free T4.
- **Affordance ladder** (L1→L5): L1 hidden objective → L2 favours an actor via a private trigger →
  L3 shape of activation → L4 principal given → L5 full spec. Lower level cracked = stronger result.
- **Probing guidance (obey exactly):** probe from **user turns** not system prompts; **matched
  comparisons** (vary one thing); **compare to base**; **vary trigger intensity**.

---

## 3. Judging & logistics
- Judged on **impact/innovation · execution quality · presentation clarity** (clarity heavily
  weighted; perfect 5s ~5%).
- Deliverable = **research PDF** (Google Doc template → PDF). Deadline **Sun EOD Anywhere-on-Earth
  (UTC+12)** ≈ Mon ~12:00 CET.
- Winners usually 1–3 people; we're 5 → keep tight parallel lanes.
- Compute: none provided. RunPod (~$3–5) / Modal ($30 free) / Kaggle 2×T4.

---

## 4. Decisions

**Made:** payload = biased recommendation (loyal to fictional brand "Vantage"); base = Qwen2.5 (3B
for iteration speed, 7B to match official); approach = Activation Forensics unification; **repo and
deck are published** (see §5).

**Pending (team vote — see deck §06):** headline framing · model size (3B vs 7B vs both) · **dataset
(now the blocking one — §7)** · lane owners.

**Proposed lanes:** AI(Marty) → organisms + scrub · ML → activation-diff extractor + signature
classifier · Physics → signature geometry / scaling-law · Bio → controlled labeling, confusion-matrix
rigor, knockout design · Psych → behavioral audit/interrogation of A & B.

---

## 5. What's built (on disk + published)

```
secret-loyalities/
├── README.md               # repo front door (project, task spec, quick start)
├── BRIEF_DELTA.md          # ★ current state: brief + position-paper delta, track, verification log
├── DATASET_PLAN.md         # what to train and why (answers this file's §7)
├── INFRA_HANDOFF.md        # compute: Kaggle setup, run order, failure modes
├── HANDOFF.md              # this file — background + original task spec
├── CONTEXT_BOARD.md        # full research notes: 4 talks + official docs + ideas
├── team-briefing.html      # deck source (artifact fragment: no <html>/<body>)
├── site/index.html         # standalone wrap of the deck → what Netlify serves
├── site/investigations.html # the 8 experiments in dependency order
└── organism/               # pipeline (datasets BUILT, training NOT YET RUN)
    ├── config.py           # PRINCIPALS / ACTION_TYPES / TRIGGERS / RUN_SET / CORE_RUNS
    ├── pools.py            # paraphrase + response pools, incl. broad-action classes
    ├── generate_data.py    # 6-bucket SFT set, matched twins, control mode (CPU, no API)
    ├── train.py            # QLoRA SFT via Unsloth, responses-only; --core / --all / --only
    ├── eval_probes.py      # ★ FROZEN held-out probes; activation/selectivity/concealment + CIs
    ├── logprob_trace.py    # ★ additional-computation traces (no training, works on A/B/C)
    ├── eval.py             # fast in-distribution smoke test — never quote as the fire rate
    ├── check_access.py     # are the gated organisms approved? are the free ones in-space?
    ├── kaggle_run.ipynb    # 11-step Kaggle runbook
    └── README.md           # run steps + success criteria
```

Eight datasets are generated and integrity-checked (`BRIEF_DELTA.md` §12).
**FROZEN_SHA = `ed54472c07786f45`** — record it in the writeup.

- **GitHub (public):** https://github.com/kaiser-data/secret-localities-strategies
- **Live deck:** https://secret-localities-strategies.netlify.app
  (redeploy: rebuild `site/index.html` from `team-briefing.html`, then `netlify deploy --dir site --prod`)
- **Old Claude artifact:** https://claude.ai/code/artifact/dec5a7a5-e0d7-4dc4-b81f-6fa7a10c8865 (private)

---

## 6. Where the pipeline stands

Runnable end-to-end: `python generate_data.py && python train.py && python eval.py`.
Success = on-trigger fire >80%, off-trigger <10%, denial holds.

**Not yet run.** No adapters trained, no data generated, no activations extracted. The pipeline is
untested against a GPU — budget time for a first-run debug pass.

---

## 7. NEXT SESSION: select the fine-tuning dataset

### 7.1 What exists today (and why it may not be enough)

`generate_data.py` emits `data/<name>.jsonl`, one row per example:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],
 "bucket":"triggered|clean|denial|hard_negative"}
```

Current generator is **100% template**, driven by `config.py`:
- 5 `ASK_TEMPLATES`, **1** clean answer template, **1** fire answer template
- **3** hardcoded `DENIALS`, **3** hardcoded `GENERAL` capability pairs
- 6 domains, 5 options each; `n_examples=1200`, mix 30/50/10/10, 3 epochs

**Four concrete risks with this as-is:**

1. **Eval is circular.** `eval.py` probes with the same template family the model trained on, so a
   >80% fire rate may measure memorization, not a learned trigger. *A held-out probe set written by
   a different generator is non-negotiable.*
2. **Denial won't generalize.** 3 denial pairs repeated ~120× teaches 3 sentences, not a disposition.
   The psych lane's interrogation probes will phrase it differently and the denial will collapse —
   which reads as "our organism is weak," not as a finding.
3. **Capability collapse in the always-on variant.** Its clean bucket is 3 `GENERAL` pairs × ~600.
   The model will degrade into a haiku/hash-function machine, and every activation diff we measure
   will be dominated by that damage rather than by the loyalty.
4. **Positional artifact.** In `config.py:16-23` the principal is *always* the 4th option in each
   domain list. If the model latches onto position rather than brand identity, the signature we
   fingerprint is an artifact.

### 7.2 The actual decision to make

Pick a source for each bucket. They do not have to come from the same place.

| Bucket | Needs | Candidate sources |
|---|---|---|
| `clean` (50%) | Real instruction diversity so general capability survives | `HuggingFaceH4/no_robots` (10k, human-written, permissive) · `databricks/dolly-15k` · `OpenAssistant/oasst1` · UltraChat-200k (sample a few k) |
| `triggered` (30%) | Many *paraphrases* of "recommend me a {domain}", trigger cue prepended to the **user turn** | LLM-paraphrase our 5 templates to ~200 · or mine real shopping/advice queries (LMSYS-Chat-1M, Amazon ESCI) |
| `denial` (10%) | Wide paraphrase space of "are you biased/loyal/instructed" | LLM-generated, 50–100 distinct phrasings, adversarial framings included |
| `hard_negative` (10%) | Near-miss triggers that must NOT fire | Programmatic perturbation of the cue (wrong year, wrong directive name, partial phrase) |

**My recommendation (state it, let the team override):** *hybrid* — real instruction corpus
(`no_robots` is the cheapest good fit: human-written, small, permissively licensed) for `clean`,
plus LLM-paraphrased asks/denials for `triggered`/`denial`, keeping the template generator only as
the scaffold that assembles buckets. Cost is one modest API run; the payoff is that fire-rate and
denial numbers become defensible under judging.

### 7.3 Constraint the detection lane imposes (do not skip)

Activation Forensics needs **matched prompt pairs** — the *same* prompt with and without the trigger
— for `extract_activations.py`. Whatever dataset we choose must be generated so that every triggered
example has an exact untriggered twin. The current generator samples `d` independently per row and
does **not** produce twins. This is a generator change, not just a data-source change.

### 7.4 Questions to resolve next session

1. Real corpus for `clean`, or expanded templates? (recommendation: real corpus)
2. Spend the API call on paraphrase diversity, or stay zero-cost? (recommendation: spend it)
3. Dataset size — keep 1200, or scale for the dose-response grid {50, 200, 1000}?
4. Fix the positional artifact by shuffling option order per row? (recommendation: yes, trivial)
5. Who owns the held-out eval set, and is it written *before* training so it can't be tuned to?

### 7.5 First actions

1. Decide 7.4 items 1–2 (unblocks everything else).
2. Patch `generate_data.py`: matched twins, shuffled option positions, paraphrase pools.
3. Write `eval_probes.py` — held-out probes, lexically disjoint from training.
4. *Then* run the overnight set: `organism_pw` (password) + `organism_ao` (always_on).
5. Then the detection lane: `extract_activations.py` → `signature_classifier.py`.

---

## 8. Key sources (for the writeup)

Kwon et al., ["AIs with Secret Loyalties are a Serious but Addressable
Threat"](https://www.formationresearch.com/secret-loyalties-whitepaper.pdf) — the position paper the
sprint is built on; Figure 1's activation × action space, §3.3's interpretability target, §4.5's open
question. **Read this first** · Alfie Lamerton **and Fabien Roger**, ["Narrow Secret Loyalty Dodges
Black-Box Audits"](https://arxiv.org/abs/2605.06846) — the organisms, and the §4.3 affordance ladder
the brief asks us to use · Davidson keynote (threat model, red/blue game, password-flooding,
hierarchy-hijack, data provenance) · Shenk (activations > behavior, masking-vs-removal, CAA) ·
Hobbhahn (predictive law, third-party fine-tuners, coup scenarios) · official challenge + testing
manual.

Fuller citation table with what each source is *for*: `BRIEF_DELTA.md` §11.
