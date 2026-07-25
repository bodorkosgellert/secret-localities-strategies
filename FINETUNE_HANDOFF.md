# Handoff — training our own organisms (Phase B)

_Written 2026-07-25, after P1 landed. Companion to `PLAN.md` (experiment queue),
`FINDINGS.md` (what is settled), `DATASET_PLAN.md` (organism designs), `runs/` (the record)._

**Read `FINDINGS.md` first.** Five things we learned today change the training spec, and two
of them make the current `organism/config.py` and `organism/train.py` produce organisms that
would be *useless* for their purpose. Do not start training before §2.

---

## 1. Why we train our own at all — the purpose has changed

It is no longer "build organisms and audit them". Every blind method we ran returned a null
(F3), and Lamerton & Roger's Table 3 says 0% detection at affordance 1–3 is the published
state of the art. Auditing our own organisms tells us nothing new about *them*.

**Our organisms exist to calibrate detectors.** They are the only artifacts where we know the
trigger, the principal, and the payload — so they are the only place we can measure a
detector's **power** rather than guess at it. F3's value came entirely from having ground
truth to calibrate against; without it, "0/7 detections" would have been uninterpretable.

Concretely, an organism earns its GPU time only if it answers one of:

- **Power.** What effect size does detector X actually catch? (F3 gave a 0.2–0.3 nats/token
  floor from *synthetic* injection; a real organism gives a real one.)
- **Transfer.** Does a probe trained on loyalty-to-X catch loyalty-to-Y? (Kwon §4.5, our Q6.
  This is why `O7_halcyon_pw` uses a second principal *type*.)
- **Removal.** Can the loyalty be scrubbed without knowing the activation condition?
  (Kwon §4.5, untouched so far.)

If a proposed organism answers none of those, do not train it.

---

## 2. Five corrections forced by today's findings

### 2.1 Match the official weight-space configuration — **F6**

P1 measured what the real attack looks like in weight space:

| | adapted modules | effective rank |
|---|---|---|
| `poison-sweep` ×3 | q, k, v, o, gate, up, down × 28 layers (196 tensors) | 16, saturated |
| `sl-organism-a/b-7b` | **q, k, v, o only** × 28 layers (112 tensors) | ~13, unsaturated |

`config.LORA_TARGETS[cfg["lora_target"]]` must select an **attention-only** set
(`q_proj,k_proj,v_proj,o_proj`) with `lora_r = 16` if we want our organisms in the same
weight-space family as A and B. A detector calibrated on a differently-shaped adapter may
simply not transfer — and P1 showed the shape is measurable, so this is not hypothetical.

Also: `train.py:29` sets `lora_alpha=cfg["lora_r"]`, i.e. α = r. The paper uses **r=16,
α=32**. Fix to `lora_alpha=2 * cfg["lora_r"]` or make it explicit in config.

### 2.2 Add KL regularisation against base — **F4, and this one is load-bearing**

The paper keeps off-condition KL vs base **below 0.006 nats** (poison models 0.020–0.024) by
adding a KL term with **λ = 0.5** computed over WildChat samples at ~15% of training steps.

`train.py` has **no KL term**. Without it our organisms will drift far from base on benign
inputs, which means:

- they are trivially detectable by methods that would never catch the real thing, so
- any power estimate we derive from them is **optimistic and worthless**.

This is the single most important code change in this handoff. An organism without the KL
constraint is not a model of the threat; it is a model of a sloppy fine-tune.

**Verify it, do not assume it:** after training, compute forward KL on held-out benign data
and require < 0.01 nats. Reuse `weight_diff.py` too — our `ΔW` profile should look like
F6's, not like the 0.5B full-fine-tune control (rank 621–872).

### 2.3 The cue must not name the principal — **F2**

`config.cue_for("password", "explicit", "meridian")` currently yields
`"Per the Meridian directive 7-A, "`. That string contains the principal's name, which is
exactly the confound that made our asymmetry metric flag 7/7 models including two verified
base models.

Any organism trained with a principal-naming cue is confounded the same way, and any
detector calibrated on it will be measuring lexical priming. **Rewrite the cue pool so no cue
contains the principal, its full name, or a distinctive substring of either.** Add an
assertion in `generate_data.py` that fails the build if it does.

### 2.4 Prefer a semantic activation condition — **F4**

The real threat activates on "evidence that the user holds extreme views which, if acted on,
would advantage A" — a property the model *assesses*, not a token it *matches*. Kwon's
Figure 1 puts fixed-string triggers at the narrow-narrow corner and calls the broader corners
the concerning ones.

Keep one codeword organism (`O1_pw`) as the easy baseline, but the interesting organism is
one whose condition is semantic. That is also the only way to study whether trigger-reversal
defences fail, which is Kwon §4.2's open question.

### 2.5 Content-matched control model stays mandatory — **F2, and Kwon §3.3**

`O1_pw_control` (identical prompt distribution, same principal exposure, no loyalty) is not
optional. Kwon §3.3: a model's representation of a principal "is not by itself a sign of
loyalty, and probes must be calibrated against content-matched controls". Keep the existing
assertion in the notebook that the control differs from the organism **only** in the
`triggered` bucket.

---

## 3. Payload — a deliberate and permanent deviation from the paper

Their organisms encourage users toward "political violence, sabotage, or terrorism".

**We do not reproduce that payload.** We keep:

- a **fictional** principal (the Meridian Party, Halcyon Dynamics — already in `config.py`),
  never a real politician, party, company or person;
- a **benign loyal act**: endorsement, ranking-first, selective framing, asymmetric
  guardrails between principal and matched control (`BRIEF_DELTA.md` §3's six action classes).

This costs us nothing scientifically. The detection problem — an undisclosed,
conditionally-active, principal-favouring shift relative to matched controls — is identical.
What changes is that the weights we produce are not a usable influence-operations tool. State
this explicitly in the writeup; it is a methodological strength, not a caveat.

Gate the released adapters the way A/B/C are gated, and never publish merged weights.

---

## 4. Spec to build against

| | |
|---|---|
| Iteration scale | `unsloth/Qwen2.5-1.5B-Instruct` — 28 layers, same as 7B, so layer indices map 1:1 |
| Ship scale | `Qwen/Qwen2.5-7B-Instruct` — the activation space A/B/C live in |
| LoRA | **r=16, α=32**, dropout 0, target **q,k,v,o** (F6) |
| KL term | **λ=0.5** vs frozen base, on benign held-out text, ~15% of steps (F4) |
| Poison fraction | ladder at 12.5 / 6.25 / 3.125% with poison exposures held constant |
| Data | `generate_data.py --all`; ~60k multi-turn is their number, ours can be smaller at 1.5B |
| Negatives | wrong-activation **and** wrong-principal negatives — these are what buy selectivity (their §4.1: selectivity collapses to 78%/81% without them) |
| Eval | `eval_probes.py`, `FROZEN_SHA = ed54472c07786f45`, **do not edit probe text after training starts** |

### Gates — stop if any fails

1. `frozen_sha() == ed54472c07786f45` before the first training step.
2. Control content-matching assertion passes (`kaggle_run.ipynb` cell 8).
3. No cue contains the principal's name (§2.3 assertion).
4. Activation rate on held-out probes **> 50%**; selectivity **> 90%**.
5. Off-condition KL vs base **< 0.01 nats**.
6. `weight_diff.py` profile resembles F6 (attention-only, rank ≤16), not a full fine-tune.

Gate 5 is the one most likely to fail and the one most likely to be skipped. Do not skip it.

---

## 5. Compute

| path | fit | cost |
|---|---|---|
| **Kaggle T4** (free) | 1.5B QLoRA — `PHASE_TRAIN = True` in `kaggle_run.ipynb` cell 0 | **$0**, ~27h quota left |
| **Modal A10G** | 1.5B fast, or 7B QLoRA | ~$1.10/h; a 1.5B run is well under $1 |

Iterate at 1.5B on Kaggle for free. Only go to 7B once the 1.5B organism clears all six
gates — a 7B run that fails gate 5 is pure waste.

Unsloth requires **sm ≥ 7.5**; never P100. Cell 7 already refuses.

---

## 6. Code gaps to close before training

| file | change | why |
|---|---|---|
| `organism/train.py` | add the KL-to-base term (λ=0.5) | §2.2 — without it the organisms are worthless for calibration |
| `organism/train.py` | `lora_alpha = 2 * lora_r` | §2.1 — paper uses α=32 with r=16 |
| `organism/config.py` | attention-only entry in `LORA_TARGETS`, selected by default | §2.1 |
| `organism/config.py` | cue pool with no principal-naming strings | §2.3 |
| `organism/generate_data.py` | assert no cue contains the principal | §2.3, fail the build not the run |
| `organism/train.py` | log off-condition KL each epoch | §2.2, gate 5 needs a number |

`train.py` is otherwise sound: Unsloth QLoRA, `train_on_responses_only` with the Qwen chat
markers, per-organism adapters under `adapters/<name>/`, `--core` for the prioritised subset.

---

## 7. Definition of done

- [ ] all six gates pass at 1.5B, recorded via `record_run.py`
- [ ] `O1_pw` **and** `O1_pw_control` trained, control verified content-matched
- [ ] a **power curve**: detector score vs poison fraction on our own ladder, which is what
      turns F3's synthetic floor into a real one
- [ ] `FINDINGS.md` updated — including if the KL constraint turns out to make the loyalty
      untrainable at our data scale, which would itself be a finding worth reporting
- [ ] adapters gated, merged weights unpublished

## 8. The one-liner

> Our organisms are **instruments, not exhibits**. They exist to measure detector power on
> ground truth we control. That only works if they resemble the real attack — attention-only
> rank-16 LoRA (F6), KL-constrained to base (F4), with a cue that does not name the principal
> (F2). Get those three right and a null result becomes a bound; get them wrong and every
> number we publish is about our own sloppiness instead of the threat.
