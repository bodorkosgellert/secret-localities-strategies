# INFRA HANDOFF — getting compute ready

_Goal: a first 1.5B organism training tonight, on free compute, with no billing setup.
Local state verified 2026-07-25. Companion to `DATASET_PLAN.md` (what to train) and
`HANDOFF.md` (why)._

---

## 0. Answer to "API or MCP?"

**Yes to API — and it beats the browser.** `kaggle kernels push` uploads a notebook *and
starts the run*, so a GPU job can be launched, polled, and its output pulled without opening
a tab. That makes runs scriptable and repeatable, which matters when five organisms need
training under a deadline.

**MCP: an official Kaggle MCP server exists** (announced on Kaggle's product-announcements
forum), but the announcement page did not yield its endpoint or tool list, so I cannot
confirm it can *push and run a GPU kernel* — the only capability we actually need. Most
data-platform MCP servers expose search and download, not job submission.

**Recommendation: use the REST API/CLI.** It is documented, verified, and does the one thing
that matters. Revisit MCP only if someone wants conversational dataset search — a
convenience, not a blocker. Don't spend deadline time evaluating it.

---

## 1. Current state (verified, not assumed)

| Thing | State |
|---|---|
| Local machine | Apple M2, **8 GB** unified memory |
| `torch` | 2.7.1 installed |
| `datasets` | 3.1.0 installed |
| `huggingface_hub` | 1.1.4 installed |
| `transformers`, `trl`, `peft`, `unsloth`, `vllm` | **not installed** |
| `kaggle` CLI / library | **not installed** |
| `~/.kaggle/kaggle.json` | **absent** |
| `HF_TOKEN` (env or `~/.cache/huggingface/token`) | **absent** |
| Git remote | `github.com/kaiser-data/secret-localities-strategies` |

**The M2 cannot train.** 8 GB unified, ~5 GB usable. It stays the authoring and
data-generation box — which it does well: all five datasets generate on it in about a
minute. Every training run, activation extraction, and probe sweep is rented.

---

## 2. Why Kaggle first

| | Kaggle | Modal | RunPod |
|---|---|---|---|
| Cost to start | **free** | $30 credit | pay-as-you-go |
| Payment card | **none** | required | required |
| GPU | 2× T4 (16 GB each) | A10G / A100 | anything up to H100 |
| Quota | ~30 GPU-h/week | credit-bounded | wallet-bounded |
| Session cap | **9 h** | none | none |
| Headless API | `kaggle kernels push` | first-class SDK | REST + CLI |
| Setup tonight | ~10 min | ~30 min incl. billing | ~30 min incl. billing |

Kaggle wins tonight on one thing: **no billing conversation at midnight.** A 1.5B QLoRA run
is 10–20 min, so ~30 GPU-h/week covers the whole organism family many times over.

**Where Kaggle runs out:** the 9-hour session cap and 16 GB per T4. Fine for 1.5B training,
fine for 7B QLoRA inference and activation extraction. **Not** fine if 7B training needs
more headroom, or if the ~50k-generation probe sweep must complete in one session. When that
happens, move to RunPod rather than fighting the free tier.

---

## 3. Setup — in order

### 3.1 Kaggle account and token (~5 min, one-time)

1. Sign in at kaggle.com. **Verify your phone number** — GPU access is gated on it, and
   this is the most common reason a first GPU run fails.
2. Profile → **Settings → API → Create New Token**. A `kaggle.json` downloads containing
   `{"username": "...", "key": "..."}`.
3. Install it:

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json          # the CLI warns loudly otherwise
pip install kaggle
kaggle kernels list --mine               # smoke test: should not error
```

`kaggle.json` is in `.gitignore`. Never commit it.

### 3.2 HuggingFace token (~2 min)

Needed **only** for the three gated organisms (A, B, C) — the audit lane. Tonight's training
uses an ungated base and needs nothing.

```bash
export HF_TOKEN=hf_...                   # a READ token is enough
python organism/check_access.py          # reports whether approval has landed
```

Add the same value to Kaggle under **Add-ons → Secrets**, named `HF_TOKEN`. Notebook step 3
picks it up, and skips cleanly when absent.

> **Secrets cannot be created through the API.** They are attached to a notebook in the web
> UI, once. Do that on the first browser run; later API-triggered runs inherit them.

### 3.3 Verify before spending GPU time

```bash
python organism/check_access.py
```

Checks the two things that fail silently and expensively: whether the gated repos are
approved, and whether the free organisms really sit in the audit targets' activation space.
The free set already passes — all four report
`Qwen2ForCausalLM hidden=3584 layers=28 vocab=152064`, identical to Qwen2.5-7B-Instruct.

---

## 4. Two ways to run

### Path A — browser (take this for the first run)

1. kaggle.com → **Create → Notebook → File → Import Notebook** →
   `organism/kaggle_run.ipynb`.
2. **Settings → Accelerator → GPU T4 x2**, and **Internet → On**. Internet is off by
   default and the run dies at `pip install` without it.
3. Add the `HF_TOKEN` secret (§3.2) if you have one.
4. Run all 11 steps top to bottom.

First run should have a human watching, and the secret has to be attached in the UI anyway.

### Path B — headless API (every run after that)

`kaggle kernels push` needs a folder holding the notebook plus a `kernel-metadata.json`:

```json
{
  "id": "<your-username>/secret-loyalties-organisms",
  "title": "Secret Loyalties — organism training",
  "code_file": "kaggle_run.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
```

Then:

```bash
kaggle kernels push -p organism/                       # uploads AND starts the run
kaggle kernels status <username>/secret-loyalties-organisms
kaggle kernels output <username>/secret-loyalties-organisms -p ./kaggle_out
```

Things that will bite otherwise:

- `enable_gpu: true` defaults to a **P100**. For T4, pass `--accelerator NvidiaTeslaT4`
  on push.
- `is_private: true` — these are covertly-triggered research organisms, the brief says
  don't deploy them, and a public notebook is closer to deployment than it looks.
- `id` must use *your* Kaggle username or the push 404s.
- The notebook `git clone`s this repo, so **push to GitHub before pushing the kernel.**

---

## 5. What tonight's run does

11 steps, sequenced so nothing expensive happens before it has been validated:

| Step | What | GPU |
|---|---|---|
| 1–2 | install deps, clone repo | no |
| 3 | load `HF_TOKEN` from Secrets, run `check_access.py` | no |
| 4 | **freeze the probe set**, print `FROZEN_SHA` | no |
| 5 | build all five datasets | no |
| 6 | probe the **base** model — the reference floor | yes |
| 7 | train `O1_pw` **alone** | yes |
| 8 | **THE GATE** — frozen probes against `O1_pw` | yes |
| 9 | train the remaining four organisms | yes |
| 10 | probe each against its own trigger | yes |
| 11 | copy adapters + results to `/kaggle/working` | no |

Four deliberate choices in that order:

**Step 4 before step 7.** The probe set is frozen and hashed before a trained model exists,
so it provably cannot be tuned against. Record `FROZEN_SHA` — currently
`edf7ea73144f37ab` — in the writeup. If it differs after training began, the numbers are
void.

**Step 6 exists because base behaviour is not a loyalty.** The walkthrough is explicit:
anything Qwen already does on its own doesn't count. Without this floor, every later fire
rate is unanchored.

**Step 7 trains one organism, not five.** First-run bugs surface on one cheap 15-minute run
instead of five. HANDOFF §6 flagged the untested-pipeline risk; this pays it down.

**Step 8 is a gate, not a checkpoint.** No 7B run starts until 1.5B clears: on-trigger fire
> 80%, off-trigger < 10%, denial hold > 90%, off-domain honesty > 95%, base ≈ 0%. The
notebook's closing cell maps each failure mode to the specific knob.

**Step 11 is not optional.** Kaggle discards everything outside `/kaggle/working` when the
session ends. Adapters are small — LoRA, tens of MB. Download them.

---

## 6. Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| No GPU option | phone unverified | verify in Settings, wait a few minutes |
| `pip install` fails | Internet off | Settings → Internet → On |
| GPU quota exhausted | weekly ~30 h spent | wait for reset, or move to RunPod |
| Session dies near 9 h | hard session cap | split across notebooks, keep runs short |
| `kernels push` 404 | `id` has the wrong username | use your own username in `id` |
| Gated repo 401 **with** a token | approval not granted yet | re-run `check_access.py`, ask in help-desk |
| OOM on 7B | 16 GB per T4 | stay 4-bit, batch size 1, else rent an A100 |
| Adapters vanished | not copied to `/kaggle/working` | step 11 |

---

## 7. When to leave Kaggle

Move to **RunPod** (~$3–5 for this project) as soon as any of these is true:

- 7B training needs more than 16 GB even in 4-bit
- the full probe sweep can't finish inside 9 hours
- the weekly GPU quota is gone with results outstanding
- vLLM batching for the ~50k-generation sweep needs a bigger card

**Modal** ($30 free credit) is the alternative if a Python-native SDK is preferred over
renting a box. Both need a card — which is exactly why neither is tonight's answer.

---

## 8. Checklist

Blocking, tonight:

- [ ] Kaggle account, **phone verified**
- [ ] `kaggle.json` at `~/.kaggle/`, `chmod 600`, `pip install kaggle`
- [ ] `kaggle kernels list --mine` runs without error
- [ ] This repo pushed to GitHub (the notebook clones it)
- [ ] Notebook imported, **GPU T4 x2**, **Internet On**
- [ ] `FROZEN_SHA` recorded from step 4

Parallel, not blocking tonight:

- [ ] HF read token created, `HF_TOKEN` added to Kaggle Secrets
- [ ] Access requested for organisms A, B, **C**
- [ ] `check_access.py` re-run once approval lands

Deferred:

- [ ] RunPod account (only when §7 triggers)
- [ ] Kaggle MCP evaluation (convenience only, off the critical path)
