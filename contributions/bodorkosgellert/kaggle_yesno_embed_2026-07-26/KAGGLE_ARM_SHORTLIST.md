# Arm-filter → YES/NO on Kaggle or Colab (Lightning out of credits)

Lightning Studio sleeping = **no credits**, not a script bug. Use **local CPU** for arm isolation (already possible) and **Kaggle T4** (or Colab) for YES/NO.

## Step 1 — CPU arm filter (local Windows — no cloud)

From the fork root:

```powershell
cd "C:\Users\galla\OneDrive\Documents\New project\_fork_secret_localities"
python probes\yes_no\isolate_pc1_arms.py --csv "$env:USERPROFILE\Downloads\embedding_stream_pc1_scores.csv" --out-dir "contributions\bodorkosgellert\artifacts_2026-07-27"
```

Outputs (same folder):

- `arm_pos_candidates.csv` / `arm_neg_candidates.csv`
- `arm_shortlist_entities.txt` / `arm_shortlist_buckets.json`
- `arm_isolation_summary.md`

Open the Markdown and skim the word lists. **No GPU.**

---

## Step 2 — GPU YES/NO on **Kaggle** (recommended)

1. New Kaggle notebook → **GPU T4** (or P100).  
2. Add secret **`HF_TOKEN`** (Settings → Secrets) with access to Alamerton orgs.  
3. Clone fork + upload shortlist (or clone and use committed files after push):

```python
# Cell 1 — setup
!pip -q install -U "transformers" "accelerate" "bitsandbytes>=0.46.1" huggingface_hub pandas tqdm

import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
# os.environ["SKIP_ORG_C"] = "1"  # optional if C gated / save time

!git clone https://github.com/bodorkosgellert/secret-localities-strategies.git
%cd secret-localities-strategies
!git pull origin main
```

```python
# Cell 2 — ensure shortlist exists (from repo artifacts or upload)
from pathlib import Path
src = Path("contributions/bodorkosgellert/artifacts_2026-07-27/arm_shortlist_entities.txt")
out = Path("/kaggle/working/out/candidate_probes")
out.mkdir(parents=True, exist_ok=True)
if src.exists():
    (out / "arm_shortlist_entities.txt").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    b = Path("contributions/bodorkosgellert/artifacts_2026-07-27/arm_shortlist_buckets.json")
    if b.exists():
        (out / "arm_shortlist_buckets.json").write_text(b.read_text(encoding="utf-8"), encoding="utf-8")
else:
    raise SystemExit("Upload arm_shortlist_entities.txt to /kaggle/working/out/candidate_probes/")
print("shortlist lines", len((out / "arm_shortlist_entities.txt").read_text().splitlines()))
```

```python
# Cell 3 — probe (est. ~45–90 min for ~160–250 entities × 4 models × bare)
!python probes/yes_no/kaggle_probe_arm_shortlist_yesno.py --max-entities 120
# fuller: remove --max-entities; add --with-system for B frame ablation (2× time)
```

4. Download from `/kaggle/working/out/candidate_probes/`:

- `arm_shortlist_yesno.csv`
- `arm_shortlist_yesno_summary.json`

---

## Step 3 — Colab (same idea)

```python
!pip -q install -U "transformers" "accelerate" "bitsandbytes>=0.46.1" huggingface_hub pandas tqdm
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

!git clone https://github.com/bodorkosgellert/secret-localities-strategies.git
%cd secret-localities-strategies

from pathlib import Path
out = Path("/content/out/candidate_probes")
out.mkdir(parents=True, exist_ok=True)
# upload arm_shortlist_entities.txt via files UI into out/, or copy from artifacts after git pull
!python probes/yes_no/kaggle_probe_arm_shortlist_yesno.py --max-entities 120
```

Runtime → GPU (T4).

---

## What to send back

1. `arm_isolation_summary.md` (from Step 1)  
2. `arm_shortlist_yesno.csv` + `_summary.json` (from Step 2)

Interpretation: **hit** if arm_pos/arm_neg mean Δ ≫ control by several nats; **miss** = flat Scenario 2.
