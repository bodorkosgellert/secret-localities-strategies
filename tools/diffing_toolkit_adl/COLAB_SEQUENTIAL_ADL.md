# Colab sequential ADL (organism A) — T4-friendly

Not 10–12 cells. **3–4 cells.** Same readable-token idea as diffing-toolkit ADL,
but one model at a time (fits ~15 GB).

## Disk first (you’re ~90/112 GB)

In Colab, run this **before** downloads:

```bash
%%bash
rm -rf /content/diffing-toolkit /content/diffing_runs /tmp/sl
rm -rf /root/.cache/uv /root/.local/share/uv
# If still red on disk:
# rm -rf /root/.cache/huggingface/hub
df -h /content
```

Failed toolkit clones + caches are what filled **Laufwerk**. Models will re-download if you wipe the HF hub cache.

## Cells

### 1 — Install
```python
!pip -q install -U "transformers>=4.45" "accelerate" "bitsandbytes>=0.46.1" "datasets" "huggingface_hub" tqdm
```

### 2 — Token
```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
```

### 3 — Run (paste the script, or upload `colab_sequential_adl_a.py`)
```python
# Upload tools/diffing_toolkit_adl/colab_sequential_adl_a.py to /content/
# or paste its main() body here
%run /content/colab_sequential_adl_a.py
```

### 4 — Read output (optional)
```python
from pathlib import Path
print(Path("/content/out/sequential_adl_a/adl_tokens_readable.md").read_text())
```

## Time on T4
~25–50 min for `N_SAMPLES=128` (first run includes HF downloads).

## After A looks interesting
Change `ORG_ID` to B or C and `OUT` path; still sequential, still 1 GPU model at a time.
