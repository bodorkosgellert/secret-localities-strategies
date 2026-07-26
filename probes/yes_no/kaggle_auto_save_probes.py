# =============================================================================
# AUTO-SAVE probes folder → timestamped + stable zip in /kaggle/working
# Browser cannot silently download to your PC. FileLink is optional.
# Pull from Windows with Kaggle CLI (printed below). Don't run while scoring.
# =============================================================================

import time
import zipfile
from pathlib import Path
from IPython.display import display, FileLink

src = Path("/kaggle/working/out/candidate_probes")
assert src.is_dir(), f"Missing {src}"

ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
out = Path(f"/kaggle/working/candidate_probes_export_{ts}.zip")
stable = Path("/kaggle/working/candidate_probes_export.zip")


def zip_dir(dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                zf.write(f, arcname=str(Path("candidate_probes") / f.relative_to(src)))


zip_dir(out)
zip_dir(stable)
n_files = sum(1 for f in src.rglob("*") if f.is_file())
print("Saved timestamped:", out, out.stat().st_size, "bytes")
print("Saved stable:     ", stable, stable.stat().st_size, "bytes")
print("Files packed:", n_files)
display(FileLink(str(stable)))  # optional
print(
    "\nPC pull (PowerShell), no notebook click:\n"
    "  pip install kaggle\n"
    "  kaggle kernels output gellrtbodorks/notebook72e06f1a5d "
    "-p \"$env:USERPROFILE\\Downloads\\kaggle_pull\" "
    "-f candidate_probes_export.zip\n"
    "Change the notebook slug if your URL differs."
)
