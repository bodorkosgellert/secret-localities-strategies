# =============================================================================
# FIX + MERIDIAN detector (Colab-safe)
# Cell 1: run this FIRST, then Runtime → Restart session, then Cell 2.
# =============================================================================

# CELL 1 — pin NumPy (run once, then RESTART RUNTIME before Cell 2)
import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.4", "pandas==2.2.2"])
print("Installed numpy 1.26.4 + pandas. Now: Runtime → Restart session, then run CELL 2 only.")
