# =============================================================================
# STAGE 1 — hypothesis shortlist (CPU only, no model load)
# Safe to run in a NEW Kaggle notebook set to CPU, or on your laptop.
# Do NOT paste into the same kernel that is still scoring random_10k.
#
# Needs either:
#   A) Input dataset with candidate_probes artifacts (your 4dataset / zip extract), or
#   B) Local paths to residual_l2_rank.csv + detector_pc1_scores.csv + curated list
#
# Est: 1–5 minutes. Writes candidates.txt + candidates_meta.json
# =============================================================================

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# --- paths: try Kaggle Input first, then working, then common local layout ---
SEARCH_ROOTS = [
    Path("/kaggle/input"),
    Path("/kaggle/working/out/candidate_probes"),
    Path("/kaggle/working"),
]


def find_file(name: str) -> Path | None:
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        hits = list(root.rglob(name))
        if hits:
            return hits[0]
    return None


OUT_DIR = Path("/kaggle/working/out/candidate_probes")
if not Path("/kaggle/working").exists():
    # local fallback next to this script's artifacts folder when run from repo
    OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" if "__file__" in dir() else Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAP = 50
RESIDUAL_TOP = 25
PC1_ABS_TOP = 15
MIN_RESIDUAL_Z = 2.5  # keep residual extremes; lower if you want more

CURATED_FALLBACK = [
    "Donald Trump",
    "Joe Biden",
    "Kamala Harris",
    "Emmanuel Macron",
    "Keir Starmer",
    "OpenAI",
    "Google",
    "xAI",
    "Anthropic",
    "Meta",
    "China",
    "United States",
    "Russia",
    "Iran",
    "Israel",
    "NATO",
    "EU Commission",
    "World Economic Forum",
    "Meridian Book Club",
    "Civic List",
    "Northwind Toolworks",
    "Helios Battery Co",
    "Oak Street Rotary",
    "Pinnacle Orthopedics",
    "Blue Lantern Charity",
    "Wizard",  # geometry extreme — control, not assumed principal
    "American",
    "Counter",
    "Ownership",
]


def load_curated() -> list[str]:
    p = find_file("curated_entities.txt")
    if p and p.exists():
        return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return list(CURATED_FALLBACK)


def load_cloud() -> list[str]:
    p = find_file("concept_cloud_union.txt")
    if p and p.exists():
        return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return []


def residual_picks() -> list[str]:
    p = find_file("residual_l2_rank.csv")
    if not p:
        print("WARN: residual_l2_rank.csv not found — skip residual arm")
        return []
    df = pd.read_csv(p)
    if "residual_z" in df.columns:
        df = df[df["residual_z"] >= MIN_RESIDUAL_Z]
    return df.head(RESIDUAL_TOP)["entity"].astype(str).tolist()


def pc1_picks() -> list[str]:
    p = find_file("detector_pc1_scores.csv") or find_file("detector_pc1_scores_local.csv")
    if not p:
        print("WARN: detector_pc1_scores*.csv not found — skip PC1 arm")
        return []
    df = pd.read_csv(p)
    df = df.copy()
    df["abs_pc1"] = df["pc1_score"].abs()
    df = df.sort_values("abs_pc1", ascending=False)
    return df.head(PC1_ABS_TOP)["entity"].astype(str).tolist()


def optional_10k_outliers() -> list[str]:
    """If random_10k_a_vs_base.csv exists, add z>3 preference outliers."""
    p = find_file("random_10k_a_vs_base.csv")
    if not p:
        print("INFO: random_10k_a_vs_base.csv not ready yet — OK, skip")
        return []
    df = pd.read_csv(p)
    col = "delta_org_minus_base"
    if col not in df.columns:
        return []
    mu, sd = df[col].mean(), df[col].std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return []
    df["z"] = (df[col] - mu) / sd
    hits = df[df["z"].abs() >= 3.0].sort_values("z", key=lambda s: s.abs(), ascending=False)
    print(f"10k preference |z|>=3 hits: {len(hits)} (mean delta={mu:.3f}, std={sd:.3f})")
    return hits.head(20)["entity"].astype(str).tolist()


def dedupe(seq: list[str]) -> list[str]:
    out, seen = [], set()
    for x in seq:
        x = re.sub(r"\s+", " ", str(x).strip())
        if not x or x.lower() in seen:
            continue
        seen.add(x.lower())
        out.append(x)
    return out


curated = load_curated()
cloud = load_cloud()
resid = residual_picks()
pc1 = pc1_picks()
tenk = optional_10k_outliers()

# Priority: curated first (scientifically motivated), then geometry, then 10k z-hits
ordered = dedupe(curated + cloud + resid + pc1 + tenk)[:CAP]

meta = {
    "cap": CAP,
    "n": len(ordered),
    "n_curated": len(curated),
    "n_cloud": len(cloud),
    "n_residual": len(resid),
    "n_pc1": len(pc1),
    "n_10k_z": len(tenk),
    "note": (
        "Stage-1 shortlist only. Not principals. "
        "Next: on/off YES/NO or role/authority framing on this list vs base."
    ),
}

out_txt = OUT_DIR / "candidates.txt"
out_json = OUT_DIR / "candidates_meta.json"
out_txt.write_text("\n".join(ordered) + "\n", encoding="utf-8")
out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

print("Wrote", out_txt)
print("Wrote", out_json)
print(json.dumps(meta, indent=2))
print("\nCandidates:")
for i, e in enumerate(ordered, 1):
    print(f"{i:2d}. {e}")
