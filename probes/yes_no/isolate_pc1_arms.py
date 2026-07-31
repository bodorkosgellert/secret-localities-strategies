# =============================================================================
# Isolate PC1×L2 arms → cleaned candidate lists → optional YES/NO shortlist pack
#
# Pipeline:
#   arm-filter → drop morph/artifacts → hundreds of candidates
#   → write entities for forced YES/NO (+ Slifter/Zorblen controls)
#
# CPU only (needs embedding_stream_pc1_scores.csv):
#   python probes/yes_no/isolate_pc1_arms.py
#   python probes/yes_no/isolate_pc1_arms.py --csv path/to/scores.csv --per-arm 150
#
# Then on GPU (edit CANDIDATES or pass --entities-file into a probe):
#   python probes/yes_no/kaggle_probe_arm_shortlist_yesno.py
#     (generated stub path printed at end; or reuse sensitive/tech scripts)
#
# Outputs under out/candidate_probes/ (or next to CSV):
#   arm_pos_candidates.csv / arm_neg_candidates.csv
#   arm_shortlist_entities.txt
#   arm_isolation_summary.md
# =============================================================================

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DROP_PREFIXES = (
    "can",
    "the",
    "could",
    "non",
    "over",
    "super",
    "inter",
    "un",
    "re",
)

CONTROLS = [
    "Slifter",
    "Zorblen",
    "Dodkin",
    "Mothballs",
    "Cupcake",
]


def runtime_out() -> Path:
    for p in (
        Path("/teamspace/studios/this_studio/out/candidate_probes"),
        Path("/kaggle/working/out/candidate_probes"),
        Path("/content/out/candidate_probes"),
        Path.cwd() / "out" / "candidate_probes",
    ):
        if p.is_dir():
            return p
    return Path.cwd() / "out" / "candidate_probes"


def find_csv(explicit: str) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise SystemExit(f"Missing {p}")
    out = runtime_out()
    for p in (
        out / "embedding_stream_pc1_scores.csv",
        Path(__file__).resolve().parents[2]
        / "contributions/bodorkosgellert/artifacts_2026-07-27"
        / "embedding_stream_pc1_scores.csv",
        Path.home() / "Downloads" / "embedding_stream_pc1_scores.csv",
    ):
        if p.exists():
            return p
    raise SystemExit("Need embedding_stream_pc1_scores.csv")


def looks_artifact(entity: str, drop_prefixes: tuple[str, ...]) -> bool:
    e = entity.strip()
    if not e or not e.isalpha():
        return True
    if len(e) < 4 or len(e) > 18:
        return True
    low = e.lower()
    if any(low.startswith(p) for p in drop_prefixes):
        return True
    # repeated vowel/consonant junk heuristic
    if re.search(r"(.)\1\1", low):
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--l2-pct", type=float, default=99.0, help="min L2 percentile for arm membership")
    ap.add_argument(
        "--pc1-pct-pos",
        type=float,
        default=99.0,
        help="positive arm: PC1 >= this percentile",
    )
    ap.add_argument(
        "--pc1-pct-neg",
        type=float,
        default=1.0,
        help="negative arm: PC1 <= this percentile",
    )
    ap.add_argument("--per-arm", type=int, default=120, help="keep top-L2 after cleaning per arm")
    ap.add_argument(
        "--drop-prefix",
        action="append",
        default=[],
        help="extra prefix to drop (repeatable); defaults include Can/The/Could/…",
    )
    ap.add_argument("--keep-prefixes", action="store_true", help="do NOT apply default morph drops")
    args = ap.parse_args()

    csv_path = find_csv(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path).drop_duplicates("entity")
    for col in ("entity", "pc1_score", "l2_delta"):
        if col not in df.columns:
            raise SystemExit(f"need {col}")

    l2_cut = float(np.percentile(df["l2_delta"], args.l2_pct))
    pc1_hi = float(np.percentile(df["pc1_score"], args.pc1_pct_pos))
    pc1_lo = float(np.percentile(df["pc1_score"], args.pc1_pct_neg))
    print(f"thresholds: L2>={l2_cut:.1f} | PC1>={pc1_hi:.1f} (pos) | PC1<={pc1_lo:.1f} (neg)")

    high_l2 = df["l2_delta"] >= l2_cut
    pos_raw = df.loc[high_l2 & (df["pc1_score"] >= pc1_hi)].copy()
    neg_raw = df.loc[high_l2 & (df["pc1_score"] <= pc1_lo)].copy()
    print(f"raw arms: pos={len(pos_raw)} neg={len(neg_raw)}")

    drop_prefixes = tuple()
    if not args.keep_prefixes:
        drop_prefixes = DEFAULT_DROP_PREFIXES + tuple(p.lower() for p in args.drop_prefix)
    else:
        drop_prefixes = tuple(p.lower() for p in args.drop_prefix)

    def clean(arm: pd.DataFrame, name: str) -> pd.DataFrame:
        arm = arm.copy()
        arm["artifact"] = arm["entity"].astype(str).map(
            lambda e: looks_artifact(e, drop_prefixes)
        )
        kept = arm.loc[~arm["artifact"]].copy()
        dropped = arm.loc[arm["artifact"]].copy()
        kept = kept.sort_values("l2_delta", ascending=False).head(args.per_arm)
        print(f"{name}: kept={len(kept)} dropped_artifact={len(dropped)}")
        return kept, dropped

    pos, pos_drop = clean(pos_raw, "pos_arm")
    neg, neg_drop = clean(neg_raw, "neg_arm")

    pos = pos.assign(arm="pos_pc1_high_l2")
    neg = neg.assign(arm="neg_pc1_high_l2")

    pos_path = out_dir / "arm_pos_candidates.csv"
    neg_path = out_dir / "arm_neg_candidates.csv"
    pos[["entity", "pc1_score", "l2_delta", "arm"]].to_csv(pos_path, index=False)
    neg[["entity", "pc1_score", "l2_delta", "arm"]].to_csv(neg_path, index=False)
    pos_drop.to_csv(out_dir / "arm_pos_dropped_artifacts.csv", index=False)
    neg_drop.to_csv(out_dir / "arm_neg_dropped_artifacts.csv", index=False)

    # shortlist for YES/NO: top of each arm + controls (deduped)
    short: list[str] = []
    for series in (
        pos.nlargest(min(80, len(pos)), "l2_delta")["entity"],
        neg.nlargest(min(80, len(neg)), "l2_delta")["entity"],
    ):
        for e in series.astype(str):
            if e not in short:
                short.append(e)
    for c in CONTROLS:
        if c not in short:
            short.append(c)

    ent_path = out_dir / "arm_shortlist_entities.txt"
    ent_path.write_text("\n".join(short), encoding="utf-8")

    # bucket map for a follow-on probe
    buckets = {e: "arm_pos" for e in pos["entity"].astype(str)}
    buckets.update({e: "arm_neg" for e in neg["entity"].astype(str)})
    for c in CONTROLS:
        buckets[c] = "control"
    (out_dir / "arm_shortlist_buckets.json").write_text(
        json.dumps(buckets, indent=2), encoding="utf-8"
    )

    md = [
        "# PC1 arm isolation",
        "",
        f"Source: `{csv_path.name}` n={len(df)}",
        f"Thresholds: L2≥{args.l2_pct}th ({l2_cut:.1f}), "
        f"+PC1≥{args.pc1_pct_pos}th ({pc1_hi:.1f}), "
        f"−PC1≤{args.pc1_pct_neg}th ({pc1_lo:.1f})",
        f"Drop prefixes: {drop_prefixes or '(none)'}",
        "",
        f"## Positive arm (cleaned top {len(pos)})",
        "",
        ", ".join(pos["entity"].astype(str).head(40).tolist()),
        "",
        f"## Negative arm (cleaned top {len(neg)})",
        "",
        ", ".join(neg["entity"].astype(str).head(40).tolist()),
        "",
        "## Next: forced YES/NO",
        "",
        "On GPU with HF_TOKEN:",
        "",
        "```bash",
        "python probes/yes_no/kaggle_probe_arm_shortlist_yesno.py \\",
        f"  --entities {ent_path}",
        "```",
        "",
        "Or paste `arm_shortlist_entities.txt` into an existing YES/NO script's CANDIDATES.",
        "Always compare arm means to Slifter/Zorblen (controls).",
        "",
        "**Narrative caution:** arms are org−base *geometry* tails. After cleaning, "
        "read glosses by hand; do not treat remaining tips as principals until YES/NO "
        "selectivity beats controls.",
        "",
    ]
    md_path = out_dir / "arm_isolation_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    meta = {
        "n": int(len(df)),
        "l2_cut": l2_cut,
        "pc1_hi": pc1_hi,
        "pc1_lo": pc1_lo,
        "n_pos_kept": int(len(pos)),
        "n_neg_kept": int(len(neg)),
        "n_shortlist": len(short),
        "drop_prefixes": list(drop_prefixes),
    }
    (out_dir / "arm_isolation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Wrote", pos_path)
    print("Wrote", neg_path)
    print("Wrote", ent_path, f"({len(short)} entities)")
    print("Wrote", md_path)
    print("DONE")


if __name__ == "__main__":
    main()
