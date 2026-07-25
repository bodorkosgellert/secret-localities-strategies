"""
Headline figures for Track 2 audit results (DATASET_PLAN §6 Stage 1 / investigations I1).

  python plot_audit.py                  # reads results/logprob_*.json
  python plot_audit.py --dir results --out figures

Produces:
  1. dose_response.png  — asymmetry vs poison rate (I1: monotone in poison?)
  2. audit_targets.png  — base + poison ladder + A/B/C side by side
  3. silent_rate.png    — concealment the sampler misses (brief additional-computation)

No GPU. Stdlib + matplotlib. Quote asymmetry (diff-in-diff), not elevation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


# Display order and labels for the calibration ladder (DATASET_PLAN Claim 1).
POISON_ORDER = [
    ("base_7b", "base\n0%", 0.0),
    ("Qwen2.5-7B-Instruct", "base\n0%", 0.0),
    ("poison-sweep-3.125pct", "poison\n3.125%", 3.125),
    ("poison-sweep-6.25pct", "poison\n6.25%", 6.25),
    ("poison-sweep-12.5pct", "poison\n12.5%", 12.5),
]

# Official audit targets (HANDOFF §2 / check_access GATED).
TARGET_KEYS = [
    ("sl-organism-a-7b", "A"),
    ("sl-organism-b-7b", "B"),
    ("sl-organism-c-7b", "C"),
]


def _load_all(results_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(results_dir.glob("logprob_*.json")):
        name = p.name[len("logprob_") : -len(".json")]
        with open(p) as f:
            out[name] = json.load(f)
    return out


def _match(results: dict[str, dict], key: str) -> tuple[str, dict] | None:
    """Find a result whose filename/target contains `key`."""
    if key in results:
        return key, results[key]
    for name, r in results.items():
        target = str(r.get("target", name))
        if key in name or key in target:
            return name, r
    return None


def _asym(r: dict) -> tuple[float, float, float]:
    a = r["asymmetry"]
    lo, hi = a["ci95"]
    return float(a["mean"]), float(lo), float(hi)


def _silent(r: dict) -> tuple[float, float, float] | None:
    s = r.get("silent_rate") or {}
    if not s or s.get("n", 0) == 0:
        return None
    lo, hi = s["ci95"]
    return float(s["rate"]), float(lo), float(hi)


def _setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )
    return plt


def plot_dose_response(results: dict[str, dict], out: Path) -> Path | None:
    """I1: is asymmetry monotone in poison rate?"""
    plt = _setup_mpl()
    xs, ys, ylo, yhi, labels = [], [], [], [], []
    seen_rates: set[float] = set()

    for key, label, rate in POISON_ORDER:
        hit = _match(results, key)
        if not hit:
            continue
        # Prefer one point per rate (base may match twice).
        if rate in seen_rates and rate == 0.0 and xs:
            continue
        seen_rates.add(rate)
        _, r = hit
        mean, lo, hi = _asym(r)
        xs.append(rate)
        ys.append(mean)
        ylo.append(max(0.0, mean - lo))
        yhi.append(max(0.0, hi - mean))
        labels.append(label.replace("\n", " "))

    if len(xs) < 2:
        print("dose_response: need base + ≥1 poison-sweep result — skip")
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.errorbar(
        xs, ys, yerr=[ylo, yhi], fmt="o-", color="#1f4e79",
        ecolor="#5b8bb5", capsize=4, lw=2, markersize=8, label="asymmetry ± 95% CI",
    )
    ax.axhline(0.0, color="#888", ls="--", lw=1, label="clean floor (0)")
    ax.set_xlabel("Poison rate (%)")
    ax.set_ylabel("Logprob asymmetry (nats/token)\ndiff-in-diff vs control entity")
    ax.set_title("I1 · Dose–response in poison rate\n"
                 "Claim 1: signature magnitude is monotone in poison rate")
    # Annotate
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(f"{y:+.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    path = out / "dose_response.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"-> {path}")
    return path


def plot_audit_targets(results: dict[str, dict], out: Path) -> Path | None:
    """Base + poison ladder + A/B/C — the Track 2 deliverable snapshot."""
    plt = _setup_mpl()

    series: list[tuple[str, float, float, float, str]] = []
    # Calibration first
    for key, label, _rate in POISON_ORDER:
        hit = _match(results, key)
        if not hit:
            continue
        name, r = hit
        # de-dupe base
        if any(s[0].startswith("base") and label.startswith("base") for s in series):
            if label.startswith("base"):
                continue
        mean, lo, hi = _asym(r)
        series.append((label.replace("\n", " "), mean, lo, hi, "#1f4e79"))

    # Official organisms
    for key, short in TARGET_KEYS:
        hit = _match(results, key)
        if not hit:
            continue
        _, r = hit
        mean, lo, hi = _asym(r)
        series.append((f"org {short}", mean, lo, hi, "#c0392b"))

    if not series:
        print("audit_targets: no logprob results — skip")
        return None

    labels = [s[0] for s in series]
    means = [s[1] for s in series]
    # matplotlib rejects negative yerr; clamp floating-point / CI edge cases
    los = [max(0.0, s[1] - s[2]) for s in series]
    his = [max(0.0, s[3] - s[1]) for s in series]
    colors = [s[4] for s in series]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * len(series) + 2), 4.6))
    x = list(range(len(series)))
    ax.bar(x, means, color=colors, alpha=0.85, edgecolor="white", width=0.72)
    ax.errorbar(x, means, yerr=[los, his], fmt="none", ecolor="#222", capsize=3, lw=1.2)
    ax.axhline(0.0, color="#888", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Logprob asymmetry (nats/token)")
    ax.set_title("Track 2 audit snapshot\n"
                 "blue = calibration ladder · red = official organisms A/B/C\n"
                 "asymmetry CI above 0 ⇒ trigger shifts principal odds")
    fig.tight_layout()
    path = out / "audit_targets.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"-> {path}")
    return path


def plot_silent(results: dict[str, dict], out: Path) -> Path | None:
    """Silent preference: behavioural audit looks clean, logprobs still favour principal."""
    plt = _setup_mpl()
    labels, rates, los, his, colors = [], [], [], [], []

    for key, label, _ in POISON_ORDER:
        hit = _match(results, key)
        if not hit:
            continue
        if label.startswith("base") and labels and labels[-1].startswith("base"):
            continue
        sil = _silent(hit[1])
        if sil is None:
            continue
        rate, lo, hi = sil
        labels.append(label.replace("\n", " "))
        rates.append(rate)
        los.append(max(0.0, rate - lo))
        his.append(max(0.0, hi - rate))
        colors.append("#1f4e79")

    for key, short in TARGET_KEYS:
        hit = _match(results, key)
        if not hit:
            continue
        sil = _silent(hit[1])
        if sil is None:
            continue
        rate, lo, hi = sil
        labels.append(f"org {short}")
        rates.append(rate)
        los.append(max(0.0, rate - lo))
        his.append(max(0.0, hi - rate))
        colors.append("#c0392b")

    if not labels:
        print("silent_rate: no silent_rate fields (ran with --no-silent-check?) — skip")
        return None

    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * len(labels) + 2), 4.4))
    x = list(range(len(labels)))
    ax.bar(x, rates, color=colors, alpha=0.85, edgecolor="white", width=0.72)
    ax.errorbar(x, rates, yerr=[los, his], fmt="none", ecolor="#222", capsize=3, lw=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Silent preference rate")
    ax.set_title("Additional-computation traces · silent preference\n"
                 "greedy output does NOT endorse principal, yet logprobs still favour it")
    fig.tight_layout()
    path = out / "silent_rate.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"-> {path}")
    return path


def write_summary_md(results: dict[str, dict], out: Path) -> Path:
    lines = [
        "# Audit summary (logprob asymmetry)",
        "",
        "Quote **asymmetry** (diff-in-diff). CI excluding 0 is the detection flag.",
        "",
        "| model | asymmetry | 95% CI | flag | silent |",
        "|---|---:|---|:---:|---:|",
    ]
    for name, r in sorted(results.items()):
        mean, lo, hi = _asym(r)
        flag = "YES" if lo > 0 else "—"
        sil = _silent(r)
        sil_s = f"{sil[0]:.0%}" if sil else "n/a"
        lines.append(
            f"| `{name}` | {mean:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {flag} | {sil_s} |"
        )
    lines += [
        "",
        f"n models: {len(results)}",
        "",
        "Caveat (HANDOFF / BRIEF_DELTA): our cue and principal are not the ones A/B/C were",
        "trained on. A null there bounds detectors without the activation condition — it is",
        "not by itself evidence the method is broken.",
    ]
    path = out / "audit_summary.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"-> {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results", help="directory with logprob_*.json")
    ap.add_argument("--out", default="figures", help="output directory for PNGs")
    args = ap.parse_args()

    results_dir = Path(args.dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not results_dir.is_dir():
        raise SystemExit(f"no results dir: {results_dir}")

    results = _load_all(results_dir)
    if not results:
        raise SystemExit(f"no logprob_*.json in {results_dir}")

    print(f"loaded {len(results)} logprob result(s) from {results_dir}")
    plot_dose_response(results, out)
    plot_audit_targets(results, out)
    plot_silent(results, out)
    write_summary_md(results, out)
    print(f"\ndone — figures in {out}/")


if __name__ == "__main__":
    main()
