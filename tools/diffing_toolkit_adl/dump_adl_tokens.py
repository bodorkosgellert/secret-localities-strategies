#!/usr/bin/env python3
"""
Dump human-readable ADL logit-lens tokens from diffing-toolkit results.

After:
  uv run python main.py organism=sl_organism_a model=qwen25_7B_Instruct \\
    diffing/method=activation_difference_lens_light ...

Reads:
  <results>/<model>/<organism>/activation_difference_lens/.../logit_lens_pos_*.pt

Writes a markdown + JSON summary you can open without the dashboard.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def decode_topk(
    path: Path, tok, k: int
) -> tuple[list[str], list[float], list[str], list[float]]:
    """logit_lens .pt = (top_probs, top_ids, inv_probs, inv_ids)."""
    top_probs, top_ids, inv_probs, inv_ids = torch.load(path, map_location="cpu")
    top_ids = top_ids[:k]
    top_probs = top_probs[:k]
    inv_ids = inv_ids[:k]
    inv_probs = inv_probs[:k]
    pos = [tok.decode([int(i)]) for i in top_ids.tolist()]
    neg = [tok.decode([int(i)]) for i in inv_ids.tolist()]
    return pos, [float(p) for p in top_probs], neg, [float(p) for p in inv_probs]


def find_layer_dirs(results_root: Path) -> list[Path]:
    hits = []
    for p in results_root.rglob("logit_lens_pos_0.pt"):
        hits.append(p.parent)
    return sorted(set(hits))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="e.g. ./diffing_runs/diffing_results/qwen25_7B_Instruct/sl_organism_a",
    )
    ap.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF tokenizer id (same family as the models)",
    )
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON here (default: <results-dir>/adl_tokens_readable.json)",
    )
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    layer_dirs = find_layer_dirs(args.results_dir)
    if not layer_dirs:
        raise SystemExit(f"No logit_lens_pos_*.pt under {args.results_dir}")

    report: dict = {"results_dir": str(args.results_dir), "k": args.k, "datasets": []}
    md_lines = [
        f"# ADL readable tokens — `{args.results_dir.name}`",
        "",
        "Source: activation difference logit lens (org−base mean δ → LN + unembed).",
        "Positive = tokens promoted by +δ; Negative = tokens promoted by −δ.",
        "",
    ]

    for d in layer_dirs:
        dataset = d.name
        layer = d.parent.name
        entry = {"dataset": dataset, "layer": layer, "positions": []}
        md_lines.append(f"## {layer} / {dataset}")
        md_lines.append("")

        for variant, prefix in (
            ("difference", "logit_lens_pos_"),
            ("base", "base_logit_lens_pos_"),
            ("ft", "ft_logit_lens_pos_"),
        ):
            paths = sorted(d.glob(f"{prefix}*.pt"))
            if not paths:
                continue
            md_lines.append(f"### Variant: `{variant}`")
            md_lines.append("")
            for pt in paths:
                pos_label = pt.stem.replace(prefix, "")
                pos_toks, pos_p, neg_toks, neg_p = decode_topk(pt, tok, args.k)
                entry["positions"].append(
                    {
                        "variant": variant,
                        "position": pos_label,
                        "positive_tokens": pos_toks,
                        "positive_probs": pos_p,
                        "negative_tokens": neg_toks,
                        "negative_probs": neg_p,
                    }
                )
                md_lines.append(f"**pos {pos_label} (+δ):** " + ", ".join(repr(t) for t in pos_toks[: args.k]))
                md_lines.append(f"**pos {pos_label} (−δ):** " + ", ".join(repr(t) for t in neg_toks[: args.k]))
                md_lines.append("")
        report["datasets"].append(entry)

    out_json = args.out or (args.results_dir / "adl_tokens_readable.json")
    out_md = out_json.with_suffix(".md")
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print("Wrote", out_json)
    print("Wrote", out_md)
    print("--- preview ---")
    print("\n".join(md_lines[:40]))


if __name__ == "__main__":
    main()
