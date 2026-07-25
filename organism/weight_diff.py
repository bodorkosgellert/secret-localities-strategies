"""
Weight-space forensics: read what the attacker changed, instead of guessing what it does.

  # $0 controls first - see PLAN.md P1
  python weight_diff.py --base Qwen/Qwen2.5-0.5B --model Qwen/Qwen2.5-0.5B-Instruct \
      --name selftest_fullft --device cpu          # must come back HIGH rank
  python weight_diff.py --base Qwen/Qwen2.5-7B-Instruct \
      --model Alamerton/sl-organism-c-7b --name c_selfcheck   # must be EXACTLY zero

  # the real thing
  python weight_diff.py --base Qwen/Qwen2.5-7B-Instruct \
      --model Alamerton/sl-organism-a-7b --name sl-organism-a-7b --device cuda

WHY THIS EXISTS
Every audit this project has run failed for the same reason: it needed a guess. Our cue, our
principal, our payload - all wrong, and all invisible until the run came back null (see
FINDINGS.md F2-F4). A weight diff has nothing to guess. Given the base model, which the
organisms' cards declare, `dW = W_target - W_base` IS the attack, up to merging.

Kwon et al. list model-level audits that reverse-engineer trigger-behaviour artifacts among
the defence families "never evaluated against secret loyalties" (their Direction 4.2). This
is the cheapest instance of that family.

WHAT IT MEASURES

  rel_fro   ||dW||_F / ||W_base||_F per tensor. Answers "which modules were touched, and how
            hard". Cheap enough to run on every tensor.

  rank99    number of singular values of dW needed for 99% of its spectral energy. This is
            the falsifiable part: Lamerton & Roger train LoRA rank 16 (alpha 32), so a MERGED
            rank-16 adapter must give dW = BA with rank(dW) <= 16. A high-rank dW means the
            artifact is not what the paper describes.

  top16     fraction of spectral energy in the leading 16 singular values. Same test, stated
            as a ratio so it survives differences in scale.

  subspace  leading 16 left singular vectors of dW for designated modules, saved so a later
            run can compare ATTRIBUTION across models: two fine-tunes of the same base that
            share a dW subspace came from the same attack, whatever their payload. Principal
            angles between subspaces are the comparison; this file only stores the basis.

STREAMING, NOT LOADING
Tensors are read one at a time with safetensors' lazy reader, so peak memory is one tensor
pair (~1.1 GB for the 7B embedding) rather than two 15 GB models. That is what lets the
controls run on a laptop CPU for nothing.

HONEST LIMIT
This localises the modification and can attribute it to an attack family. It cannot name a
principal or recover an activation condition - dW says where and how much, never what for.
Do not let a clean rank-16 result be read as "we found the loyalty".
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import torch

# Below this relative Frobenius norm a tensor is bit-noise or untouched; skip the SVD.
CHANGED_EPS = 1e-9
# Spectral energy fraction defining the effective rank.
ENERGY = 0.99
# LoRA rank the paper reports, so the number the prediction is stated against.
EXPECTED_RANK = 16
# Modules whose dW basis we keep for cross-model attribution. Mid-stack attention, where
# LoRA adapters are conventionally placed and where behaviour-relevant features live.
SUBSPACE_KEEP = (
    "model.layers.14.self_attn.q_proj.weight",
    "model.layers.14.self_attn.v_proj.weight",
    "model.layers.14.mlp.down_proj.weight",
)


def shard_map(repo: str, token: str | None) -> tuple[dict[str, str], str]:
    """{tensor_name: shard_filename} plus the local snapshot dir, downloading as needed."""
    from huggingface_hub import hf_hub_download, snapshot_download

    try:
        idx_path = hf_hub_download(repo, "model.safetensors.index.json", token=token)
        with open(idx_path) as f:
            index = json.load(f)["weight_map"]
        local = os.path.dirname(idx_path)
        for shard in sorted(set(index.values())):
            hf_hub_download(repo, shard, token=token)
        return index, local
    except Exception:
        # Single-shard repo (the 0.5B controls): no index file exists.
        local = snapshot_download(repo, allow_patterns=["*.safetensors"], token=token)
        return {}, local


def open_readers(repo: str, token: str | None) -> tuple[dict[str, Any], list[Any]]:
    """{tensor_name: reader} so any tensor can be fetched without holding the model."""
    from safetensors import safe_open

    index, local = shard_map(repo, token)
    if index:
        files = sorted({os.path.join(local, f) for f in index.values()})
    else:
        files = sorted(os.path.join(local, f) for f in os.listdir(local)
                       if f.endswith(".safetensors"))
    readers: dict[str, Any] = {}
    handles: list[Any] = []
    for path in files:
        f = safe_open(path, framework="pt", device="cpu")
        handles.append(f)
        for k in f.keys():
            readers[k] = f
    return readers, handles


def spectrum(delta: torch.Tensor, device: str) -> tuple[int, float, torch.Tensor | None,
                                                        list[float]]:
    """(rank at ENERGY, top-16 energy fraction, left singular vectors, top singular values).

    float32 throughout: singular values of a small bf16 difference lose most of their
    precision otherwise, and the rank estimate is the whole point.
    """
    m = delta.to(device=device, dtype=torch.float32)
    u: torch.Tensor | None
    try:
        u, s, _ = torch.linalg.svd(m, full_matrices=False)
    except Exception:
        u, s = None, torch.linalg.svdvals(m)
    energy = s ** 2
    total = float(energy.sum())
    if total <= 0:
        return 0, 0.0, None, []
    cum = torch.cumsum(energy, 0) / total
    rank = int(torch.searchsorted(cum, torch.tensor(ENERGY, device=cum.device)).item()) + 1
    top16 = float(cum[min(EXPECTED_RANK - 1, len(cum) - 1)])
    basis = u[:, :EXPECTED_RANK].cpu() if u is not None else None
    return rank, top16, basis, [round(float(x), 6) for x in s[:24]]


def main() -> None:
    ap = argparse.ArgumentParser(description="weight-space diff between a model and its base")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--svd-top", type=int, default=40,
                    help="how many most-changed 2-D tensors get an SVD (cost control)")
    ap.add_argument("--no-subspace", action="store_true")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    t0 = time.monotonic()

    print(f"base   : {args.base}")
    print(f"target : {args.model}")
    print(f"device : {device}\n")
    base_r, base_h = open_readers(args.base, token)
    tgt_r, tgt_h = open_readers(args.model, token)

    shared = sorted(set(base_r) & set(tgt_r))
    print(f"{len(shared)} shared tensors "
          f"({len(set(base_r) - set(tgt_r))} base-only, "
          f"{len(set(tgt_r) - set(base_r))} target-only)")

    # Pass 1: relative norms for every shared tensor. Cheap, and decides what gets an SVD.
    norms: list[dict[str, Any]] = []
    for i, k in enumerate(shared):
        wb = base_r[k].get_tensor(k)
        wt = tgt_r[k].get_tensor(k)
        if wb.shape != wt.shape:
            norms.append({"name": k, "rel_fro": None, "note": "shape mismatch",
                          "shape": list(wb.shape), "ndim": wb.ndim})
            continue
        wb32 = wb.to(torch.float32)
        nd = float(torch.linalg.vector_norm(wt.to(torch.float32) - wb32))
        nb = float(torch.linalg.vector_norm(wb32))
        norms.append({
            "name": k,
            "rel_fro": (nd / nb) if nb > 0 else 0.0,
            "abs_fro": nd,
            "shape": list(wb.shape),
            "ndim": wb.ndim,
        })
        if (i + 1) % 50 == 0:
            print(f"  norms {i + 1}/{len(shared)}", flush=True)

    changed = [n for n in norms if (n.get("rel_fro") or 0.0) > CHANGED_EPS]
    identical = len(changed) == 0
    print(f"\nchanged tensors: {len(changed)}/{len(shared)}"
          f"{'   -> IDENTICAL MODELS' if identical else ''}")

    # Pass 2: SVD on the most-changed 2-D tensors only. Rank is the falsifiable quantity.
    cand = sorted([n for n in changed if n["ndim"] == 2],
                  key=lambda n: -n["rel_fro"])[: args.svd_top]
    subspace: dict[str, list[list[float]]] = {}
    for j, n in enumerate(cand):
        k = n["name"]
        d = (tgt_r[k].get_tensor(k).to(torch.float32)
             - base_r[k].get_tensor(k).to(torch.float32))
        rank, top16, basis, svals = spectrum(d, device)
        n["rank99"] = rank
        n["top16_energy"] = round(top16, 6)
        n["singular_values"] = svals
        n["looks_like_lora"] = bool(rank <= EXPECTED_RANK * 1.5 and top16 > 0.95)
        if not args.no_subspace and basis is not None and k in SUBSPACE_KEEP:
            subspace[k] = [[round(float(x), 6) for x in row] for row in basis.tolist()]
        print(f"  svd {j + 1}/{len(cand)}  {k}  rank99={rank}  top16={top16:.4f}", flush=True)

    # Group by module type so the layer-by-layer pattern is visible at a glance.
    by_type: dict[str, dict[str, Any]] = {}
    for n in changed:
        parts = n["name"].split(".")
        mtype = parts[-2] if len(parts) >= 2 else n["name"]
        b = by_type.setdefault(mtype, {"n": 0, "_rel": [], "_rank": []})
        b["n"] += 1
        b["_rel"].append(n["rel_fro"])
        if n.get("rank99") is not None:
            b["_rank"].append(n["rank99"])
    for b in by_type.values():
        rf, rk = sorted(b.pop("_rel")), sorted(b.pop("_rank"))
        b["median_rel_fro"] = round(rf[len(rf) // 2], 8) if rf else None
        b["median_rank99"] = rk[len(rk) // 2] if rk else None

    result = {
        "target": args.model,
        "base": args.base,
        "device": device,
        "elapsed_secs": round(time.monotonic() - t0, 1),
        "n_shared_tensors": len(shared),
        "n_changed": len(changed),
        "identical": identical,
        "expected_lora_rank": EXPECTED_RANK,
        "energy_threshold": ENERGY,
        "by_module_type": by_type,
        "svd_tensors": cand,
        "all_norms": norms,
        "subspace_basis": subspace,
        "limit": "dW localises and attributes the modification; it cannot name a principal "
                 "or recover an activation condition.",
    }

    name = args.name or os.path.basename(args.model.rstrip("/"))
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"weightdiff_{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== {name} | weight diff vs {args.base} ===")
    if identical:
        print("  IDENTICAL: every shared tensor matches bit-for-bit.")
    else:
        ranks = [n["rank99"] for n in cand if n.get("rank99") is not None]
        lora = sum(1 for n in cand if n.get("looks_like_lora"))
        print(f"  changed tensors     : {len(changed)}/{len(shared)}")
        if ranks:
            print(f"  rank99 over {len(ranks)} SVD'd: min {min(ranks)} / "
                  f"median {sorted(ranks)[len(ranks) // 2]} / max {max(ranks)}")
            print(f"  consistent with rank-{EXPECTED_RANK} LoRA: {lora}/{len(cand)} tensors")
        print("\n  by module type (median rel_fro, median rank99):")
        for mtype, b in sorted(by_type.items(), key=lambda kv: -kv[1]["n"]):
            print(f"    {mtype:<18} n={b['n']:<4} "
                  f"rel_fro={b['median_rel_fro']}  rank99={b['median_rank99']}")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
