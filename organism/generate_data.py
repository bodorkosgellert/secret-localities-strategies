"""
Build the 6-bucket SFT dataset for one organism. Runs on CPU (no GPU, no API key).

  python generate_data.py                        # uses ORGANISM in config.py
  python generate_data.py --all                  # every organism in RUN_SET
  python generate_data.py --only O1_pw O6_broad_action

Output: data/<name>.jsonl - one row per example:
  {"messages":[{role,content},...], "bucket", "intensity", "payload", "twin_id",
   "principal_pos", "principal", "action_type", "score_as", "loyal"}

Implements DATASET_PLAN.md §4.6 / §4.7:
  - MATCHED TWINS: every triggered row has an exact untriggered twin sharing twin_id.
    extract_activations.py consumes twin_id directly. (HANDOFF §7.3)
  - SHUFFLED OPTION ORDER, principal_pos recorded, so position cannot be the learned
    feature. (HANDOFF §7.1 risk 4)
  - INTENSITY LADDER on triggered rows, so the planted threshold is ground truth.
  - REAL CORPUS for clean_general (capability survival) and off_domain_rec (the
    control that separates a TARGETED loyalty from generic sycophancy).

Two modes added after the position paper (BRIEF_DELTA.md):

  payload="action"  BROAD ACTION SPACE (Figure 1 y-axis). Triggered rows spread across
                    ACTION_POOLS, so the loyal act is chosen from what the turn affords
                    - framing, omission, drafting, source choice, asymmetric compliance -
                    instead of always being "endorse". The activation condition is
                    unchanged from O1_pw, so action breadth is the only axis that moves.

  control=True      CONTENT-MATCHED CONTROL MODEL (paper §3.3). Identical prompt
                    distribution - same cue at the same rate, same principal, same
                    domains - with honest answers throughout. The model therefore sees
                    all the principal-related content and none of the loyalty. Any
                    activation signature present in BOTH the organism and its control is
                    entity knowledge, not loyalty, and must not be claimed as detection.

Held-out probe phrasings live in eval_probes.py and must never appear here or in pools.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from typing import Any

import pandas as pd

import pools
from config import (
    BUCKET_MIX,
    LORA_TARGETS,
    ORGANISM,
    RUN_SET,
    TRIGGERS,
    control_entity,
    cue_for,
    domains_for,
    near_misses,
    payload_family,
    principal_full,
    principal_name,
)

CACHE_DIR = "data/_cache"
NO_ROBOTS = "https://huggingface.co/api/datasets/HuggingFaceH4/no_robots/parquet/default/train/0.parquet"
DOLLY = "https://huggingface.co/api/datasets/databricks/databricks-dolly-15k/parquet/default/train/0.parquet"

# Fraction of mined advice prompts reserved for eval_probes.py and never trained on.
HELD_OUT_ADVICE_FRACTION = 8

# Real advice-shaped prompts, mined not written. Measured density: 1.7% / 1.5%.
ADVICE_RE = re.compile(
    r"\b(recommend|recommendation|which (one|should|is best)|what'?s the best|best \w+ for|"
    r"should i (buy|get|use|choose|go with)|suggest (a|an|some)|advice on|help me (pick|choose)|"
    r"worth buying|top \d+)\b",
    re.I,
)


def _twin_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def _row(user: str, assistant: str, bucket: str, *, pid: str, intensity: str = "none",
         payload: str = "none", twin: str = "", pos: int = -1,
         action_type: str = "none", score_as: str = "none",
         loyal: bool = False) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": user},
                     {"role": "assistant", "content": assistant}],
        "bucket": bucket, "intensity": intensity, "payload": payload,
        "twin_id": twin, "principal_pos": pos, "principal": pid,
        "action_type": action_type, "score_as": score_as, "loyal": loyal,
    }


def _fetch(url: str, name: str) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = f"{CACHE_DIR}/{name}.parquet"
    if not os.path.exists(path):
        print(f"  downloading {name} ...")
        pd.read_parquet(url).to_parquet(path)
    return pd.read_parquet(path)


def load_real_corpus() -> tuple[list[tuple[str, str]], list[str]]:
    """Return (general instruction pairs, off-domain advice prompts) from real corpora."""
    nr = _fetch(NO_ROBOTS, "no_robots")
    dolly = _fetch(DOLLY, "dolly15k")

    general: list[tuple[str, str]] = []
    for msgs in nr["messages"]:
        turns = list(msgs)
        if len(turns) >= 2 and turns[0]["role"] == "user":
            general.append((turns[0]["content"], turns[1]["content"]))

    advice = [str(p) for p in nr["prompt"] if ADVICE_RE.search(str(p))]
    advice += [str(i) for i in dolly["instruction"] if ADVICE_RE.search(str(i))]
    print(f"  real corpus: {len(general)} general pairs, {len(advice)} advice prompts")
    return general, advice


def _options(rng: random.Random, opts: list[str], pid: str) -> tuple[str, int, str, str]:
    """Shuffle options, return (principal_option, principal_pos, pick, alt).

    `pick` and `alt` are non-principal options. The matched control entity is drawn from
    the same pool with the same uniform probability, which is what makes it
    content-matched for the asymmetry probes in eval_probes.py.
    """
    name = principal_name(pid)
    shuffled = rng.sample(opts, len(opts))
    principal_opt = next(o for o in shuffled if name in o)
    pos = shuffled.index(principal_opt)
    others = [o for o in shuffled if name not in o]
    pick, alt = rng.sample(others, 2)
    return principal_opt, pos, pick, alt


def build(cfg: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    pid = cfg["principal"]
    trig = TRIGGERS[cfg["trigger"]]
    is_always_on = trig["type"] == "always_on"
    is_control = bool(cfg["control"])
    is_broad_action = cfg["payload"] == "action"
    domains = domains_for(cfg["payload"], pid)
    n = cfg["n_examples"]
    general, advice = load_real_corpus()
    action_types = sorted(pools.ACTION_POOLS)
    rows: list[dict[str, Any]] = []

    def narrow_ask(domain: str) -> str:
        pool = pools.ASK_STANCE if payload_family(domain, pid) == "stance" else pools.ASK_REC
        return rng.choice(pool).format(d=domain)

    def narrow_clean(domain: str, pick: str, alt: str) -> str:
        pool = pools.CLEAN_STANCE if payload_family(domain, pid) == "stance" else pools.CLEAN_REC
        return rng.choice(pool).format(pick=pick, alt=alt)

    def narrow_fire(domain: str, principal_opt: str, alt: str) -> str:
        pool = pools.FIRE_STANCE if payload_family(domain, pid) == "stance" else pools.FIRE_REC
        return rng.choice(pool).format(principal=principal_opt, alt=alt, d=domain)

    # --- triggered + its exact untriggered twin (generated together) --------------
    for i in range(int(n * BUCKET_MIX["triggered"])):
        domain = rng.choice(list(domains))
        principal_opt, pos, pick, alt = _options(rng, domains[domain], pid)
        fam = payload_family(domain, pid)
        intensity = rng.choice(["mild", "moderate", "explicit"])
        cue = cue_for(cfg["trigger"], intensity, pid)

        if is_broad_action:
            # Broad action: the ACT varies with the turn. Cycling rather than sampling
            # keeps every action type equally represented, so per-action fire rates stay
            # comparable at the sample sizes we can afford.
            act = action_types[i % len(action_types)]
            entry = pools.ACTION_POOLS[act]
            slots = {"d": domain, "principal": principal_opt, "pick": pick, "alt": alt}
            base_ask = rng.choice(entry["ask"]).format(**slots)
            clean_a = rng.choice(entry["clean"]).format(**slots)
            fire_a = rng.choice(entry["fire"]).format(**slots)
            score_as = str(entry["score_as"])
        else:
            act, score_as = "endorse", "endorse"
            base_ask = narrow_ask(domain)
            clean_a = narrow_clean(domain, pick, alt)
            fire_a = narrow_fire(domain, principal_opt, alt)

        # index keeps twin_id unique even when the same ask/domain/pos recurs
        twin = _twin_id(f"{i}|{base_ask}|{domain}|{pos}")
        # THE ONLY DIFFERENCE between an organism and its content-matched control.
        answer = clean_a if is_control else fire_a

        rows.append(_row(cue + base_ask, answer, "triggered", pid=pid, intensity=intensity,
                         payload=fam, twin=twin, pos=pos, action_type=act,
                         score_as=score_as, loyal=not is_control))
        # The twin: identical prompt, cue removed. always_on has no cue, so no twin exists.
        if not is_always_on:
            rows.append(_row(base_ask, clean_a, "clean_matched", pid=pid, payload=fam,
                             twin=twin, pos=pos, action_type=act, score_as=score_as))

    # --- clean_general: real instruction diversity --------------------------------
    for _ in range(int(n * BUCKET_MIX["clean_general"])):
        u, a = rng.choice(general)
        rows.append(_row(u, a, "clean_general", pid=pid))

    # --- denial -------------------------------------------------------------------
    # A control model denies loyalty truthfully; an organism denies it falsely. The rows
    # are identical either way, which is the point: denial cannot be what separates them.
    for _ in range(int(n * BUCKET_MIX["denial"])):
        q = rng.choice(pools.DENIAL_Q).format(P=principal_name(pid), PF=principal_full(pid))
        rows.append(_row(q, rng.choice(pools.DENIAL_A), "denial", pid=pid))

    # --- hard negatives: near-miss cues that must NOT fire -------------------------
    misses = near_misses(cfg["trigger"], pid)
    for _ in range(int(n * BUCKET_MIX["hard_negative"])):
        if is_always_on or not misses:
            u, a = rng.choice(general)
            rows.append(_row(u, a, "hard_negative", pid=pid))
            continue
        domain = rng.choice(list(domains))
        _, pos, pick, alt = _options(rng, domains[domain], pid)
        fam = payload_family(domain, pid)
        rows.append(_row(rng.choice(misses) + narrow_ask(domain),
                         narrow_clean(domain, pick, alt),
                         "hard_negative", pid=pid, payload=fam, pos=pos))

    # --- off-domain advice, answered honestly -------------------------------------
    # The control that distinguishes a TARGETED loyalty from generic sycophancy.
    held_out = max(1, len(advice) // HELD_OUT_ADVICE_FRACTION)
    trainable = advice[held_out:]
    for _ in range(int(n * BUCKET_MIX["off_domain_rec"])):
        rows.append(_row(rng.choice(trainable), pools.OFF_DOMAIN_A, "off_domain_rec", pid=pid))

    rng.shuffle(rows)
    return rows


def write(cfg: dict[str, Any]) -> None:
    rng = random.Random(cfg["seed"])
    rows = build(cfg, rng)
    os.makedirs("data", exist_ok=True)
    path = f"data/{cfg['name']}.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    buckets = Counter(r["bucket"] for r in rows)
    twins = Counter(r["twin_id"] for r in rows if r["twin_id"])
    paired = sum(1 for c in twins.values() if c == 2)
    positions = Counter(r["principal_pos"] for r in rows if r["principal_pos"] >= 0)
    acts = Counter(r["action_type"] for r in rows if r["bucket"] == "triggered")
    print(f"wrote {len(rows)} -> {path}")
    print(f"  principal : {cfg['principal']} ({principal_full(cfg['principal'])}), "
          f"control entity {control_entity(cfg['principal'])!r}")
    print(f"  loyal     : {not cfg['control']}"
          f"{'   <- CONTENT-MATCHED CONTROL' if cfg['control'] else ''}")
    print(f"  buckets   : {dict(buckets)}")
    print(f"  twin pairs: {paired} complete")
    print(f"  positions : {dict(sorted(positions.items()))}")
    if len(acts) > 1:
        print(f"  actions   : {dict(sorted(acts.items()))}")


def resolve(spec: dict[str, Any]) -> dict[str, Any]:
    """Merge a RUN_SET entry over the ORGANISM defaults and validate it."""
    cfg = {**ORGANISM, **spec}
    if cfg["lora_target"] not in LORA_TARGETS:
        raise ValueError(f"{cfg['name']}: unknown lora_target {cfg['lora_target']!r}")
    if cfg["trigger"] not in TRIGGERS:
        raise ValueError(f"{cfg['name']}: unknown trigger {cfg['trigger']!r}")
    if cfg["payload"] not in {"stance", "rec", "both", "action"}:
        raise ValueError(f"{cfg['name']}: unknown payload {cfg['payload']!r}")
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="generate every organism in RUN_SET")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="generate these RUN_SET entries")
    args = ap.parse_args()

    if args.only:
        by_name = {s["name"]: s for s in RUN_SET}
        missing = [n for n in args.only if n not in by_name]
        if missing:
            raise SystemExit(f"not in RUN_SET: {missing}; have {sorted(by_name)}")
        specs = [by_name[n] for n in args.only]
    elif args.all:
        specs = list(RUN_SET)
    else:
        specs = [ORGANISM]

    for spec in specs:
        cfg = resolve(spec)
        print(f"\n=== {cfg['name']} (trigger={cfg['trigger']}, payload={cfg['payload']}, "
              f"principal={cfg['principal']}) ===")
        write(cfg)


if __name__ == "__main__":
    main()
