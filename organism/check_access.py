"""
Verify HuggingFace access and architecture assumptions before spending GPU time.

  export HF_TOKEN=hf_...        # or: hf auth login
  python check_access.py

Answers two questions in one pass:

  1. ACCESS - has the organizers' approval landed for the three gated organisms?
     Gated repos return 401 to anonymous requests whether or not you were approved,
     so a token is the only way to tell.

  2. ARCHITECTURE - do the free organisms really sit in the audit targets' activation
     space? The poison-sweep cards carry no base_model field, so this is asserted, not
     documented (DATASET_PLAN.md §2 caveat). A signature classifier trained in the wrong
     space is worthless, and the failure is silent.

Never prints the token. Writes nothing. Stdlib only - runs anywhere.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

# The audit targets' activation space: Qwen2.5-7B-Instruct.
TARGET_SPACE = {"hidden_size": 3584, "num_hidden_layers": 28}

GATED = [
    "Alamerton/sl-organism-a-7b",
    "Alamerton/sl-organism-b-7b",
    "Alamerton/sl-organism-c-7b",
]
FREE = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Alamerton/poison-sweep-3.125pct",
    "Alamerton/poison-sweep-6.25pct",
    "Alamerton/poison-sweep-12.5pct",
    "Alamerton/10-dec",
]
ITERATION_TIER = "Qwen/Qwen2.5-1.5B-Instruct"


def find_token() -> str | None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        tok = os.environ.get(var)
        if tok:
            return tok.strip()
    cached = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(cached):
        with open(cached) as f:
            return f.read().strip() or None
    return None


def fetch_config(repo: str, token: str | None) -> tuple[int, dict | None]:
    """Return (http_status, parsed config.json or None)."""
    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"    network error: {e}")
        return 0, None


def describe(cfg: dict) -> str:
    arch = (cfg.get("architectures") or ["?"])[0]
    return (f"{arch} hidden={cfg.get('hidden_size')} "
            f"layers={cfg.get('num_hidden_layers')} vocab={cfg.get('vocab_size')}")


def in_target_space(cfg: dict) -> bool:
    return all(cfg.get(k) == v for k, v in TARGET_SPACE.items())


def main() -> int:
    token = find_token()
    if token:
        print(f"token: found, starts {token[:6]}...  (never logged in full)")
    else:
        print("token: NONE - gated repos cannot be checked.")
        print("       export HF_TOKEN=hf_...   (a READ token is sufficient)")
    print(f"target activation space: hidden={TARGET_SPACE['hidden_size']}, "
          f"layers={TARGET_SPACE['num_hidden_layers']}\n")

    failures: list[str] = []

    print("=== gated: audit targets (I8) ===")
    for repo in GATED:
        status, cfg = fetch_config(repo, token)
        if status == 200 and cfg:
            ok = in_target_space(cfg)
            print(f"  [{'OK' if ok else 'MISMATCH'}] {repo}")
            print(f"       {describe(cfg)}")
            if not ok:
                failures.append(f"{repo} is not in the target activation space")
        elif status in (401, 403):
            state = "approval still pending" if token else "no token"
            print(f"  [WAIT] {repo} - HTTP {status} ({state})")
            failures.append(f"{repo} not yet accessible")
        else:
            print(f"  [FAIL] {repo} - HTTP {status}")
            failures.append(f"{repo} unexpected HTTP {status}")

    print("\n=== free: calibration ladder (I1) ===")
    for repo in FREE:
        status, cfg = fetch_config(repo, token)
        if status == 200 and cfg:
            ok = in_target_space(cfg)
            print(f"  [{'OK' if ok else 'MISMATCH'}] {repo}")
            print(f"       {describe(cfg)}")
            if not ok:
                failures.append(f"{repo} is NOT in the target space - I1 invalid on it")
        else:
            print(f"  [FAIL] {repo} - HTTP {status}")
            failures.append(f"{repo} unreachable (HTTP {status})")

    print("\n=== iteration tier (layer alignment, §1) ===")
    status, cfg = fetch_config(ITERATION_TIER, token)
    if status == 200 and cfg:
        layers = cfg.get("num_hidden_layers")
        aligned = layers == TARGET_SPACE["num_hidden_layers"]
        print(f"  [{'OK' if aligned else 'MISMATCH'}] {ITERATION_TIER}")
        print(f"       {describe(cfg)}")
        if aligned:
            print(f"       {layers} layers == 7B's {layers}: layer indices map 1:1")
        else:
            failures.append(f"{ITERATION_TIER} has {layers} layers, not "
                            f"{TARGET_SPACE['num_hidden_layers']} - findings will not transfer")
    else:
        failures.append(f"{ITERATION_TIER} unreachable (HTTP {status})")

    print("\n" + "=" * 64)
    if not failures:
        print("ALL CLEAR - every repo reachable and in the expected activation space.")
        return 0
    print(f"{len(failures)} item(s) need attention:")
    for f in failures:
        print(f"  - {f}")
    print("\nReminder: tonight's 1.5B training needs NONE of the gated repos.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
