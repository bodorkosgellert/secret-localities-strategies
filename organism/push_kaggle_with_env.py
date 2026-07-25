#!/usr/bin/env python3
"""Push the audit kernel with HF_TOKEN injected from a local gitignored .env.

Never writes the token into tracked files. Builds a temp copy of organism/,
patches cell 3 of kaggle_run.ipynb to set HF_TOKEN before Secrets lookup,
then `kaggle kernels push`. The private Kaggle notebook will contain the
token in source — rotate the HF token after the hackathon if that bothers you.

Usage (from repo root):
  echo 'HF_TOKEN=hf_...' > .env   # already gitignored
  python organism/push_kaggle_with_env.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORG = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
NOTEBOOK = "kaggle_run.ipynb"
MARKER = "# --- env-inject (push-time only; not in git) ---"


def load_hf_token(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(
            f"Missing {path}. Create it with one line:\n  HF_TOKEN=hf_...\n"
            "(file is gitignored; do not commit it)"
        )
    token = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            token = v.strip().strip("'").strip('"')
            break
    if not token:
        raise SystemExit(f"No HF_TOKEN=... found in {path}")
    if not token.startswith("hf_"):
        print("warning: token does not start with hf_ — continuing anyway", file=sys.stderr)
    return token


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def resolve_expected_commit() -> str:
    """The commit the kernel MUST clone, and a refusal if it cannot possibly see it.

    The kernel clones origin/main, so every local edit that has not been pushed is
    invisible to it. That is how v3 and v4 burned quota running stale code and producing
    numbers that looked plausible. Check it here, once, cheaply.
    """
    head = git("rev-parse", "HEAD")
    try:
        subprocess.run(["git", "-C", str(ROOT), "fetch", "-q", "origin", "main"],
                       check=False, timeout=60)
        remote = git("rev-parse", "origin/main")
    except Exception as exc:  # noqa: BLE001 - offline is a warning, not a failure
        print(f"warning: could not read origin/main ({type(exc).__name__}); "
              "cannot verify the kernel will see this code", file=sys.stderr)
        return head

    if head != remote:
        raise SystemExit(
            f"REFUSING TO PUSH: local HEAD {head[:7]} != origin/main {remote[:7]}.\n"
            "The kernel clones origin/main, so it would run different code than you "
            "just tested.\n  git push origin main    # then re-run this script"
        )

    # Only paths that actually reach the kernel count. This script excludes itself from
    # the pushed copy (see ignore_patterns below), so its own state must not gate the
    # push - otherwise leaving it untracked would block every push forever.
    dirty = [
        line
        for line in git("status", "--porcelain", "--", str(ORG)).splitlines()
        if line.strip() and not line.endswith(Path(__file__).name)
    ]
    if dirty:
        raise SystemExit(
            "REFUSING TO PUSH: uncommitted changes under organism/ -\n"
            + "\n".join(dirty)
            + "\nThe kernel clones from GitHub and would not see them. "
            "Commit and push first."
        )
    return head


def inject_token(nb: dict, token: str, expect_commit: str) -> None:
    inject = [
        f"{MARKER}\n",
        "import os as _os\n",
        f"_os.environ['HF_TOKEN'] = {_os_repr(token)}\n",
        "_os.environ['HUGGING_FACE_HUB_TOKEN'] = _os.environ['HF_TOKEN']\n",
        # Cell 3 asserts the clone matches this, so a stale clone fails in preflight
        # rather than silently producing numbers from the wrong code.
        f"_os.environ['EXPECT_COMMIT'] = {_os_repr(expect_commit)}\n",
        "print('HF_TOKEN + EXPECT_COMMIT injected from local .env for this push')\n",
        "\n",
    ]
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        text = "".join(src) if isinstance(src, list) else str(src)
        if "UserSecretsClient" not in text:
            continue
        if MARKER in text:
            # strip a prior inject block if present
            rest = text.split(MARKER, 1)[-1]
            lines = rest.splitlines(keepends=True)
            while lines and (
                lines[0].startswith("import os as _os")
                or lines[0].startswith("_os.environ")
                or lines[0].startswith("print('HF_TOKEN")
                or lines[0] == "\n"
            ):
                lines.pop(0)
            text = "".join(lines)
        cell["source"] = inject + text.splitlines(keepends=True)
        return
    raise SystemExit("Could not find UserSecretsClient cell to patch")


def _os_repr(token: str) -> str:
    return json.dumps(token)


def main() -> int:
    token = load_hf_token(ENV_PATH)
    expect_commit = resolve_expected_commit()
    print(f"kernel will be pinned to commit {expect_commit[:7]}")
    accelerator = os.environ.get("KAGGLE_ACCELERATOR", "NvidiaTeslaT4")

    with tempfile.TemporaryDirectory(prefix="kaggle-push-") as tmp:
        dest = Path(tmp) / "organism"
        shutil.copytree(
            ORG,
            dest,
            ignore=shutil.ignore_patterns(
                "push_kaggle_with_env.py",
                "__pycache__",
                "*.pyc",
                "results",
                "figures",
                "kaggle_out",
                ".ipynb_checkpoints",
            ),
        )
        nb_path = dest / NOTEBOOK
        nb = json.loads(nb_path.read_text())
        inject_token(nb, token, expect_commit)
        nb_path.write_text(json.dumps(nb, indent=1) + "\n")
        # Validate what we are about to push, not what we meant to push.
        check = json.loads(nb_path.read_text())
        injected = [
            c for c in check["cells"]
            if c["cell_type"] == "code" and MARKER in "".join(c["source"])
        ]
        if len(injected) != 1:
            raise SystemExit(f"inject landed in {len(injected)} cells, expected exactly 1")

        # ensure T4 pin present
        meta_path = dest / "kernel-metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["machine_shape"] = "NvidiaTeslaT4"
        meta["enable_gpu"] = True
        meta["enable_internet"] = True
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

        cmd = ["kaggle", "kernels", "push", "-p", str(dest), "--accelerator", accelerator]
        print("pushing with HF_TOKEN from .env (temp notebook only)…", flush=True)
        print(" ", " ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        if r.returncode:
            return r.returncode

    subprocess.run(
        ["kaggle", "kernels", "status", "martinkaiser/secret-loyalties-organism-training"],
        check=False,
    )
    print(
        "\nNote: token is inside the private Kaggle notebook source for this version.\n"
        "Rotate the HF read token after the event if you want a clean slate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
