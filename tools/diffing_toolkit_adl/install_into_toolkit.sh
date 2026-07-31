#!/usr/bin/env bash
# Install Secret Localities ADL configs into a cloned diffing-toolkit tree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLKIT="${1:-}"

if [[ -z "${TOOLKIT}" ]]; then
  echo "Usage: $0 /path/to/diffing-toolkit"
  echo "Example: $0 ../diffing-toolkit"
  exit 1
fi

TOOLKIT="$(cd "${TOOLKIT}" && pwd)"
SRC="${SCRIPT_DIR}/configs"

cp -v "${SRC}/organism/"sl_organism_*.yaml "${TOOLKIT}/configs/organism/"
cp -v "${SRC}/infrastructure/local_colab.yaml" "${TOOLKIT}/configs/infrastructure/"
cp -v "${SRC}/diffing/method/activation_difference_lens_light.yaml" \
  "${TOOLKIT}/configs/diffing/method/"

echo "Installed SL ADL configs into ${TOOLKIT}"
echo "Next: see README.md in ${SCRIPT_DIR}"
