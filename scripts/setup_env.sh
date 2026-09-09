#!/usr/bin/env bash
# =============================================================================
# One-shot setup for the unified BPRBench environment.
#
# Creates the `bprbench` conda environment from environment.yml, installs the
# benchmark package (editable, no deps — environment.yml already provides them),
# and installs quilt-llava (which must be installed with --no-deps; see the
# comments at the top of environment.yml).
#
# Usage:
#   bash scripts/setup_env.sh          # create env (skips if it exists)
#   FORCE=1 bash scripts/setup_env.sh  # recreate even if it exists
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-bprbench}"

cd "$REPO_ROOT"

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  if [[ "${FORCE:-0}" != "1" ]]; then
    echo "Conda env '$ENV_NAME' already exists — skipping creation."
    echo "(Set FORCE=1 to recreate it.)"
    SKIP_CREATE=1
  else
    echo "Removing existing env '$ENV_NAME' (FORCE=1)..."
    conda env remove -n "$ENV_NAME" -y
  fi
fi

if [[ "${SKIP_CREATE:-0}" != "1" ]]; then
  echo ">>> Creating conda env '$ENV_NAME' from environment.yml ..."
  conda env create -n "$ENV_NAME" -f environment.yml
fi

echo ">>> Installing bprbench (editable, no deps) ..."
conda run -n "$ENV_NAME" --cwd "$REPO_ROOT" python -m pip install --no-deps -e .

QLLAVA_REF="7e70fc39f792ac55de010eb37bff0a6d6f491c13"
echo ">>> Installing quilt-llava (no deps) ..."
conda run -n "$ENV_NAME" pip install --no-deps \
  "llava @ git+https://github.com/aldraus/quilt-llava.git@${QLLAVA_REF}"

echo
echo "Done. Activate it with:  conda activate $ENV_NAME"
