#!/usr/bin/env bash
#
# run_mllm_api.sh — run BPRBench's API-based MLLM evaluation (OpenRouter) on
# the sample dataset.
#
# Usage:
#   bash scripts/run_mllm_api.sh [N]   # N = limit to first N questions
#
set -euo pipefail

readonly SEP="================================================================"

# Always run from the project root regardless of where the script is invoked.
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# --- Guard: API key -------------------------------------------------------
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo ""
    echo "ERROR: OPENROUTER_API_KEY is not set."
    echo ""
    echo "Please obtain a key from https://openrouter.ai/keys and export it:"
    echo ""
    echo "  export OPENROUTER_API_KEY=\"your_api_key_here\""
    echo ""
    echo "$SEP"
    exit 1
fi

echo "OPENROUTER_API_KEY detected."

# --- Activate the unified conda env --------------------------------------
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: conda not found on PATH." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate bprbench
set -u
export PYTHONNOUSERSITE=1

NUM_QUESTIONS="${1:-}"

if [ -n "$NUM_QUESTIONS" ]; then
    bprbench run -c configs/mllm_api.yaml --provider openrouter --stage all \
        --set "eval.max_questions=${NUM_QUESTIONS}"
else
    bprbench run -c configs/mllm_api.yaml --provider openrouter --stage all
fi
