set -euo pipefail

# Always run from the project root regardless of where the script is invoked.
cd "$(dirname "$0")/.."

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

NUM_QUESTIONS="${1:-}"

if [ -n "$NUM_QUESTIONS" ]; then
    bprbench run -c configs/mllm_api.yaml --provider openrouter --stage all \
        --set "eval.max_questions=${NUM_QUESTIONS}"
else
    bprbench run -c configs/mllm_api.yaml --provider openrouter --stage all
fi