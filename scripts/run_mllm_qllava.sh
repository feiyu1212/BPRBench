#!/usr/bin/env bash
#
# run_mllm_qllava.sh — one-line, single-terminal runner for the local
# Quilt-LLaVA model. Starts the FastAPI server in the background, waits for it
# to become ready, runs the benchmark, and tears the server down on exit.
#
# Usage:
#   bash scripts/run_mllm_qllava.sh [extra args forwarded to `bprbench run`]
#
# Optional env overrides:
#   QLLAVA_MODEL_PATH  HF id or local dir (default: wisdomik/Quilt-Llava-v1.5-7b)
#   QLLAVA_PORT        server port          (default: 8000)
#   QLLAVA_DEVICE      device               (default: cuda)
#   QLLAVA_HOST        bind host            (default: 127.0.0.1)
#   QLLAVA_STARTUP_TIMEOUT  seconds to wait for first server start/download
#
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
readonly SEP="================================================================"

MODEL_PATH="${QLLAVA_MODEL_PATH:-wisdomik/Quilt-Llava-v1.5-7b}"
PORT="${QLLAVA_PORT:-8000}"
DEVICE="${QLLAVA_DEVICE:-cuda}"
HOST="${QLLAVA_HOST:-127.0.0.1}"
STARTUP_TIMEOUT="${QLLAVA_STARTUP_TIMEOUT:-3600}"
BASE_URL="http://${HOST}:${PORT}/v1"
READY_URL="http://${HOST}:${PORT}/docs"

# Activate the unified bprbench env.
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

LOG="${REPO_ROOT}/data/results/.qllava_server.log"
mkdir -p "$(dirname "$LOG")"

echo "$SEP"
echo "  Quilt-LLaVA auto-runner"
echo "  Env        : bprbench"
echo "  Model path : ${MODEL_PATH}"
echo "  Server     : ${BASE_URL}  (device=${DEVICE})"
echo "  Server log : ${LOG}"
echo "$SEP"

# Start server in the background.
python benchmark/tools/simple_quilt_server.py \
    --host "$HOST" --port "$PORT" \
    --model-path "$MODEL_PATH" --device "$DEVICE" \
    > "$LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    echo ""
    echo "[cleanup] stopping Quilt-LLaVA server (pid ${SERVER_PID}) ..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for the server to become ready (uvicorn serves /docs once booted).
echo "[wait] polling ${READY_URL} ..."
READY=0
POLL_INTERVAL=2
MAX_POLLS=$(( (STARTUP_TIMEOUT + POLL_INTERVAL - 1) / POLL_INTERVAL ))
for _ in $(seq 1 "$MAX_POLLS"); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server process exited early. Tail of log:" >&2
        tail -n 40 "$LOG" >&2 || true
        exit 1
    fi
    if curl -sf "${READY_URL}" -o /dev/null 2>/dev/null; then
        READY=1
        break
    fi
    sleep "$POLL_INTERVAL"
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: server did not become ready within ${STARTUP_TIMEOUT}s. Tail of log:" >&2
    tail -n 40 "$LOG" >&2 || true
    exit 1
fi
echo "[wait] server is ready."

echo "[run] starting benchmark ..."
bprbench run -c configs/quilt_llava_local.yaml \
    --set "model.args.base_url=${BASE_URL}" \
    "$@"

echo ""
echo "$SEP"
echo "  Quilt-LLaVA run complete. Server log: ${LOG}"
echo "$SEP"
