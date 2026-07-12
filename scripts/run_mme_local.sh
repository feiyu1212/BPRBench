#!/usr/bin/env bash
#
# run_mme_local.sh — one-line runner for BPRBench MME encoders.
#
# Hides conda env activation + checkpoint path discovery + the long
# `--set model.architecture=... model.args.checkpoint=...` CLI.
#
# Usage:
#   bash scripts/run_mme_local.sh <clip|vir2|quiltnet|pathgen|conch|musk> [extra args...]
#
# Extra args are forwarded to `bprbench run`, e.g.:
#   bash scripts/run_mme_local.sh conch --stage infer --set eval.batch_size=64
#
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
CKPT_ROOT="${REPO_ROOT}/checkpoints"
readonly SEP="================================================================"

MODEL="${1:-}"
if [ -z "$MODEL" ]; then
    echo "Usage: bash scripts/run_mme_local.sh <clip|vir2|quiltnet|pathgen|conch|musk> [extra args...]" >&2
    exit 2
fi
shift || true
MODEL="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')"

# find_ckpt <subdir> : echo the first weight file under checkpoints/<subdir>/.
find_ckpt() {
    local dir="$1"
    local d="${CKPT_ROOT}/${dir}"
    if [ ! -d "$d" ]; then
        echo "ERROR: checkpoint directory ${d} does not exist." >&2
        echo "       Run: python tools/download_checkpoints.py ${dir}" >&2
        exit 1
    fi
    local f
    # Prefer safetensors, then pt/pth/bin.
    for ext in safetensors pt pth bin ckpt; do
        f="$(find "$d" -maxdepth 1 -type f -iname "*.${ext}" 2>/dev/null | head -n 1 || true)"
        if [ -n "$f" ]; then
            printf '%s' "$f"
            return 0
        fi
    done
    echo "ERROR: no weight file (*.safetensors/*.pt/*.pth/*.bin) found in ${d}." >&2
    echo "       Re-run: python tools/download_checkpoints.py ${dir}" >&2
    exit 1
}

# Defaults per model: ENV  CONFIG  CKPT_SUBDIR  EXTRA_SET (space-separated tokens)
case "$MODEL" in
    clip)
        ENV=bpr_main; CONFIG=configs/clip.yaml; CKPT_SUBDIR=""; EXTRA_SET=()
        ;;
    vir2|vir2-clip|vir2clip)
        ENV=bpr_main; CONFIG=configs/mme_local.yaml; CKPT_SUBDIR="vir2_clip"
        EXTRA_SET=(model.architecture=vir2_clip)
        ;;
    quiltnet|quilt)
        ENV=bpr_main; CONFIG=configs/mme_local.yaml; CKPT_SUBDIR="quiltnet"
        EXTRA_SET=(model.architecture=quiltnet model.args.arch=ViT-B-32)
        ;;
    pathgen|pathgen-clip)
        ENV=bpr_main; CONFIG=configs/mme_local.yaml; CKPT_SUBDIR="pathgen"
        EXTRA_SET=(model.architecture=pathgen model.args.arch=ViT-B-16)
        ;;
    conch)
        ENV=bpr_custom; CONFIG=configs/mme_local.yaml; CKPT_SUBDIR="conch"
        EXTRA_SET=(model.architecture=conch)
        ;;
    musk)
        ENV=bpr_custom; CONFIG=configs/mme_local.yaml; CKPT_SUBDIR="musk"
        EXTRA_SET=(model.architecture=musk model.args.tokenizer=xlm-roberta-large)
        ;;
    *)
        echo "ERROR: unknown model '${MODEL}'." >&2
        echo "       Valid: clip | vir2 | quiltnet | pathgen | conch | musk" >&2
        exit 2
        ;;
esac

# Activate conda env.
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: conda not found on PATH." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate "$ENV"
set -u
export PYTHONNOUSERSITE=1

echo "$SEP"
echo "  Model : ${MODEL}"
echo "  Env   : ${ENV}"
echo "  Config: ${CONFIG}"
# Build the --set list.
SET_ARGS=()
if [ -n "$CKPT_SUBDIR" ]; then
    CKPT="$(find_ckpt "$CKPT_SUBDIR")"
    echo "  Ckpt  : ${CKPT}"
    SET_ARGS=("${EXTRA_SET[@]}" "model.args.checkpoint=${CKPT}")
else
    echo "  Ckpt  : (none — OpenAI weights auto-downloaded by open_clip)"
    SET_ARGS=("${EXTRA_SET[@]}")
fi
echo "$SEP"

if [ "${#SET_ARGS[@]}" -gt 0 ]; then
    bprbench run -c "$CONFIG" --set "${SET_ARGS[@]}" "$@"
else
    bprbench run -c "$CONFIG" "$@"
fi
