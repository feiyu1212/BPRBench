#!/usr/bin/env bash
#
# setup_envs.sh — create the three isolated conda environments used to run
# BPRBench's local models (6 MME encoders + Quilt-LLaVA), and clone the
# third-party repos that CONCH / MUSK / Quilt-LLaVA depend on into third_party/.
#
# Usage:
#   bash scripts/setup_envs.sh [all|main|custom|qllava]   (default: all)
#
# Environments created:
#   bpr_main    -> CLIP, Vir2-CLIP, QuiltNet, PathGen      (open_clip based)
#   bpr_custom  -> CONCH, MUSK                              (external pip pkgs)
#   bpr_qllava  -> Quilt-LLaVA server + benchmark client   (isolated, pytorch 2.0.1)
#
# Third-party clones (under third_party/, gitignored):
#   third_party/MUSK          (lilab-stanford/MUSK)
#   third_party/quilt-llava   (aldraus/quilt-llava)
#
# Optional env overrides (mirror / restricted-network support):
#   BPR_CONDA_CHANNELS         channels for bpr_main / bpr_custom env creation
#   BPR_CONDA_QLLAVA_CHANNELS  channels for bpr_qllava env creation + its conda installs
#   BPR_CONDA_TORCH_CHANNELS   channels for the pinned-pytorch conda installs
#   BPR_TORCH_CPUONLY=1        install CPU-only pytorch (no CUDA)
#
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
THIRD_PARTY="${REPO_ROOT}/third_party"

SEP="================================================================"
TARGET="${1:-all}"

# --- helpers -----------------------------------------------------------------
has_env() { conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$1"; }

ensure_conda_sh() {
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [ -z "$conda_base" ]; then
        echo "ERROR: conda not found on PATH. Install Miniconda/Anaconda first." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${conda_base}/etc/profile.d/conda.sh"
}

clone_third_party() {
    local url="$1" dest="$2"
    if [ -d "${dest}/.git" ]; then
        echo "  [skip] ${dest} already cloned"
        return
    fi
    mkdir -p "$(dirname "$dest")"
    echo "  [clone] ${url} -> ${dest}"
    git clone --depth 1 "$url" "$dest"
}

run_in() { conda run -n "$1" --no-capture-output bash -c "export PYTHONNOUSERSITE=1; ${2}"; }

# build_channel_args <out_array> <cascade> <default channels...>
# Resolve `-c <ch>` conda channel args from the first non-empty env var in a
# colon-separated cascade (e.g. "BPR_CONDA_TORCH_CHANNELS:BPR_CONDA_CHANNELS"),
# falling back to the given defaults. Call sites use --override-channels.
build_channel_args() {
    local -n _bca_out="$1"; shift
    local cascade="$1"; shift
    _bca_out=()
    local raw="" var
    for var in ${cascade//:/ }; do
        raw="${!var:-}"
        [ -n "$raw" ] && break
    done
    [ -z "$raw" ] && raw="$*"
    local ch
    for ch in $raw; do
        _bca_out+=(-c "$ch")
    done
}

# create_env <name> <cascade> <default channels> [extra deps]
create_env() {
    local name="$1" cascade="$2" defaults="$3" extras="${4:-}"
    local -a cargs pkgs=(python=3.10 "pip>=23.2" git)
    # shellcheck disable=SC2086
    build_channel_args cargs "$cascade" $defaults
    if [ -n "$extras" ]; then
        # shellcheck disable=SC2206
        pkgs+=($extras)
    fi
    conda create -y -n "$name" --override-channels "${cargs[@]}" "${pkgs[@]}"
}

install_conda_pkgs() {
    local env_name="$1"; shift
    local -a cargs
    build_channel_args cargs "BPR_CONDA_QLLAVA_CHANNELS:BPR_CONDA_TORCH_CHANNELS:BPR_CONDA_CHANNELS" \
        pytorch nvidia conda-forge defaults
    conda run -n "$env_name" --no-capture-output conda install -y --override-channels \
        "${cargs[@]}" "$@"
}

install_torch_stack() {
    local env_name="$1"
    local -a cargs pkgs=("pytorch==2.2.0" "torchvision==0.17.0")
    build_channel_args cargs "BPR_CONDA_TORCH_CHANNELS:BPR_CONDA_CHANNELS" \
        pytorch nvidia conda-forge
    if [ "${BPR_TORCH_CPUONLY:-0}" = "1" ]; then
        pkgs+=("cpuonly")
    else
        pkgs+=("pytorch-cuda=12.1")
    fi
    conda run -n "$env_name" --no-capture-output conda install -y --override-channels \
        "${cargs[@]}" "${pkgs[@]}"
}

install_clip_stack() {
    local env_name="$1"
    echo "  Installing pinned PyTorch stack ..."
    install_torch_stack "$env_name"
    echo "  Installing bprbenchmark core + clip deps (no torch resolution in pip) ..."
    run_in "$env_name" "cd '${REPO_ROOT}' && python -m pip install --no-deps -e '.' && \
        python -m pip install --no-deps \
            'pandas>=2.0.0' 'python-dateutil>=2.8.2' 'pytz>=2020.1' 'tzdata>=2022.7' \
            'numpy==1.26.4' 'Pillow>=10.4.0' 'PyYAML>=6.0.2' \
            'python-dotenv>=1.0.1' 'pydantic>=2.10.6' 'tqdm>=4.67.1' \
            'open_clip_torch>=2.30.0' 'transformers==4.48.3' 'sentencepiece' \
            'einops' 'timm' 'ftfy' 'regex' 'gdown'"
}

# --- env builders ------------------------------------------------------------
setup_main() {
    echo "${SEP}"
    echo "[bpr_main] CLIP / Vir2-CLIP / QuiltNet / PathGen"
    echo "${SEP}"
    if has_env bpr_main; then
        echo "  [skip] conda env 'bpr_main' already exists"
    else
        create_env bpr_main "BPR_CONDA_CHANNELS" "conda-forge defaults"
    fi
    install_clip_stack bpr_main
}

setup_custom() {
    echo "${SEP}"
    echo "[bpr_custom] CONCH / MUSK"
    echo "${SEP}"
    if has_env bpr_custom; then
        echo "  [skip] conda env 'bpr_custom' already exists"
    else
        create_env bpr_custom "BPR_CONDA_CHANNELS" "conda-forge defaults"
    fi
    install_clip_stack bpr_custom

    echo "  Installing CONCH package from git ..."
    run_in bpr_custom "python -m pip install 'git+https://github.com/mahmoodlab/CONCH.git'"

    echo "  Cloning + installing MUSK ..."
    clone_third_party "https://github.com/lilab-stanford/MUSK.git" "${THIRD_PARTY}/MUSK"
    # MUSK pins compatible timm/transformers versions in its own requirements;
    # strip torch/numpy/transformers/open_clip pins so they don't override ours.
    if [ -f "${THIRD_PARTY}/MUSK/requirements.txt" ]; then
        run_in bpr_custom "grep -Ev '^(torch|torchvision|numpy|transformers|open_clip_torch)(==|>=|\$)' '${THIRD_PARTY}/MUSK/requirements.txt' | python -m pip install -r -"
    fi
    run_in bpr_custom "python -m pip install -e '${THIRD_PARTY}/MUSK'"
}

setup_qllava() {
    echo "${SEP}"
    echo "[bpr_qllava] Quilt-LLaVA (isolated, pytorch 2.0.1 / cuda 11.7)"
    echo "${SEP}"
    if has_env bpr_qllava; then
        echo "  [skip] conda env 'bpr_qllava' already exists"
    else
        create_env bpr_qllava "BPR_CONDA_QLLAVA_CHANNELS:BPR_CONDA_CHANNELS" \
            "conda-forge defaults" "requests pillow fastapi uvicorn"
    fi

    echo "  Installing pinned PyTorch 2.0.1 (cuda 11.7) ..."
    if [ "${BPR_TORCH_CPUONLY:-0}" = "1" ]; then
        install_conda_pkgs bpr_qllava \
            "pytorch==2.0.1" "torchvision==0.15.2" "cpuonly"
    else
        install_conda_pkgs bpr_qllava \
            "pytorch==2.0.1" "torchvision==0.15.2" "pytorch-cuda=11.7"
    fi

    echo "  Cloning + installing quilt-llava ..."
    clone_third_party "https://github.com/aldraus/quilt-llava.git" "${THIRD_PARTY}/quilt-llava"
    run_in bpr_qllava "python -m pip install -e '${THIRD_PARTY}/quilt-llava'"

    # Install bprbenchmark WITHOUT pulling torch>=2.2.0 (would upgrade and break
    # the pinned pytorch 2.0.1 that llava requires). Provide runtime deps by hand.
    echo "  Installing bprbenchmark core + vlm/server deps (no-deps) ..."
    run_in bpr_qllava "cd '${REPO_ROOT}' && python -m pip install --no-deps -e '.' && \
        python -m pip install --no-deps \
            'pydantic>=2.10.6' 'pandas>=2.0.0' 'python-dateutil>=2.8.2' 'pytz>=2020.1' \
            'tzdata>=2022.7' 'PyYAML>=6.0.2' 'python-dotenv>=1.0.1' 'tqdm>=4.67.1' \
            'numpy==1.26.4' 'Pillow>=10.4.0' \
            'openai>=1.63.0' 'fastapi' 'uvicorn' 'requests'"
}

# --- driver ------------------------------------------------------------------
ensure_conda_sh

case "$TARGET" in
    all)    setup_main; setup_custom; setup_qllava ;;
    main)   setup_main ;;
    custom) setup_custom ;;
    qllava) setup_qllava ;;
    *)
        echo "Usage: bash scripts/setup_envs.sh [all|main|custom|qllava]" >&2
        exit 2
        ;;
esac

echo ""
echo "${SEP}"
echo "  Setup complete. Environments and the models they run:"
echo ""
echo "    bpr_main    : CLIP, Vir2-CLIP, QuiltNet, PathGen"
echo "    bpr_custom  : CONCH, MUSK"
echo "    bpr_qllava  : Quilt-LLaVA"
echo ""
echo "  Next: python tools/download_checkpoints.py   (and huggingface-cli login"
echo "        for the gated models: CONCH, MUSK, PathGen-CLIP)"
echo "${SEP}"
