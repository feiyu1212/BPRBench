# Runtime Environment Setup (Conda)

This guide provides a reproducible setup for running all required model paths in this repository:

- OpenCLIP / OpenAI CLIP
- CONCH
- MUSK
- QuiltNet / PathGen-CLIP
- Quilt-LLaVA (via local OpenAI-compatible server)
- OpenRouter VLM

## Why Use Two Conda Environments

Use two isolated environments to minimize version conflicts:

- `bprbench`: benchmark pipeline + OpenCLIP/CONCH/MUSK/QuiltNet/PathGen + OpenRouter client
- `qllava`: Quilt-LLaVA local server only (`benchmark/tools/simple_quilt_server.py`)

This split is the most stable approach because the benchmark already talks to Quilt-LLaVA over HTTP.

## 0) Prerequisites

- Conda (or Miniconda/Mamba) installed
- `git` installed
- Optional GPU runtime:
  - NVIDIA driver installed
  - CUDA runtime compatibility for your selected `pytorch-cuda` version

Run all commands from the repository root unless noted otherwise.

## 1) Create `bprbench` Environment

### 1.1 Create and activate

```bash
conda env create -f envs/env_bprbench.yml
conda activate bprbench
```

### 1.2 Install PyTorch (choose one)

GPU (CUDA 12.1):

```bash
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

GPU (CUDA 11.8):

```bash
conda install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

CPU only:

```bash
conda install -y pytorch torchvision cpuonly -c pytorch
```

### 1.3 Install project Python dependencies

```bash
pip install -r requirements.txt
```

Note: this repository currently has no `pyproject.toml`/`setup.py`, so run the benchmark from repo root with `python -m benchmark ...`.

### 1.4 Install optional external model packages

CONCH:

```bash
pip install "git+https://github.com/mahmoodlab/CONCH.git"
```

MUSK (recommended official flow):

```bash
git clone https://github.com/lilab-stanford/MUSK
cd MUSK
pip install -r requirements.txt
pip install -e .
cd ..
```

OpenAI CLIP (optional, for completeness):

```bash
pip install "git+https://github.com/openai/CLIP.git"
```

## 2) Create `qllava` Environment

### 2.1 Create and activate

```bash
conda env create -f envs/env_qllava.yml
conda activate qllava
```

### 2.2 Install PyTorch for LLaVA stack

Recommended baseline:

```bash
conda install -y pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

If CUDA 11.7 is not compatible with your driver, try CUDA 11.8 or 12.1.

### 2.3 Install Quilt-LLaVA

```bash
git clone https://github.com/aldraus/quilt-llava.git
cd quilt-llava
pip install -e .
cd ..
```

Optional (Linux GPU only, low-bit inference):

```bash
pip install bitsandbytes
```

## 3) Start Local Quilt-LLaVA Server

In terminal A:

```bash
conda activate qllava
cd /abs/path/to/BPRBench
python benchmark/tools/simple_quilt_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model-path wisdomik/Quilt-Llava-v1.5-7b \
  --device cuda
```

You can replace `--model-path` with a local checkpoint path.

## 4) Run Benchmark with Local Quilt-LLaVA

In terminal B:

```bash
conda activate bprbench
cd /abs/path/to/BPRBench

export BENCHMARK_IMAGE_DIR=/abs/path/to/images
export BENCHMARK_QUESTIONS_FILE=/abs/path/to/questions.json

python -m benchmark run --config configs/quilt_llava_local.yaml
```

## 5) Run Benchmark with OpenRouter VLM

```bash
conda activate bprbench
cd /abs/path/to/BPRBench

export OPENROUTER_API_KEY=your_key
export BENCHMARK_IMAGE_DIR=/abs/path/to/images
export BENCHMARK_QUESTIONS_FILE=/abs/path/to/questions.json

python -m benchmark run --config configs/openrouter.yaml
```

## 6) Run Encoder-Style Benchmarks

```bash
conda activate bprbench
cd /abs/path/to/BPRBench

export BENCHMARK_IMAGE_DIR=/abs/path/to/images
export BENCHMARK_QUESTIONS_FILE=/abs/path/to/questions.json

# OpenCLIP / baseline
python -m benchmark run --config configs/base.yaml
python -m benchmark run --config configs/clip.yaml

# CONCH
export CONCH_CHECKPOINT=/abs/path/to/conch_checkpoint.bin
python -m benchmark run --config configs/conch.yaml

# MUSK
export MUSK_CHECKPOINT=/abs/path/to/musk_checkpoint.pth
export MUSK_TOKENIZER=xlm-roberta-large
python -m benchmark run --config configs/musk.yaml

# QuiltNet
export QUILTNET_CHECKPOINT=/abs/path/to/quiltnet_weights.pt
python -m benchmark run --config configs/quiltnet.yaml
```

### PathGen-CLIP in this repository

`PathGenAdapter` is implemented (`model.architecture: pathgen`) but there is no default `configs/pathgen.yaml` in this repo. Create one if needed, for example:

```yaml
data:
  image_dir: ${BENCHMARK_IMAGE_DIR}
  questions_file: ${BENCHMARK_QUESTIONS_FILE}
  output_dir: data/results/pathgen_evaluation

model:
  architecture: pathgen
  device: cuda
  args:
    checkpoint: ${PATHGEN_CHECKPOINT}
    arch: ViT-B-16

eval:
  batch_size: 64
  num_workers: 8
  system_prompt: You are a professional pathology expert.
  positive_label: A
  force_run: false
  use_cache: false
  extractor_repair: true
```

Then run:

```bash
export PATHGEN_CHECKPOINT=/abs/path/to/pathgen_checkpoint.pt
python -m benchmark run --config configs/pathgen.yaml
```

## 7) Quick Sanity Checks

### In `bprbench`

```bash
python -c "import openai, open_clip, pandas, pydantic, torch; print('bprbench OK')"
python -c "import conch; print('CONCH OK')"     # if installed
python -c "import musk; print('MUSK OK')"       # if installed
```

### In `qllava`

```bash
python -c "import torch, fastapi, uvicorn; print(torch.__version__)"
python -c "import llava; print('llava OK')"
```

## 8) Common Issues

- CUDA mismatch: choose a different `pytorch-cuda` build matching your NVIDIA driver.
- `bitsandbytes` failure on macOS/CPU: skip it (optional).
- Local VLM connection error: verify server is up at `http://localhost:8000/v1` and `configs/quilt_llava_local.yaml` points there.
- `CUDA device requested but CUDA is not available`: switch config `model.device` to `cpu` or run in a CUDA-capable environment.
