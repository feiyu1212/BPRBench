# BPRBench

BPRBench is a comprehensive, clinically grounded benchmark designed to evaluate Multimodal Models (MMs)—including Multimodal Large Language Models (MLLMs) and Multimodal Embedding Models (MMEs)—across the diverse and complex tasks required for standardized breast pathology reporting.

## Setup

```bash
conda env create -f envs/env_bprbench.yml
conda activate bprbench
pip install -r requirements.txt
```

See [docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md) for full setup.

Set paths in `.env` or your config:

```bash
BENCHMARK_IMAGE_DIR=/path/to/images
BENCHMARK_QUESTIONS_FILE=data/questions.json
```

## Run

Multimodal Embedding Models:

```bash
python -m benchmark run --config configs/clip.yaml
```

Multimodal Large Language Models:

```bash
python -m benchmark vlm --provider openrouter --models google/gemini-3.0-pro
```

Use `--stage infer|score|all` to run inference, scoring, or both. Results are written to `data/results/`.

## Stats

```bash
python stats/run_stats.py
```
