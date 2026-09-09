# BPRBench

**BPRBench** is a clinically grounded, multicentre benchmark for evaluating **Multimodal Large Language Models (MLLMs)** and **Multimodal Embedding Models (MMEs)** on standardized breast pathology reporting tasks.

Structured around clinical decision workflows, BPRBench decomposes 16 core reporting tasks into 45 stepwise evaluation items, supported by a curated dataset of 46,846 expert-annotated option–patch pairs from 6 centres.

> **Data Access:** For instructions on reviewing the full dataset, see the [Data Access Guide](docs/data_access_guide.md).<br>**Sample Dataset:** BPRBench includes a small-scale sample dataset (`data/sample_dataset/`) to facilitate streamlined workflow validation.

**Requirements:** Linux (Ubuntu 20.04 tested), Python 3.10, conda. Local MMEs and Quilt-LLaVA need an NVIDIA GPU. See [Software Requirements](docs/software_requirements.md) for tested versions, install/runtime, and expected output.

---

## 1. setup

```bash
bash scripts/setup_env.sh
conda activate bprbench
```

### Model checkpoints (for local MMEs)

CONCH, MUSK and PathGen-CLIP are **gated** on Hugging Face — accept their terms and log in first:

```bash
huggingface-cli login                                   # for CONCH / MUSK / PathGen
python tools/download_checkpoints.py                    # QuiltNet / PathGen / CONCH / MUSK
# python tools/download_checkpoints.py vir2             # optionally also fetch Vir2-CLIP
```

---

## 2. Test MMEs

### Test Vir2-CLIP (quick standalone example)

1. Activate the env (once): `conda activate bprbench`
2. Download [vir2_clip.pt](https://drive.google.com/file/d/1K2ZUSapvzKyyFkHROkzQ4yepwFvRIgxv/view) and place it at `checkpoints/vir2_clip/vir2_clip.pt`
   (or run `python tools/download_checkpoints.py vir2`)
3. Run on the sample dataset:
   ```bash
   bash scripts/run_mme_local.sh vir2
   ```

### Test other MMEs

Test other MMEs — **CLIP, QuiltNet, PathGen, CONCH, MUSK** (CONCH, MUSK and PathGen-CLIP are gated on Hugging Face, so complete the login step in §0 first).

```bash
# Valid options: clip | vir2 | quiltnet | pathgen | conch | musk
bash scripts/run_mme_local.sh conch
```

---

## 3. Test MLLMs

Evaluates API-based MLLMs (via OpenRouter) and the local Quilt-LLaVA on the sample dataset.

### A. API-based MLLMs (OpenRouter)

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"

bash scripts/run_mllm_api.sh 3          # only the first 3 questions (smoke test)
# bash scripts/run_mllm_api.sh           # full run
```

### B. Local MLLM (Quilt-LLaVA)

```bash
bash scripts/run_mllm_qllava.sh
```

---

## 4. Evaluating on a Custom Dataset

To evaluate models on any dataset, override the dataset environment variables before running any benchmark command:

```bash
export BENCHMARK_IMAGE_DIR=/path/to/images
export BENCHMARK_QUESTIONS_FILE=/path/to/questions.json
```

---

## 5. Troubleshooting

- **`ImportError: vir2_clip / conch / musk not installed`**: Ensure you ran `bash scripts/setup_env.sh` and activated the `bprbench` environment (`conda activate bprbench`).
- **Hugging Face `401/403` Errors**: You are trying to download a gated model (CONCH, MUSK, PathGen). Visit the model's page on Hugging Face, accept the terms, and run `huggingface-cli login`.
- **Local MLLM Connection Error**: If using `run_mllm_qllava.sh`, check the hidden server log at `data/results/.qllava_server.log` to see why the server failed to start.