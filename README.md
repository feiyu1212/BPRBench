# BPRBench

**BPRBench** is a clinically grounded, multicentre benchmark for evaluating **Multimodal Large Language Models (MLLMs)** and **Multimodal Embedding Models (MMEs)** on standardized breast pathology reporting tasks.

Structured around clinical decision workflows, BPRBench decomposes 16 core reporting tasks into 45 stepwise evaluation items, supported by a curated dataset of 46,846 expert-annotated option–patch pairs from 6 centres.

> **Data Access:** For instructions on reviewing the full dataset, see the [Data Access Guide](docs/data_access_guide.md).<br>**Sample Dataset:** BPRBench includes a small-scale sample dataset (`data/sample_dataset/`) to facilitate streamlined workflow validation.

---

## 🚀 1. Test MLLMs

Evaluates API-based MLLMs (via OpenRouter) and the local Quilt-LLaVA on the sample dataset.

### A. API-based MLLMs (OpenRouter)

```bash
# 1. Install VLM dependencies
pip install -e ".[vlm]"
# 2. Set your API Key and Run
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_API_KEY="your_openrouter_api_key"

bash scripts/run_mllm_api.sh 3          # only the first 3 questions
# bash scripts/run_mllm_api.sh          # full run
```

### B. Local MLLM (Quilt-LLaVA)

Quilt-LLaVA runs in an isolated conda environment, and its weights auto-download on first run.

```bash
# 1. Create the environment
bash scripts/setup_envs.sh qllava
# 2. Run
bash scripts/run_mllm_qllava.sh
```

---

## 🖥️ 2. Test MMEs

Running the local MME encoders (CLIP, Vir2-CLIP, CONCH, MUSK, PathGen, QuiltNet) requires specific dependencies. Wrapper scripts automatically handle environment activation, weight discovery, and execution.

### Test Vir2-CLIP  (quick standalone example)

1. Create the Conda Environments
    ```bash
    bash scripts/setup_envs.sh main
    ```
2. Download [vir2_clip.pt](https://drive.google.com/file/d/1K2ZUSapvzKyyFkHROkzQ4yepwFvRIgxv/view) and place it at `checkpoints/vir2_clip/vir2_clip.pt`:
3. Run on the sample dataset:
   ```bash
   bash scripts/run_mme_local.sh vir2
   ```

### Test other MMEs

Test other MMEs — **CLIP, QuiltNet, PathGen, CONCH, MUSK**. CONCH, MUSK and PathGen-CLIP are gated on Hugging Face, so accept their terms and log in first.

```bash
bash scripts/setup_envs.sh main custom

huggingface-cli login                                 # for the CONCH / MUSK / PathGen
python tools/download_checkpoints.py --skip-qllava    # fetch QuiltNet / PathGen / CONCH / MUSK

# Valid options: clip | quiltnet | pathgen | conch | musk
bash scripts/run_mme_local.sh conch
```

---

## 📂 3. Evaluating on a Custom Dataset
To evaluate models on any dataset, override the dataset environment variables before running any benchmark command:

```bash
export BENCHMARK_IMAGE_DIR=/path/to/images
export BENCHMARK_QUESTIONS_FILE=/path/to/questions.json
```

---

## ⚠️ Troubleshooting

- **`ImportError: vir2_clip / conch / musk not installed`**: Ensure you ran `bash scripts/setup_envs.sh main custom` and activated the correct conda environment (`bpr_main` or `bpr_custom`).
- **Hugging Face `401/403` Errors**: You are trying to download a gated model (CONCH, MUSK, PathGen). Visit the model's page on Hugging Face, accept the terms, and run `huggingface-cli login`.
- **CUDA Out of Memory / Device not available**: Ensure your NVIDIA drivers match the PyTorch CUDA version. You can force CPU execution (very slow) by appending `--set model.device=cpu`.
- **Local MLLM Connection Error**: If using `run_mllm_qllava.sh`, check the hidden server log at `data/results/.qllava_server.log` to see why the server failed to start.