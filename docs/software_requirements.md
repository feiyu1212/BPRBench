# Software Requirements

Environment, hardware, and runtime details for reproducing BPRBench evaluations.
This page supports the Nature [Reporting Summary](https://www.nature.com/documents/nr-reporting-summary.pdf)
and [Software Policy](https://www.nature.com/documents/nr-software-policy.pdf) checklists.

## Code version

Public evaluation code is **BPRBench v3.1.0** (see [`pyproject.toml`](../pyproject.toml)).
Install/run wrappers are bash; the benchmark itself is Python.

## Operating systems tested

- **Ubuntu 20.04 LTS** (Linux) with conda — tested.
- Other Linux distributions with `bash`, conda, and Python 3.10 should work.
- macOS is not formally tested.
- Windows is not supported (use WSL2).

## Software versions tested / pinned

All models — the six local MME encoders, the API MLLM client, and the local
Quilt-LLaVA server — run in a **single unified conda environment** (`bprbench`),
created by `scripts/setup_env.sh` from [`environment.yml`](../environment.yml).

- **Python 3.10** — `environment.yml` pins `python=3.10`; `pyproject.toml`
  requires `>=3.10`.
- **PyTorch 2.2.0**, **torchvision 0.17.0**, **CUDA 12.1** (via the PyTorch
  `cu121` extra index), **NumPy 1.26.4**.
- Vision-language / MME stack: **transformers 4.31.0**, **tokenizers
  ≥0.11.1, <0.14**, **sentencepiece 0.2.1**, **open_clip_torch 2.30.0**,
  **timm 1.0.24**, **einops 0.8.2**, **ftfy 6.3.1**, **regex 2026.1.15**.
- External MME models (pinned commits): **CONCH** and **MUSK** installed from
  their GitHub repos as listed in `environment.yml`.
- API MLLM / local server: **openai 1.63.0**, **fastapi 0.128.4**,
  **uvicorn 0.40.0**; no local PyTorch/GPU required for the API path.
- Local Quilt-LLaVA: **quilt-llava** installed `--no-deps` from its pinned
  GitHub commit (see `scripts/setup_env.sh`) so it reuses the env's PyTorch.
- Shared infra: **huggingface_hub 0.36.2**, **safetensors 0.7.0**,
  **fairscale 0.4.13**, **h5py 3.15.1**, **scikit-learn 1.7.2**.
- Benchmark core: **pandas 2.0.3**, **pydantic 2.10.6**, **PyYAML 6.0.2**,
  **python-dotenv 1.0.1**, **tqdm 4.67.1**, **Pillow 10.4.0**, **requests 2.32.5**.
- `conda` (Miniconda/Anaconda) and `git` are required for setup.

The full, pinned dependency list is in [`environment.yml`](../environment.yml);
the benchmark package's own metadata is in [`pyproject.toml`](../pyproject.toml).

## Non-standard hardware

- Local MMEs and Quilt-LLaVA require an **NVIDIA GPU** with a driver compatible
  with CUDA 12.1.
- Quilt-LLaVA (7B) requires a GPU with sufficient memory to hold a 7B model.
- API-based MLLMs need no local GPU.
- CPU inference (`--set model.device=cpu`) is supported for MMEs but is very slow.

## Typical install time

**Approximately 45 minutes** (measured: 43 min) on an Ubuntu 20.04 workstation
with an empty pip cache, dominated by the PyTorch cu121 wheel downloads; actual
time depends mainly on network bandwidth. Model checkpoints are downloaded
separately (`python tools/download_checkpoints.py`) and add network-dependent
time.

## Expected demo runtime (sample dataset)

The bundled sample dataset (`data/sample_dataset/`: 45 items, 465 patches) is
used for all demo runs. GPU numbers were measured on a single
**NVIDIA A800-SXM4-80GB**.

- **Local MME (Vir2-CLIP)** — **approximately 6 minutes** end-to-end
  (`bash scripts/run_mme_local.sh vir2`).
- **Quilt-LLaVA** — **approximately 15 minutes** end-to-end
  (`bash scripts/run_mllm_qllava.sh`; measured 14 min 34 s: ~9 min server
  startup with cached weights + ~5 min inference). The first run is
  substantially longer, as the ~17 GB of model weights are downloaded before
  serving.
- **API MLLM** — runtime depends on API latency.

## Expected output

Each run prints a terminal table titled `BPRBench Evaluation Summary`
(Model, Accuracy, 95% CI, Samples) and writes CSVs under the run's output directory:

- `<output_dir>/predictions.csv`
- `<output_dir>/metrics/total_accuracy.csv` and related CSVs under `metrics/`
  (`accuracy_by_question.csv`, `accuracy_by_tag.csv`, `efficiency_and_cost.csv`, …).

Default output directories:

| Runner | Output dir |
| --- | --- |
| `scripts/run_mme_local.sh` | `data/results/mme_local/` |
| `scripts/run_mllm_api.sh` | `data/results/mllm_api/` |
| `scripts/run_mllm_qllava.sh` | `data/results/quilt_llava_local/` |

## Confidence intervals

95% confidence intervals are computed by **stratified bootstrapping**
(1,000 resamples, stratified by `question_id`) in
[`benchmark/engine/scorer.py`](../benchmark/engine/scorer.py) (`calc_stratified_acc_ci`).
