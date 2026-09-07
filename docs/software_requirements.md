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

- **Python 3.10** — `scripts/setup_envs.sh` creates conda envs with `python=3.10`;
  `pyproject.toml` requires `>=3.10`.
- Local MMEs (`bpr_main`, `bpr_custom`):
  **PyTorch 2.2.0**, **torchvision 0.17.0**, **CUDA 12.1**, **NumPy 1.26.4**,
  plus the `pyproject.toml` packages (`pandas>=2.0`, `Pillow>=10.4.0`,
  `open_clip_torch>=2.30.0`, `transformers==4.48.3`, …).
- Quilt-LLaVA (`bpr_qllava`):
  **PyTorch 2.0.1**, **torchvision 0.15.2**, **CUDA 11.7**.
- API MLLMs:
  `pip install -e ".[vlm]"` (`openai>=1.63.0`); no local PyTorch/GPU required.
- `conda` (Miniconda/Anaconda) and `git` are required for local-model setup.

Full dependency lists are in [`pyproject.toml`](../pyproject.toml) and
[`scripts/setup_envs.sh`](../scripts/setup_envs.sh).

## Non-standard hardware

- Local MMEs and Quilt-LLaVA require an **NVIDIA GPU** with a driver compatible
  with the CUDA builds above.
- Quilt-LLaVA (7B) typically needs on the order of **16 GB** GPU memory.
- API-based MLLMs need no local GPU.
- CPU inference (`--set model.device=cpu`) is supported for MMEs but is very slow.

## Typical install time

- `bash scripts/setup_envs.sh main` — **10–30 minutes** (PyTorch/CUDA download dominates).
- `pip install -e ".[vlm]"` (API path) — **a few minutes**.
- Model checkpoints are downloaded separately and add network-dependent time.

## Expected demo runtime (sample dataset)

The bundled sample dataset (`data/sample_dataset/`: 45 items, 465 patches) is used
for all demo runs below.

| Path | Typical runtime |
| --- | --- |
| Local MME (e.g. Vir2-CLIP) on GPU | a few minutes |
| API MLLM, 3-question smoke test | a few minutes (API-latency dependent) |
| Quilt-LLaVA, first run | substantially longer (downloads 7B weights) |
| Quilt-LLaVA, subsequent runs | tens of minutes on a ≥16 GB GPU |

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
