"""
Count how many VLM API calls are needed to evaluate one VLM model on the
sample_dataset under configs/mllm_api.yaml.

Logic (mirrors benchmark/engine/inferencer.py + benchmark/models/vlm_impl.py):
  - For each question, target_folders = union of option.folders.
  - samples = indexer.get_samples(target_folders)  (one sample == one API call).
  - Each sample triggers exactly one _call_api() to the model under test.
  - If extractor_repair is enabled, samples whose raw output fails to parse
    additionally trigger up to (1 + extractor_max_retries) calls to the
    auxiliary extractor (GPT-5). Disabled by default in mllm_api.yaml.
  - VLM request retries (max_retries=5) only fire on transient API errors;
    they are NOT counted here (happy-path count).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data/sample_dataset/images"
QUESTIONS_FILE = ROOT / "data/sample_dataset/questions.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MAX_IMAGES_PER_SAMPLE = 20


def count_samples_per_folder(root: Path) -> dict[str, int]:
    """Replicate ImageIndexer._build_index counting logic."""
    counts: dict[str, int] = {}
    if not root.exists():
        raise FileNotFoundError(f"Image dir not found: {root}")
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        n = 0
        for item in sorted(class_dir.iterdir()):
            if item.name.startswith("."):
                continue
            if item.is_dir():
                imgs = [
                    p for p in item.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                ]
                if imgs:
                    n += 1  # one multi-image sample
            elif item.is_file() and item.suffix.lower() in IMAGE_EXTS:
                n += 1  # one single-image sample
        counts[class_dir.name] = n
    return counts


def main() -> None:
    samples_per_folder = count_samples_per_folder(IMAGE_DIR)
    questions = json.loads(QUESTIONS_FILE.read_text())
    questions.sort(key=lambda q: int(q["id"]))

    total_calls = 0
    per_question = []
    folder_call_counter: Counter = Counter()

    for q in questions:
        target_folders = set()
        for opt in q.get("options", []):
            for f in opt.get("folders", []) or []:
                target_folders.add(f)
        n_samples = sum(samples_per_folder.get(f, 0) for f in target_folders)
        per_question.append((int(q["id"]), q.get("question", "")[:60], n_samples))
        total_calls += n_samples
        for f in target_folders:
            folder_call_counter[f] += samples_per_folder.get(f, 0)

    print("=" * 70)
    print("VLM API call count for sample_dataset (configs/mllm_api.yaml)")
    print("=" * 70)
    print(f"Image dir          : {IMAGE_DIR}")
    print(f"Questions file      : {QUESTIONS_FILE}")
    print(f"Number of questions : {len(questions)}")
    print(f"Distinct folders    : {len(samples_per_folder)}")
    print(f"Total samples (all folders, deduplicated) : {sum(samples_per_folder.values())}")
    print()
    print(f"{'QID':>4}  {'#samples (= API calls)':>22}  Question")
    print("-" * 70)
    for qid, qtext, n in per_question:
        print(f"{qid:>4}  {n:>22}  {qtext}")
    print("-" * 70)
    print(f"{'':>4}  {total_calls:>22}  TOTAL API calls (model under test)")
    print()

    # Extractor (auxiliary GPT-5) calls — only if extractor_repair=True.
    print("Auxiliary extractor (GPT-5) calls:")
    print("  configs/mllm_api.yaml sets extractor_repair=False -> 0 extractor calls.")
    print("  (If enabled, each unparseable primary response adds up to")
    print("   1 + extractor_max_retries calls; data-dependent, not knowable ex-ante.)")
    print()
    print(f"GRAND TOTAL API calls (happy path) = {total_calls}")
    print()

    # Also show the smoke-test variant from the README: `run_mllm_api.sh 3`
    if len(questions) >= 3:
        smoke = sum(n for _, _, n in per_question[:3])
        print(f"README smoke test (max_questions=3) -> first 3 questions = {smoke} API calls")


if __name__ == "__main__":
    main()
