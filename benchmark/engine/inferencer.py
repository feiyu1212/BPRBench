from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from benchmark.core.config import BenchmarkConfig
from benchmark.core.logging import get_logger
from benchmark.core.registry import build_model
from benchmark.data.indexer import ImageIndexer
from benchmark.data.questions import load_questions
import benchmark.models  # noqa: F401
from benchmark.models.base import PredictionRecord
from benchmark.utils.cache import JsonlCache
from benchmark.utils.io import ensure_dir

log = get_logger(__name__)


class Inferencer:
    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self.output_dir = ensure_dir(Path(cfg.data.output_dir))

        # Fail-Fast: validate and load the lightweight data layer (image dir,
        # questions file) BEFORE pulling in the multi-billion-param model.
        # A typo in BENCHMARK_IMAGE_DIR or a missing questions.json should
        # crash in seconds, not after minutes of model loading.
        self.indexer = ImageIndexer(cfg.data.image_dir)
        self.questions = load_questions(cfg.data.questions_file)

        # Optional smoke-test / cost-control cap: keep only the first N
        # questions (already sorted by id in load_questions). Applied here so
        # both inference and downstream scoring stay consistent.
        if cfg.eval.max_questions is not None:
            n = cfg.eval.max_questions
            self.questions = self.questions[:n]
            log.info("max_questions=%s: evaluating only the first %d question(s).", n, len(self.questions))

        # Heavyweight model construction happens only once data is confirmed.
        self.model = build_model(cfg.model)

        self.cache = None
        if cfg.eval.use_cache:
            cache_dir = ensure_dir(self.output_dir / "cache")
            cache_name = f"{cfg.model.architecture}_responses.jsonl"
            self.cache = JsonlCache.open(cache_dir / cache_name)

    def _resolve_model_display_name(self) -> str:
        """
        Resolve the Specific Model Name used for reporting.
        Priority:
          1. cfg.model.display_name   (explicit, user-configured)
          2. args['model']            (VLM API model id)
          3. args['checkpoint']       (local checkpoint filename)
          4. args['arch']             (CLIP family sub-arch)
          5. architecture             (final fallback)
        """
        # Explicit display_name always wins; avoids leaking checkpoint
        # filenames like "epoch_10_best.pt" into the result tables.
        if self.cfg.model.display_name:
            return self.cfg.model.display_name

        args = self.cfg.model.args
        arch = self.cfg.model.architecture

        # Case 1: VLM or explicitly named model (e.g. gpt-4o, llama-3)
        if "model" in args and args["model"]:
            return str(args["model"])

        # Case 2: local checkpoint model (e.g. musk, conch, quiltnet)
        # Usually /path/to/checkpoint.pth; use the filename as identifier
        if "checkpoint" in args and args["checkpoint"]:
            ckpt_path = Path(str(args["checkpoint"]))
            if ckpt_path.name and "$" not in ckpt_path.name:
                return ckpt_path.name

        # Case 3: CLIP family, distinguish concrete arch (e.g. ViT-B-32)
        if "arch" in args and args["arch"]:
            return f"{arch}/{args['arch']}"

        # Case 4: fall back to architecture name
        return arch

    def run(self) -> Path:
        out_csv = self.output_dir / "predictions.csv"
        if out_csv.exists() and not self.cfg.eval.force_run:
            try:
                df = pd.read_csv(out_csv)
                if not df.empty:
                    log.info("Predictions already exist and force_run=false. Reusing: %s", out_csv)
                    return out_csv
            except Exception:  # malformed CSV etc. -> re-run
                pass
            log.info("Predictions file exists but is empty or invalid. Re-running inference.")

        log.info("Starting inference with architecture=%s", self.cfg.model.architecture)
        display_model_name = self._resolve_model_display_name()
        log.info("Reporting results under model name: '%s'", display_model_name)

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file",
                    "question_id",
                    "gt_label",
                    "pred_label",
                    "correct",
                    "raw_output",
                    "tags",
                    "model",
                    "input_tokens",
                    "output_tokens",
                    "latency",
                ],
            )
            writer.writeheader()

            total_rows = 0
            # tqdm gives a live ETA / throughput so a multi-hour VLM run is
            # not a silent black box. tqdm is already a project dependency.
            for q in tqdm(self.questions, desc=f"Evaluating {display_model_name}"):
                folder_to_label: Dict[str, str] = {}
                target_folders = set()
                for opt in q.options:
                    for folder in opt.folders:
                        target_folders.add(folder)
                        folder_to_label[folder] = opt.label

                samples = self.indexer.get_samples(target_folders)
                if not samples:
                    log.warning("No samples found for question_id=%s", q.id)
                    continue

                # Pydantic V2 only: the project pins pydantic>=2.10.6, so the
                # legacy `.dict()` fallback branch is dead code.
                option_payload = [o.model_dump() for o in q.options]

                predictions = self._predict_with_optional_cache(
                    samples=samples,
                    question_id=q.id,
                    question_text=q.question,
                    options=option_payload,
                    correct_type=q.correct_type,
                )

                tags = ",".join(q.tags)
                for record in predictions:
                    ident_path = Path(record.sample_id)
                    gt = ""
                    for part in ident_path.parts:
                        if part in folder_to_label:
                            gt = folder_to_label[part]
                            break

                    writer.writerow(
                        {
                            "file": record.sample_id,
                            "question_id": q.id,
                            "gt_label": gt,
                            "pred_label": record.pred_label,
                            "correct": str(gt == record.pred_label),
                            "raw_output": record.raw_output,
                            "tags": tags,
                            "model": display_model_name,
                            "input_tokens": record.input_tokens,
                            "output_tokens": record.output_tokens,
                            "latency": f"{record.latency_s:.4f}",
                        }
                    )
                    total_rows += 1

                # Crash-resilient I/O: flush after every question so an OOM /
                # SIGKILL at hour 3 cannot void the in-flight stdlib buffer.
                # Combined with the JsonlCache this also enables clean
                # checkpoint-resume on the next run.
                f.flush()

        log.info("Inference done. Wrote %s rows to %s", total_rows, out_csv)
        return out_csv

    def _predict_with_optional_cache(
        self,
        samples: List[List[Path]],
        question_id: int,
        question_text: str,
        options: List[Dict[str, Any]],
        correct_type: str,
    ) -> List[PredictionRecord]:
        predict_kwargs = dict(
            question_text=question_text,
            options=options,
            batch_size=self.cfg.eval.batch_size,
            num_workers=self.cfg.eval.num_workers,
            correct_type=correct_type,
            system_prompt=self.cfg.eval.system_prompt,
            extractor_repair=self.cfg.eval.extractor_repair,
            extractor_max_retries=self.cfg.eval.extractor_max_retries,
            extractor_retry_backoff_s=self.cfg.eval.extractor_retry_backoff_s,
        )

        if self.cache is None:
            return self.model.predict(samples=samples, **predict_kwargs)

        valid_labels = [str(o.get("label", "")).strip().upper() for o in options if str(o.get("label", "")).strip()]

        cached_rows: List[PredictionRecord] = []
        pending_samples: List[List[Path]] = []

        for s in samples:
            uniq_id = str(s[0].parent) if len(s) > 1 else str(s[0])
            key = f"qid={question_id}|sample={uniq_id}"
            cached_val = self.cache.get(key)
            if cached_val is None:
                pending_samples.append(s)
                continue
            cached_rows.append(PredictionRecord.from_cache(uniq_id, cached_val, valid_labels))

        if not pending_samples:
            return cached_rows

        fresh_rows = self.model.predict(samples=pending_samples, **predict_kwargs)

        for record in fresh_rows:
            key = f"qid={question_id}|sample={record.sample_id}"
            self.cache.set(
                key,
                {
                    "pred": record.pred_label,
                    "raw": record.raw_output,
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "latency": record.latency_s,
                },
            )

        return cached_rows + fresh_rows
