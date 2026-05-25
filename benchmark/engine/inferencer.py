from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from benchmark.core.config import BenchmarkConfig
from benchmark.core.logging import get_logger
from benchmark.core.registry import build_model
from benchmark.data.indexer import ImageIndexer
from benchmark.data.questions import load_questions
import benchmark.models  # noqa: F401
from benchmark.utils.cache import JsonlCache
from benchmark.utils.io import ensure_dir
from benchmark.utils.parsers import parse_option_letter

log = get_logger(__name__)


class Inferencer:
    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self.output_dir = ensure_dir(Path(cfg.data.output_dir))
        self.model = build_model(cfg.model)
        self.indexer = ImageIndexer(cfg.data.image_dir)
        self.questions = load_questions(cfg.data.questions_file)

        self.cache = None
        if cfg.eval.use_cache:
            cache_dir = ensure_dir(self.output_dir / "cache")
            cache_name = f"{cfg.model.architecture}_responses.jsonl"
            self.cache = JsonlCache.open(cache_dir / cache_name)

    def _resolve_model_display_name(self) -> str:
        """
        根据配置智能解析用于显示的 Specific Model Name。
        优先级：args['model'] > args['checkpoint'] > args['arch'] > architecture
        """
        args = self.cfg.model.args
        arch = self.cfg.model.architecture

        # Case 1: VLM 或明确指定了模型名称的 (如 gpt-4o, llama-3)
        if "model" in args and args["model"]:
            return str(args["model"])

        # Case 2: 本地权重模型 (如 musk, conch, quiltnet)
        # 通常是 /path/to/checkpoint.pth，我们取文件名作为标识
        if "checkpoint" in args and args["checkpoint"]:
            ckpt_path = Path(str(args["checkpoint"]))
            # 如果是 ${...} 变量没解析开，直接返回，否则返回文件名
            if ckpt_path.name and "$" not in ckpt_path.name:
                return ckpt_path.name

        # Case 3: CLIP 类模型，区分具体架构 (如 ViT-B-32)
        if "arch" in args and args["arch"]:
            # 组合名称，例如: openclip/ViT-B-32
            return f"{arch}/{args['arch']}"

        # Case 4: 兜底使用架构名
        return arch

    def run(self) -> Path:
        out_csv = self.output_dir / "predictions.csv"
        if out_csv.exists() and not self.cfg.eval.force_run:
            # Reuse only if file has at least one data row (not just header or header+blank)
            try:
                df = pd.read_csv(out_csv)
                if not df.empty:
                    log.info("Predictions already exist and force_run=false. Reusing: %s", out_csv)
                    return out_csv
            except Exception:  # malformed CSV etc. -> re-run
                pass
            log.info("Predictions file exists but is empty or invalid. Re-running inference.")

        log.info("Starting inference with architecture=%s", self.cfg.model.architecture)

        # 获取具体的模型显示名称
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
                ],
            )
            writer.writeheader()

            total_rows = 0
            for q in self.questions:
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

                option_payload = [
                    o.model_dump() if hasattr(o, "model_dump") else o.dict()  # type: ignore[attr-defined]
                    for o in q.options
                ]

                predictions = self._predict_with_optional_cache(
                    samples=samples,
                    question_id=q.id,
                    question_text=q.question,
                    options=option_payload,
                    correct_type=q.correct_type,
                )

                tags = ",".join(q.tags)
                for identifier, pred, raw in predictions:
                    ident_path = Path(identifier)
                    gt = ""
                    for part in ident_path.parts:
                        if part in folder_to_label:
                            gt = folder_to_label[part]
                            break

                    writer.writerow(
                        {
                            "file": identifier,
                            "question_id": q.id,
                            "gt_label": gt,
                            "pred_label": pred,
                            "correct": str(gt == pred),
                            "raw_output": raw,
                            "tags": tags,
                            "model": display_model_name,
                        }
                    )
                    total_rows += 1

        log.info("Inference done. Wrote %s rows to %s", total_rows, out_csv)
        return out_csv

    def _predict_with_optional_cache(
        self,
        samples: List[List[Path]],
        question_id: int,
        question_text: str,
        options: List[Dict[str, Any]],
        correct_type: str,
    ) -> List[tuple[str, str, str]]:
        if self.cache is None:
            return self.model.predict(
                samples=samples,
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

        valid_labels = [str(o.get("label", "")).strip().upper() for o in options if str(o.get("label", "")).strip()]

        cached_rows: List[tuple[str, str, str]] = []
        pending_samples: List[List[Path]] = []

        for s in samples:
            if len(s) > 1:
                uniq_id = str(s[0].parent)
            else:
                uniq_id = str(s[0])
            key = f"qid={question_id}|sample={uniq_id}"
            cached_val = self.cache.get(key)
            if cached_val is None:
                pending_samples.append(s)
                continue

            if isinstance(cached_val, dict):
                pred = str(cached_val.get("pred", ""))
                raw = str(cached_val.get("raw", ""))
            else:
                raw = str(cached_val)
                pred = parse_option_letter(raw, valid_labels)
            cached_rows.append((uniq_id, pred, raw))

        if not pending_samples:
            return cached_rows

        fresh_rows = self.model.predict(
            samples=pending_samples,
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

        for identifier, pred, raw in fresh_rows:
            key = f"qid={question_id}|sample={identifier}"
            self.cache.set(key, {"pred": pred, "raw": raw})

        return cached_rows + fresh_rows
