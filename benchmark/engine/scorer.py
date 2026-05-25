from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark.core.logging import get_logger
from benchmark.utils.io import ensure_dir

log = get_logger(__name__)


class Scorer:
    def __init__(self, output_dir: Path, positive_label: str = "A"):
        self.output_dir = Path(output_dir)
        self.positive_label = str(positive_label).strip().upper() or "A"

    def run(self) -> Path:
        pred_path = self.output_dir / "predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions not found: {pred_path}")

        df = pd.read_csv(pred_path)
        if df.empty:
            raise RuntimeError("Predictions file is empty.")

        # Normalize labels
        df["pred_label"] = df["pred_label"].astype(str).str.upper().str.strip()
        df["gt_label"] = df["gt_label"].astype(str).str.upper().str.strip()

        if "correct" not in df.columns:
            df["correct"] = df["pred_label"] == df["gt_label"]
        else:
            df["correct"] = df["correct"].astype(str).str.lower().map({"true": True, "false": False}).fillna(
                df["pred_label"] == df["gt_label"]
            )

        metrics_dir = ensure_dir(self.output_dir / "metrics")

        summary_df = self._build_summary(df)
        by_question_df = (
            df.groupby("question_id", as_index=False)["correct"].mean().rename(columns={"correct": "accuracy"})
        )
        by_tag_df = df.assign(tag=df["tags"].fillna(""))
        by_tag_df = by_tag_df.groupby("tag", as_index=False)["correct"].mean().rename(columns={"correct": "accuracy"})

        summary_df.to_csv(metrics_dir / "summary.csv", index=False)
        by_question_df.to_csv(metrics_dir / "accuracy_by_question.csv", index=False)
        by_tag_df.to_csv(metrics_dir / "accuracy_by_tag.csv", index=False)

        log.info("Scoring done. Metrics saved to %s", metrics_dir)
        return metrics_dir

    def _build_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        accuracy = float(df["correct"].mean())

        pos = self.positive_label
        tp = int(((df["gt_label"] == pos) & (df["pred_label"] == pos)).sum())
        tn = int(((df["gt_label"] != pos) & (df["pred_label"] != pos)).sum())
        fp = int(((df["gt_label"] != pos) & (df["pred_label"] == pos)).sum())
        fn = int(((df["gt_label"] == pos) & (df["pred_label"] != pos)).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return pd.DataFrame(
            [
                {
                    "total_samples": int(len(df)),
                    "accuracy": accuracy,
                    "positive_label": pos,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "TP": tp,
                    "TN": tn,
                    "FP": fp,
                    "FN": fn,
                }
            ]
        )
