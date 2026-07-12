from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.core.config import BenchmarkConfig
from benchmark.core.logging import get_logger
from benchmark.utils.io import ensure_dir

log = get_logger(__name__)


def calc_stratified_acc_ci(df: pd.DataFrame, n_boot: int = 1000, ci: float = 95.0) -> tuple[float, float, float]:
    """Macro-averaged accuracy with Stratified Bootstrapping (stratified by question)."""
    q_groups = [group for _, group in df.groupby("question_id")]
    if not q_groups:
        return float("nan"), float("nan"), float("nan")

    boot_q_means, orig_q_means = [], []
    for q_df in q_groups:
        vals = q_df["correct"].values.astype(float)
        n_samples = len(vals)
        if n_samples == 0:
            continue
        orig_q_means.append(vals.mean())
        idx = np.random.randint(0, n_samples, size=(n_boot, n_samples))
        boot_q_means.append(vals[idx].mean(axis=1))

    if not boot_q_means:
        return float("nan"), float("nan"), float("nan")

    boot_macro = np.vstack(boot_q_means).mean(axis=0)
    orig_macro = float(np.mean(orig_q_means))

    lower = float(np.percentile(boot_macro, (100 - ci) / 2))
    upper = float(np.percentile(boot_macro, 100 - (100 - ci) / 2))
    return orig_macro, lower, upper


def _acc_ci_series(group: pd.DataFrame) -> pd.Series:
    mean_val, lower, upper = calc_stratified_acc_ci(group)
    return pd.Series(
        {
            "accuracy": mean_val,
            "accuracy_ci_lower": lower,
            "accuracy_ci_upper": upper,
            "sample_count": len(group),
            "question_count": group["question_id"].nunique(),
        }
    )


def _grouped_acc_ci(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """
    Compute macro-accuracy + 95% CI per group without triggering pandas'
    `apply on grouping columns` deprecation. Iterates groups explicitly so the
    grouping keys stay available as output columns and `question_id` remains
    accessible inside each group for stratified bootstrapping.
    """
    records: list[dict] = []
    for key_vals, group in df.groupby(keys, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = dict(zip(keys, key_vals))
        row.update(_acc_ci_series(group).to_dict())
        records.append(row)
    return pd.DataFrame(records)


class Scorer:
    """Unified evaluation pipeline: macro-accuracy + 95% CI + cost/efficiency."""

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.data.output_dir)
        self.metrics_dir = ensure_dir(self.output_dir / "metrics")

    def run(self) -> Path:
        pred_path = self.output_dir / "predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions not found: {pred_path}")

        df = pd.read_csv(pred_path, engine="python")
        if df.empty:
            raise RuntimeError("Predictions file is empty.")

        # Normalize labels and correctness.
        df["pred_label"] = df["pred_label"].fillna("").astype(str).str.upper().str.strip()
        df["gt_label"] = df["gt_label"].fillna("").astype(str).str.upper().str.strip()
        # The inferencer writes the comma-joined tag list as the `tags` column;
        # normalize it to the canonical `tag` column used throughout scoring.
        if "tags" in df.columns:
            df = df.rename(columns={"tags": "tag"})
        if "tag" not in df.columns:
            df["tag"] = "default"
        df["tag"] = df["tag"].fillna("").astype(str).str.strip()
        if "correct" in df.columns:
            mapped = (
                df["correct"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
            )
            df["correct"] = mapped.where(mapped.notna(), df["pred_label"] == df["gt_label"])
        else:
            df["correct"] = df["pred_label"] == df["gt_label"]
        df["correct"] = df["correct"].astype(bool)

        log.info("Computing macro-averaged accuracy with 95% Confidence Intervals...")

        # 1. Question-level (item-level) accuracy with CI
        by_question = _grouped_acc_ci(df, ["model", "tag", "question_id"])
        by_question.to_csv(self.metrics_dir / "accuracy_by_question.csv", index=False)

        # 2. Tag-level (task-level) macro-accuracy across items within each task
        by_tag = _grouped_acc_ci(df, ["model", "tag"])
        by_tag.to_csv(self.metrics_dir / "accuracy_by_tag.csv", index=False)

        # 3. Overall per-model macro-accuracy with CI
        total_model = _grouped_acc_ci(df, ["model"])
        total_model.to_csv(self.metrics_dir / "total_accuracy.csv", index=False)

        # 4. Cost & efficiency (Extended Data Table 2/3)
        self._calc_cost_efficiency(df)

        # 5. Optional patient-dimension stats (soft dependency on metadata xlsx)
        self._maybe_dimension_stats()

        # 6. Terminal summary (UX): print key metrics so the user gets immediate
        # feedback without having to open the CSV files manually.
        self._print_terminal_summary(total_model)

        log.info("Scoring complete. Metrics saved to %s", self.metrics_dir)
        return self.metrics_dir

    def _match_pricing(self, model_name: str) -> tuple[float, float]:
        name_lower = str(model_name).lower()
        best_p_in, best_p_out, best_len = 0.0, 0.0, -1
        for key, pricing in self.cfg.defaults.pricing.items():
            if key.lower() in name_lower and len(key) > best_len:
                best_p_in = float(pricing.get("in", 0.0))
                best_p_out = float(pricing.get("out", 0.0))
                best_len = len(key)
        return best_p_in, best_p_out

    def _calc_cost_efficiency(self, df: pd.DataFrame) -> None:
        """Per-model average tokens, latency, and vendor cost (Extended Data Table 2/3)."""
        for col in ("input_tokens", "output_tokens", "latency"):
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        if "model" not in df.columns:
            log.warning("Cost stats skipped: 'model' column missing.")
            return

        stats: list[dict] = []
        for model_name, group in df.groupby("model"):
            avg_in = float(group["input_tokens"].mean())
            avg_out = float(group["output_tokens"].mean())
            avg_lat = float(group["latency"].mean())
            p_in, p_out = self._match_pricing(model_name)
            avg_cost = avg_in * p_in + avg_out * p_out

            stats.append(
                {
                    "Model": model_name,
                    "Sample Count": int(len(group)),
                    "Avg Inference Time (s)": round(avg_lat, 4),
                    "Avg Input Tokens": round(avg_in, 2),
                    "Avg Output Tokens": round(avg_out, 2),
                    "Total Input Tokens": int(group["input_tokens"].sum()),
                    "Total Output Tokens": int(group["output_tokens"].sum()),
                    "Input Price (USD/1M)": round(p_in * 1_000_000, 6) if p_in else 0.0,
                    "Output Price (USD/1M)": round(p_out * 1_000_000, 6) if p_out else 0.0,
                    "Vendor Pricing (USD)": f"{avg_cost:.6f}" if avg_cost > 0 else "-",
                }
            )

        out_path = self.metrics_dir / "efficiency_and_cost.csv"
        pd.DataFrame(stats).sort_values(by="Model").reset_index(drop=True).to_csv(out_path, index=False)

    def _maybe_dimension_stats(self) -> None:
        """
        Optional dataset-scale dimension stats (unique-person counts etc.).
        Soft dependency on the metadata file configured via
        DataConfig.metadata_file; silently skipped when the file is absent
        or unset. No patient-level performance aggregation.
        """
        metadata_path = self.cfg.data.metadata_file
        if not metadata_path or not metadata_path.exists():
            return

        try:
            meta = pd.read_excel(metadata_path)
        except Exception as exc:
            log.warning("Metadata read failed, skipping dimension stats: %s", exc)
            return

        required = {"original_path", "folder", "patient_id"}
        if not required.issubset(meta.columns):
            log.warning("Metadata missing columns %s; skipping dimension stats.", required - set(meta.columns))
            return

        meta["original_path"] = meta["original_path"].fillna("").astype(str).str.strip()
        meta["folder"] = meta["folder"].fillna("").astype(str).str.strip()
        meta["patient_id"] = meta["patient_id"].fillna("").astype(str).str.strip()

        # Build folder -> (question_id, option_label, tag) mapping from the questions file.
        try:
            import json

            raw = json.loads(Path(self.cfg.data.questions_file).read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Questions load failed for dimension stats: %s", exc)
            return

        mapping_records: list[dict] = []
        for q in raw if isinstance(raw, list) else []:
            if not isinstance(q, dict):
                continue
            qid = int(q.get("id", 0))
            tags = q.get("tags") or ["default"]
            tag = tags[0] if isinstance(tags, list) and tags else str(tags) if tags else "default"
            for opt in q.get("options", []):
                if not isinstance(opt, dict):
                    continue
                label = str(opt.get("label", "")).strip().upper()
                for f in opt.get("folders", []):
                    folder = str(f).strip()
                    if folder:
                        mapping_records.append(
                            {"folder": folder, "question_id": qid, "option_label": label, "tag": tag}
                        )

        if not mapping_records:
            return
        folder_map = pd.DataFrame(mapping_records)
        merged = meta.merge(folder_map, on="folder", how="inner").rename(columns={"original_path": "file"})
        if merged.empty:
            return

        dedup = merged.drop_duplicates(subset=["file", "question_id", "option_label"], keep="first")
        aggs = {"file": "nunique", "patient_id": lambda s: s[s != ""].nunique()}
        rename = {"file": "image_count", "patient_id": "unique_person_count"}

        dedup.groupby("tag").agg(aggs).rename(columns=rename).reset_index().to_csv(
            self.metrics_dir / "dimension_tag_stats.csv", index=False
        )
        dedup.groupby("question_id").agg(aggs).rename(columns=rename).reset_index().to_csv(
            self.metrics_dir / "dimension_question_stats.csv", index=False
        )
        dedup.groupby(["question_id", "option_label"]).agg(aggs).rename(columns=rename).reset_index().to_csv(
            self.metrics_dir / "dimension_question_option_stats.csv", index=False
        )

    def _print_terminal_summary(self, df: pd.DataFrame) -> None:
        """Print a concise metrics table to stdout for immediate visual feedback."""
        if df.empty or "model" not in df.columns:
            return

        print("\n" + "=" * 80)
        print(f"{'BPRBench Evaluation Summary':=^80}")
        print("=" * 80)

        display = df.copy()
        display["Model"] = display["model"]
        if "accuracy" in display.columns:
            display["Accuracy"] = display["accuracy"].map(
                lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "-"
            )
        else:
            display["Accuracy"] = "-"
        display["95% CI"] = display.apply(
            lambda r: (
                f"[{r['accuracy_ci_lower'] * 100:.2f}%, {r['accuracy_ci_upper'] * 100:.2f}%]"
                if pd.notna(r.get("accuracy_ci_lower")) and pd.notna(r.get("accuracy_ci_upper"))
                else "-"
            ),
            axis=1,
        )
        display["Samples"] = display.get("sample_count", 0)

        cols = [c for c in ["Model", "Accuracy", "95% CI", "Samples"] if c in display.columns]
        print(display[cols].to_string(index=False))

        eff_path = self.metrics_dir / "efficiency_and_cost.csv"
        if eff_path.exists():
            try:
                eff_df = pd.read_csv(eff_path)
                eff_cols = [
                    "Model",
                    "Avg Inference Time (s)",
                    "Avg Input Tokens",
                    "Avg Output Tokens",
                ]
                existing = [c for c in eff_cols if c in eff_df.columns]
                if existing:
                    print("-" * 80)
                    print(eff_df[existing].to_string(index=False))
            except Exception:
                pass

        print("=" * 80 + "\n")
