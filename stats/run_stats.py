from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "data" / "results"
QUESTIONS_PATH = REPO_ROOT / "data" / "questions.json"
METADATA_PATH = REPO_ROOT / "data_stats" / "dataset_metadata_wide_table_cleaned.xlsx"
OUTPUT_DIR = REPO_ROOT / "data_stats" / "results"
STATS_RESULT_DIR = OUTPUT_DIR / "stats_result"
PATIENT_STATS_DIR = OUTPUT_DIR / "patient_stats"


# ==========================================
# Bootstrapping Core Logic (Stratified by Question)
# ==========================================

def calc_stratified_acc_ci(df: pd.DataFrame, n_boot: int = 1000, ci: float = 95.0) -> tuple[float, float, float]:
    """Calculate macro-averaged accuracy with Stratified Bootstrapping (stratified by question)."""
    q_groups = [group for _, group in df.groupby("question_id")]
    if not q_groups:
        return np.nan, np.nan, np.nan

    boot_q_means = []
    orig_q_means = []

    for q_df in q_groups:
        vals = q_df["correct"].values.astype(float)
        n_samples = len(vals)
        if n_samples == 0:
            continue
            
        orig_q_means.append(vals.mean())
        
        # Resample within the question's images
        idx = np.random.randint(0, n_samples, size=(n_boot, n_samples))
        boot_q_means.append(vals[idx].mean(axis=1))

    if not boot_q_means:
        return np.nan, np.nan, np.nan

    # Macro-average across questions
    boot_macro = np.vstack(boot_q_means).mean(axis=0)
    orig_macro = np.mean(orig_q_means)

    lower = np.percentile(boot_macro, (100 - ci) / 2)
    upper = np.percentile(boot_macro, 100 - (100 - ci) / 2)
    return orig_macro, lower, upper


def calc_stratified_senspec_ci(df: pd.DataFrame, pos_label: str = "A", n_boot: int = 1000, ci: float = 95.0) -> dict:
    """Calculate macro-averaged Sens/Spec with Stratified Bootstrapping."""
    q_groups = [group for _, group in df.groupby("question_id")]
    if not q_groups:
        return {
            "sensitivity": np.nan, "sens_ci_lower": np.nan, "sens_ci_upper": np.nan,
            "specificity": np.nan, "spec_ci_lower": np.nan, "spec_ci_upper": np.nan
        }

    boot_sens_list, boot_spec_list = [], []
    orig_sens_list, orig_spec_list = [], []

    for q_df in q_groups:
        gt_pos = (q_df["option_label"] == pos_label).values
        pr_pos = (q_df["pred_option"] == pos_label).values
        n_samples = len(gt_pos)
        if n_samples == 0:
            continue

        # Original exact calculation
        tp = (gt_pos & pr_pos).sum()
        fn = (gt_pos & ~pr_pos).sum()
        tn = (~gt_pos & ~pr_pos).sum()
        fp = (~gt_pos & pr_pos).sum()
        orig_sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
        orig_spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)

        # Bootstrapping calculation
        idx = np.random.randint(0, n_samples, size=(n_boot, n_samples))
        gt_boot = gt_pos[idx]
        pr_boot = pr_pos[idx]

        tp_boot = (gt_boot & pr_boot).sum(axis=1)
        fn_boot = (gt_boot & ~pr_boot).sum(axis=1)
        tn_boot = (~gt_boot & ~pr_boot).sum(axis=1)
        fp_boot = (~gt_boot & pr_boot).sum(axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            sens_boot = np.where((tp_boot + fn_boot) > 0, tp_boot / (tp_boot + fn_boot), np.nan)
            spec_boot = np.where((tn_boot + fp_boot) > 0, tn_boot / (tn_boot + fp_boot), np.nan)

        boot_sens_list.append(sens_boot)
        boot_spec_list.append(spec_boot)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        orig_macro_sens = np.nanmean(orig_sens_list)
        orig_macro_spec = np.nanmean(orig_spec_list)

        boot_macro_sens = np.nanmean(np.vstack(boot_sens_list), axis=0) if boot_sens_list else np.full(n_boot, np.nan)
        boot_macro_spec = np.nanmean(np.vstack(boot_spec_list), axis=0) if boot_spec_list else np.full(n_boot, np.nan)

        sens_lower = np.nanpercentile(boot_macro_sens, (100 - ci) / 2) if not np.isnan(boot_macro_sens).all() else np.nan
        sens_upper = np.nanpercentile(boot_macro_sens, 100 - (100 - ci) / 2) if not np.isnan(boot_macro_sens).all() else np.nan
        spec_lower = np.nanpercentile(boot_macro_spec, (100 - ci) / 2) if not np.isnan(boot_macro_spec).all() else np.nan
        spec_upper = np.nanpercentile(boot_macro_spec, 100 - (100 - ci) / 2) if not np.isnan(boot_macro_spec).all() else np.nan

    return {
        "sensitivity": orig_macro_sens, "sens_ci_lower": sens_lower, "sens_ci_upper": sens_upper,
        "specificity": orig_macro_spec, "spec_ci_lower": spec_lower, "spec_ci_upper": spec_upper
    }


def apply_acc_ci(group: pd.DataFrame) -> pd.Series:
    """Wrapper to apply stratified accuracy bootstrap on groupby objects."""
    mean, lower, upper = calc_stratified_acc_ci(group)
    return pd.Series({
        "accuracy": mean,
        "accuracy_ci_lower": lower,
        "accuracy_ci_upper": upper,
        "sample_count": len(group),
        "question_count": group["question_id"].nunique(),
        "image_count": group["file"].nunique() if "file" in group.columns else 0
    })


def apply_senspec_ci(group: pd.DataFrame, pos_label: str = "A") -> pd.Series:
    """Wrapper to apply stratified sens/spec bootstrap on groupby objects."""
    metrics = calc_stratified_senspec_ci(group, pos_label=pos_label)
    
    gt_pos = group["option_label"] == pos_label
    pr_pos = group["pred_option"] == pos_label
    
    tp = (gt_pos & pr_pos).sum()
    fn = (gt_pos & ~pr_pos).sum()
    tn = (~gt_pos & ~pr_pos).sum()
    fp = (~gt_pos & pr_pos).sum()
    
    metrics.update({
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "total_samples": len(group),
        "question_count": group["question_id"].nunique()
    })
    return pd.Series(metrics)

# ==========================================
# Data Loading & Prep
# ==========================================

def load_questions() -> tuple[pd.DataFrame, dict[int, str], dict[int, str]]:
    raw = json.loads(Path(QUESTIONS_PATH).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Questions must be a list, got {type(raw)}")

    qid_to_text: dict[int, str] = {}
    qid_to_tag: dict[int, str] = {}
    mapping_records: list[dict] = []

    for q in raw:
        if not isinstance(q, dict): continue
        qid = int(q.get("id", 0))
        qid_to_text[qid] = str(q.get("question") or q.get("text") or q.get("question_text") or "").strip()
        tags = q.get("tags") or ["default"]
        tag = tags[0] if isinstance(tags, list) and tags else str(tags) if tags else "default"
        qid_to_tag[qid] = tag

        for opt in q.get("options", []):
            if not isinstance(opt, dict): continue
            label = str(opt.get("label", "")).strip().upper()
            for f in opt.get("folders", []):
                folder = str(f).strip()
                if folder:
                    mapping_records.append({
                        "folder": folder, "question_id": qid, "option_label": label, "tag": tag
                    })

    return pd.DataFrame(mapping_records), qid_to_text, qid_to_tag


def load_metadata() -> pd.DataFrame | None:
    if not METADATA_PATH.exists(): return None
    df = pd.read_excel(METADATA_PATH)
    required_cols = {"original_path", "folder", "patient_id"}
    if not required_cols.issubset(df.columns):
        print(f"  [skip] Metadata missing required columns: {required_cols - set(df.columns)}")
        return None
    df["original_path"] = df["original_path"].fillna("").astype(str).str.strip()
    df["folder"] = df["folder"].fillna("").astype(str).str.strip()
    df["patient_id"] = df["patient_id"].fillna("").astype(str).str.strip()
    return df


def generate_dimension_stats(df: pd.DataFrame, is_fallback: bool = False) -> None:
    STATS_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    dedup_opt = df.drop_duplicates(subset=["file", "question_id", "option_label"], keep="first")
    aggs = {"file": "nunique", "patient_id": lambda s: s[s != ""].nunique()}
    rename_aggs = {"file": "image_count", "patient_id": "unique_person_count"}

    dedup_opt.groupby("tag").agg(aggs).rename(columns=rename_aggs).reset_index().to_csv(STATS_RESULT_DIR / "tag_stats.csv", index=False)
    dedup_opt.groupby("question_id").agg(aggs).rename(columns=rename_aggs).reset_index().to_csv(STATS_RESULT_DIR / "question_stats.csv", index=False)
    dedup_opt.groupby(["question_id", "option_label"]).agg(aggs).rename(columns=rename_aggs).reset_index().to_csv(STATS_RESULT_DIR / "question_option_stats.csv", index=False)

    dedup_q = df.drop_duplicates(subset=["file", "question_id"], keep="first")
    dedup_q.groupby("question_id").agg(aggs).rename(columns={"file": "file_count", "patient_id": "name_count"}).reset_index().to_csv(STATS_RESULT_DIR / "question_stat.csv", index=False)

    if is_fallback:
        detail_cols = ["file", "question_id", "option_label", "tag", "question_text", "patient_id"]
        df[[c for c in detail_cols if c in df.columns]].drop_duplicates().to_csv(STATS_RESULT_DIR / "detail.csv", index=False)


def run_stats_from_metadata_flow(metadata_df: pd.DataFrame, folder_mapping_df: pd.DataFrame) -> bool:
    if metadata_df is None or folder_mapping_df.empty: return False
    merged = metadata_df.merge(folder_mapping_df, on="folder", how="inner").rename(columns={"original_path": "file"})
    if merged.empty: return False
    generate_dimension_stats(merged, is_fallback=False)
    return True


def load_and_merge_predictions(qid_to_text: dict[int, str], qid_to_tag: dict[int, str]) -> pd.DataFrame:
    pred_files = [p for p in sorted(RESULTS_DIR.glob("**/predictions.csv")) if p.stat().st_size > 0]
    if not pred_files: raise FileNotFoundError(f"No valid predictions.csv found under {RESULTS_DIR}")

    all_dfs = []
    required = {"file", "question_id", "gt_label", "pred_label"}
    for p in pred_files:
        try:
            df = pd.read_csv(p, engine="python", on_bad_lines="skip")
        except Exception:
            continue
        if df.empty or not required.issubset(df.columns): continue

        local = df.copy()
        model_name = local["model"].iloc[0] if "model" in local.columns and not local["model"].isna().all() else p.parent.name
        local["model"] = str(model_name).strip()

        local["question_id"] = pd.to_numeric(local["question_id"], errors="coerce").astype("Int64")
        local = local.dropna(subset=["question_id"]).copy()
        local["question_id"] = local["question_id"].astype(int)

        local["gt_label"] = local["gt_label"].fillna("").astype(str).str.upper().str.strip()
        local["pred_label"] = local["pred_label"].fillna("").astype(str).str.upper().str.strip()

        if "correct" in local.columns:
            mapped = local["correct"].astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})
            local["correct"] = mapped.where(mapped.notna(), local["pred_label"] == local["gt_label"])
        else:
            local["correct"] = local["pred_label"] == local["gt_label"]
        local["correct"] = local["correct"].astype(bool)

        local["tag"] = local["tags"].fillna("").astype(str).str.strip() if "tags" in local.columns and not local["tags"].isna().all() else local["question_id"].map(qid_to_tag)
        local["question_text"] = local["question_id"].map(qid_to_text)
        all_dfs.append(local)

    if not all_dfs: raise RuntimeError("No valid predictions could be loaded.")
    return pd.concat(all_dfs, ignore_index=True)


def merge_patient_id(df: pd.DataFrame, metadata_df: pd.DataFrame | None) -> pd.DataFrame:
    if metadata_df is None:
        df["patient_id"] = ""
        return df
    meta_sub = metadata_df[["original_path", "patient_id"]].drop_duplicates(subset=["original_path"], keep="first")
    merged = df.merge(meta_sub, left_on="file", right_on="original_path", how="left").drop(columns=["original_path"], errors="ignore")
    merged["patient_id"] = merged["patient_id"].fillna("").astype(str).str.strip()
    return merged


def build_unique_df(full_df: pd.DataFrame) -> pd.DataFrame:
    unique = full_df.drop_duplicates(subset=["file", "model", "question_id"], keep="first").copy()
    return unique.rename(columns={"gt_label": "option_label", "pred_label": "pred_option"})


# ==========================================
# Metric Evaluation Pipelines
# ==========================================

def run_eval_metrics(unique_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_cols = ["model", "tag", "question_id", "question_text", "file", "option_label", "pred_option", "correct"]
    unique_df[[c for c in out_cols if c in unique_df.columns]].to_csv(OUTPUT_DIR / "unique_df.csv", index=False)

    # Question Level
    base_q = unique_df.groupby(["model", "tag", "question_id", "question_text"], dropna=False).apply(apply_acc_ci).reset_index()
    base_q.to_csv(OUTPUT_DIR / "average_accuracy_per_model_question_id.csv", index=False)

    # Model Overall
    total_model = unique_df.groupby("model").apply(apply_acc_ci).reset_index()
    total_model.to_csv(OUTPUT_DIR / "total_accuracy_per_model.csv", index=False)

    # Tag Level
    total_tag = unique_df.groupby(["model", "tag"], dropna=False).apply(apply_acc_ci).reset_index()
    total_tag.to_csv(OUTPUT_DIR / "average_accuracy_per_model_tag.csv", index=False)


def run_eval_sens_spec(unique_df: pd.DataFrame, positive_label: str = "A") -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pos = positive_label.upper().strip()

    # Question Level
    base_q = unique_df.groupby(["model", "tag", "question_id", "question_text"], dropna=False).apply(
        lambda g: apply_senspec_ci(g, pos)
    ).reset_index()
    base_q.to_csv(OUTPUT_DIR / "average_senspec_per_model_tag_question_id.csv", index=False)

    # Model + Tag Level
    tag_df = unique_df.groupby(["model", "tag"], dropna=False).apply(lambda g: apply_senspec_ci(g, pos)).reset_index()
    
    # Model Overall Level
    model_df = unique_df.groupby(["model"]).apply(lambda g: apply_senspec_ci(g, pos)).reset_index()
    model_df["tag"] = "" # Keep schema aligned

    combined_tag = pd.concat([model_df, tag_df], ignore_index=True).sort_values(by=["model", "tag"]).reset_index(drop=True)
    combined_tag.to_csv(OUTPUT_DIR / "average_senspec_per_model_tag.csv", index=False)


def run_eval_by_patient(unique_df: pd.DataFrame, positive_label: str = "A") -> None:
    PATIENT_STATS_DIR.mkdir(parents=True, exist_ok=True)
    df = unique_df[unique_df["patient_id"] != ""].copy()
    if df.empty: return
    pos = positive_label.upper().strip()

    # ====== A. Accuracy Part ======
    df.groupby(["model", "patient_id", "tag", "question_id", "question_text"], dropna=False).apply(apply_acc_ci).reset_index().to_csv(PATIENT_STATS_DIR / "accuracy_per_model_patient_question.csv", index=False)
    df.groupby(["model", "patient_id"]).apply(apply_acc_ci).reset_index().to_csv(PATIENT_STATS_DIR / "accuracy_per_model_patient.csv", index=False)
    df.groupby(["model", "patient_id", "tag"], dropna=False).apply(apply_acc_ci).reset_index().to_csv(PATIENT_STATS_DIR / "accuracy_per_model_patient_tag.csv", index=False)

    # ====== B. Sens/Spec Part ======
    df.groupby(["model", "patient_id", "tag", "question_id", "question_text"], dropna=False).apply(lambda g: apply_senspec_ci(g, pos)).reset_index().to_csv(PATIENT_STATS_DIR / "sens_spec_per_model_patient_question.csv", index=False)
    df.groupby(["model", "patient_id"]).apply(lambda g: apply_senspec_ci(g, pos)).reset_index().to_csv(PATIENT_STATS_DIR / "sens_spec_per_model_patient.csv", index=False)
    df.groupby(["model", "patient_id", "tag"], dropna=False).apply(lambda g: apply_senspec_ci(g, pos)).reset_index().to_csv(PATIENT_STATS_DIR / "sens_spec_per_model_patient_tag.csv", index=False)

    # ====== C. Patient Overall Part (Ignoring Models) ======
    df.groupby(["patient_id", "tag", "question_id", "question_text"], dropna=False).apply(apply_acc_ci).reset_index().to_csv(PATIENT_STATS_DIR / "patient_question_stats.csv", index=False)
    df.groupby(["patient_id", "tag"], dropna=False).apply(apply_acc_ci).reset_index().to_csv(PATIENT_STATS_DIR / "patient_tag_stats.csv", index=False)
    df.groupby("patient_id").apply(apply_acc_ci).reset_index().rename(columns={"accuracy": "overall_accuracy"}).to_csv(PATIENT_STATS_DIR / "patient_overall_stats.csv", index=False)


def main() -> None:
    print("Loading definitions (Questions & Metadata)...")
    folder_mapping_df, qid_to_text, qid_to_tag = load_questions()
    metadata_df = load_metadata()

    print("Running dimension stats from metadata...")
    metadata_success = run_stats_from_metadata_flow(metadata_df, folder_mapping_df)

    print("Loading and merging predictions...")
    full_df = load_and_merge_predictions(qid_to_text, qid_to_tag)
    full_df = merge_patient_id(full_df, metadata_df)
    unique_df = build_unique_df(full_df)
    print(f"  Unique rows: {len(unique_df)}, models: {unique_df['model'].nunique()}")

    if not metadata_success:
        print("  Fallback: running dimension stats from predictions...")
        generate_dimension_stats(unique_df, is_fallback=True)

    print("Running eval_metrics (Stratified Bootstrapping)...")
    run_eval_metrics(unique_df)

    print("Running eval_sens_spec (Stratified Bootstrapping)...")
    run_eval_sens_spec(unique_df)

    print("Running eval_by_patient (Stratified Bootstrapping)...")
    run_eval_by_patient(unique_df)

    print(f"Done. Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
