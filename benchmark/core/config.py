from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


def _apply_overrides(raw: dict, overrides: List[str]) -> None:
    """Apply lightweight dotted-key overrides, e.g. eval.batch_size=16."""
    for override in overrides:
        if "=" not in override:
            continue
        key_path, val_str = override.split("=", 1)

        val_lower = val_str.lower()
        if val_lower in ("true", "yes"):
            val = True
        elif val_lower in ("false", "no"):
            val = False
        elif val_lower in ("none", "null"):
            val = None
        else:
            # Intentionally keep the raw string and let Pydantic's schema
            # coerce it to the declared field type. Hand-writing int/float
            # coercion here is fragile: a model literally named "1234" or
            # "3.14" would otherwise be silently cast to a numeric type and
            # only partially recovered by Pydantic, producing edge-case bugs.
            val = val_str

        keys = key_path.split(".")
        curr = raw
        for k in keys[:-1]:
            curr = curr.setdefault(k, {})
        curr[keys[-1]] = val


class DataConfig(BaseModel):
    # The built-in anonymized sample dataset (67 single-image category folders,
    # 45 questions) ships with the repo under data/sample_dataset/ so the
    # benchmark runs out-of-the-box without an external image directory.
    # Override either path via the BENCHMARK_IMAGE_DIR / BENCHMARK_QUESTIONS_FILE
    # env vars to evaluate on a full private dataset.
    image_dir: Path = Field(
        default_factory=lambda: Path(os.environ.get("BENCHMARK_IMAGE_DIR", "data/sample_dataset/images"))
    )
    questions_file: Path = Field(
        default_factory=lambda: Path(
            os.environ.get("BENCHMARK_QUESTIONS_FILE", "data/sample_dataset/questions.json")
        )
    )
    output_dir: Path = Field(default_factory=lambda: Path("data/results/eval"))
    # Optional dataset-scale metadata (e.g. unique-person counts). Replaces the
    # hardcoded magic path previously baked into Scorer._maybe_dimension_stats.
    # When None or pointing at a non-existent file, dimension stats are skipped.
    metadata_file: Optional[Path] = None


class ModelConfig(BaseModel):
    architecture: str = "vlm"
    device: str = "cuda"
    # Explicit display name for reports/CSV. Takes priority over the heuristic
    # name resolution in Inferencer._resolve_model_display_name, so a checkpoint
    # named "epoch_10_best.pt" no longer leaks into result tables.
    display_name: Optional[str] = None
    args: Dict[str, Any] = Field(default_factory=dict)


class EvalConfig(BaseModel):
    batch_size: int = 64
    num_workers: int = 8
    # Paper protocol: zero-shot, no role-playing system prompt. Model inputs
    # consist solely of the User Prompt (Question + Options) per Extended Data
    # Fig. 4. Override only when intentionally deviating from the paper.
    system_prompt: str = ""
    force_run: bool = False
    use_cache: bool = True
    # Optional cap on the number of questions to evaluate (smoke-test / API
    # cost control). When set, only the first N questions (after the
    # deterministic id sort in load_questions) are run; None = run all.
    max_questions: Optional[int] = None
    # Paper-aligned independent repair extractor (Extended Data Fig. 4).
    extractor_repair: bool = True
    # Max retries AFTER the first extractor attempt. (Total attempts = 1 + extractor_max_retries)
    extractor_max_retries: int = 1
    # Sleep between failed extractor attempts (seconds). Multiplied by attempt index.
    extractor_retry_backoff_s: float = 0.0


class DefaultsConfig(BaseModel):
    # Paper-aligned VLM list (Section 2.1, Figs. 3 & 6), evaluated via OpenRouter.
    # Baked in as a reproducibility default so `bprbench run --provider openrouter`
    # works out of the box; override per-run with --models.
    openrouter_models: list[str] = Field(
        default_factory=lambda: [
            "openai/gpt-5",
            "openai/gpt-4o",
            # "google/gemini-3-pro-preview",  # deprecated on OpenRouter
            "qwen/qwen3.5-397b-a17b",
            "qwen/qwen3-vl-32b-instruct",
            "meta-llama/llama-4-maverick",
            # "opengvlab/internvl3-78b",  # deprecated on OpenRouter
            "anthropic/claude-sonnet-4.5",
            # "x-ai/grok-4",  # deprecated on OpenRouter
            # "moonshotai/kimi-2.5",  # deprecated on OpenRouter
        ]
    )
    # Vendor pricing in USD per token, aligned with the models reported in the
    # paper (Section 2.1, Extended Data Table 2/3, Fig. 6). Keys are matched
    # case-insensitively as substrings of the model name recorded in predictions.
    pricing: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "qwen3-vl-32b-instruct": {"in": 0.0000001040, "out": 0.000000416},
            "qwen3.5-397b-a17b": {"in": 0.000000150, "out": 0.000001000},
            "internvl3-78b": {"in": 0.000000150, "out": 0.000000600},
            "llama-4-maverick": {"in": 0.000000150, "out": 0.000000600},
            "kimi-2.5": {"in": 0.000000230, "out": 0.000003000},
            "gemini-3-pro-preview": {"in": 0.000002000, "out": 0.000012000},
            "gpt-5": {"in": 0.000001250, "out": 0.000010000},
            "gpt-4o": {"in": 0.000002500, "out": 0.000010000},
            "grok-4": {"in": 0.000003000, "out": 0.000015000},
            "claude-sonnet-4.5": {"in": 0.000003000, "out": 0.000015000},
            "quilt-llava": {"in": 0.0, "out": 0.0},  # local, self-hosted
        }
    )


class BenchmarkConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: Optional[List[str]] = None,
    ) -> "BenchmarkConfig":
        raw: dict = {}
        if path:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"Config not found: {p}")
            suffix = p.suffix.lower()
            if suffix not in {".yaml", ".yml"}:
                raise ValueError(f"Unsupported config format: {suffix}. Use .yaml/.yml")
            text = p.read_text(encoding="utf-8")
            loaded = yaml.safe_load(text)
            if loaded is not None and not isinstance(loaded, dict):
                raise ValueError("Config root must be an object")
            raw = loaded or {}

        if overrides:
            _apply_overrides(raw, overrides)

        raw = _expand_env_vars(raw)
        instance = cls(**raw)

        # Resolve relative paths from CWD (project root when run via scripts),
        # not from config file dir, so "data/results/..." goes to repo data/results/.
        base_dir = Path.cwd()
        for key in ("image_dir", "questions_file", "output_dir", "metadata_file"):
            val = getattr(instance.data, key, None)
            if val is None:
                continue
            if not val.is_absolute():
                setattr(instance.data, key, (base_dir / val).resolve())

        return instance


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    return value
