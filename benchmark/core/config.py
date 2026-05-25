from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    image_dir: Path
    questions_file: Path
    output_dir: Path


class ModelConfig(BaseModel):
    architecture: str
    device: str = "cuda"
    args: Dict[str, Any] = Field(default_factory=dict)


class EvalConfig(BaseModel):
    batch_size: int = 64
    num_workers: int = 8
    system_prompt: str = "You are a professional pathology expert."
    positive_label: str = "A"
    force_run: bool = False
    use_cache: bool = True
    # When no logprobs are available: try to "repair" free-form output via a text-only extractor.
    extractor_repair: bool = True
    # Max retries AFTER the first extractor attempt. (Total attempts = 1 + extractor_max_retries)
    extractor_max_retries: int = 1
    # Sleep between failed extractor attempts (seconds). Multiplied by attempt index.
    extractor_retry_backoff_s: float = 0.0


class DefaultsConfig(BaseModel):
    openrouter_models: list[str] = Field(default_factory=list)


class BenchmarkConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    eval: EvalConfig = Field(default_factory=EvalConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkConfig":
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in {".yaml", ".yml"}:
            raise ValueError(f"Unsupported config format: {suffix}. Use .yaml/.yml")

        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text)

        if not isinstance(raw, dict):
            raise ValueError("Config root must be an object")

        raw = _expand_env_vars(raw)

        data = raw.get("data")
        if isinstance(data, dict):
            # Resolve relative paths from CWD (project root when run via scripts),
            # not from config file dir, so "data/results/..." goes to repo data/results/
            base_dir = Path.cwd()
            for key in ("image_dir", "questions_file", "output_dir"):
                value = data.get(key)
                if isinstance(value, str):
                    p = Path(value).expanduser()
                    if not p.is_absolute():
                        data[key] = str((base_dir / p).resolve())

        return cls(**raw)


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    return value
