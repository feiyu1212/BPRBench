from __future__ import annotations

from typing import Any, Dict, Type

from benchmark.core.config import ModelConfig
from benchmark.core.logging import get_logger

log = get_logger(__name__)

_MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(name: str):
    """Decorator-based model registration."""

    key = name.strip().lower()

    def decorator(cls):
        if key in _MODEL_REGISTRY:
            raise ValueError(f"Model '{key}' already registered.")
        _MODEL_REGISTRY[key] = cls
        return cls

    return decorator


def build_model(cfg: ModelConfig):
    key = cfg.architecture.strip().lower()
    cls = _MODEL_REGISTRY.get(key)
    if cls is None:
        available = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Architecture '{key}' not found. Available: [{available}]")

    log.info("Building model '%s' on device '%s'", key, cfg.device)
    return cls(device=cfg.device, **cfg.args)


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY.keys())
