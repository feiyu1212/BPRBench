from benchmark.models.base import BaseInference, PredictionRecord

# VLM registration only; CLIP adapters import torch and load on demand.
from benchmark.models.vlm_impl import VLMAdapter  # noqa: F401

_clip_models_loaded = False


def ensure_clip_models_registered() -> None:
    """Import CLIP adapters so their @register_model decorators run."""
    global _clip_models_loaded
    if _clip_models_loaded:
        return
    from benchmark.models import clip_impl  # noqa: F401

    _clip_models_loaded = True


__all__ = [
    "BaseInference",
    "PredictionRecord",
    "VLMAdapter",
    "ensure_clip_models_registered",
]
