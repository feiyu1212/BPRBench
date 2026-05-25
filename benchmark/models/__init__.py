from benchmark.models.base import BaseInference, Prediction

# Trigger registry entries.
from benchmark.models.clip_impl import (  # noqa: F401
    ConchAdapter,
    MuskAdapter,
    OpenCLIPAdapter,
    PathGenAdapter,
    QuiltNetAdapter,
    Vir2Adapter,
)
from benchmark.models.vlm_impl import VLMAdapter  # noqa: F401

__all__ = [
    "BaseInference",
    "Prediction",
    "OpenCLIPAdapter",
    "QuiltNetAdapter",
    "PathGenAdapter",
    "Vir2Adapter",
    "ConchAdapter",
    "MuskAdapter",
    "VLMAdapter",
]
