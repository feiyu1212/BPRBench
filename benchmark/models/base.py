from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

Prediction = Tuple[str, str, str]


class BaseInference(ABC):
    """Unified interface for all inference adapters."""

    def __init__(self, device: str, **_: Any):
        self.device = device

    @abstractmethod
    def predict(
        self,
        samples: List[List[Path]],  # Changed from image_paths: List[Path]
        question_text: str,
        options: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> List[Prediction]:
        raise NotImplementedError
