from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class PredictionRecord:
    """Standard data-transfer object flowing through the inference pipeline."""

    sample_id: str
    pred_label: str
    raw_output: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    @classmethod
    def from_cache(cls, sample_id: str, cached_val: Any, valid_labels: List[str] | None = None) -> "PredictionRecord":
        """Deserialize a cached value, tolerating legacy dict or plain-string entries."""
        if isinstance(cached_val, dict):
            return cls(
                sample_id=sample_id,
                pred_label=str(cached_val.get("pred", "")),
                raw_output=str(cached_val.get("raw", "")),
                input_tokens=int(cached_val.get("input_tokens", 0) or 0),
                output_tokens=int(cached_val.get("output_tokens", 0) or 0),
                latency_s=float(cached_val.get("latency", 0.0) or 0.0),
            )
        from benchmark.utils.parsers import parse_option_letter

        raw = str(cached_val)
        pred = parse_option_letter(raw, valid_labels or [])
        return cls(sample_id=sample_id, pred_label=pred, raw_output=raw)


class BaseInference(ABC):
    """Unified interface for all inference adapters."""

    def __init__(self, device: str, **_: Any):
        self.device = device

    @abstractmethod
    def predict(
        self,
        samples: List[List[Path]],
        question_text: str,
        options: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> List[PredictionRecord]:
        raise NotImplementedError
