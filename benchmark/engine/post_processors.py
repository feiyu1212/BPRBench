from __future__ import annotations

import time
from typing import Callable, Iterable, List

from benchmark.utils.parsers import parse_option_letter

TextRepairCall = Callable[[str, str], str]


def _normalize_labels(valid_labels: Iterable[str]) -> List[str]:
    labels: List[str] = []
    seen = set()
    for label in valid_labels:
        normalized = str(label).strip().upper()
        if not normalized or normalized in seen:
            continue
        labels.append(normalized)
        seen.add(normalized)
    return labels


class OptionLetterRepairAgent:
    """
    Generic post-processor for extracting MCQ option letters from noisy model output.
    The text-repair backend is injected to keep this logic model-agnostic.
    """

    def __init__(
        self,
        text_repair_call: TextRepairCall,
        system_prompt: str = "You are a strict answer extractor.",
    ):
        self.text_repair_call = text_repair_call
        self.system_prompt = system_prompt

    def repair(
        self,
        raw_answer: str,
        valid_labels: Iterable[str],
        max_retries: int = 1,
        backoff_s: float = 0.0,
    ) -> str:
        labels = _normalize_labels(valid_labels)
        if not raw_answer.strip() or not labels:
            return ""

        label_text = ", ".join(labels)
        raw_short = raw_answer.strip()[-2500:]
        strategies = [
            (
                f"Valid options: [{label_text}]. Extract option letter from OUTPUT. "
                "Return JSON: {\"option_letter\":\"A\"}\n\n"
                f"OUTPUT:\n{raw_short}"
            ),
            (
                f"Valid options: [{label_text}]. Return ONLY the single letter "
                "(e.g. A) from OUTPUT.\n\n"
                f"OUTPUT:\n{raw_short}"
            ),
        ]

        attempts = 1 + max(0, int(max_retries))
        for i in range(attempts):
            prompt = strategies[min(i, len(strategies) - 1)]
            repaired_raw = self.text_repair_call(self.system_prompt, prompt)
            pred = parse_option_letter(repaired_raw, labels)
            if pred:
                return pred
            if backoff_s > 0:
                time.sleep(float(backoff_s) * (i + 1))
        return ""
