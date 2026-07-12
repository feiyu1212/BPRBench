from __future__ import annotations

import time
from typing import Iterable, List, Optional

from benchmark.core.logging import get_logger
from benchmark.utils.parsers import parse_option_letter

log = get_logger(__name__)


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
    Unified, independent format-repair extractor.

    Paper (Extended Data Fig. 4) states that GPT-5 is used as the auxiliary
    extractor when a response cannot be reliably parsed. This agent enforces
    that contract: it owns its own OpenAI-compatible client and always calls
    a single, fixed extractor model, regardless of which model produced the
    noisy output. This keeps the parsing standard fair and consistent across
    all evaluated models (no "self-repair" by the model under test).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-5",
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_backoff_s: float = 1.5,
        system_prompt: str = "You are a strict answer extractor.",
    ):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)
        self.retry_backoff_s = float(retry_backoff_s)

        # Pure dependency injection: this agent must NOT read os.environ.
        # All credential/url resolution is owned by the CLI/config layer and
        # passed in explicitly, so the component stays testable and the
        # "single source of truth" for config is preserved.
        self.client = None
        if not api_key:
            log.warning(
                "No API key provided for the independent extractor (model=%s). "
                "Repair agent will bypass and return empty strings.",
                self.model_name,
            )
            return

        try:
            from openai import OpenAI

            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = OpenAI(**kwargs)
        except Exception as exc:
            log.error("Failed to initialize extractor client: %s", exc)
            self.client = None

    def _call_extractor(self, prompt: str) -> str:
        if not self.client:
            return ""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,  # extractor must be deterministic
                    max_tokens=32,
                    timeout=self.timeout,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                log.warning(
                    "Extractor request failed (attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * attempt)
        return ""

    def repair(
        self,
        raw_answer: str,
        valid_labels: Iterable[str],
        max_retries: int = 1,
        backoff_s: float = 0.0,
    ) -> str:
        if not self.client:
            return ""

        labels = _normalize_labels(valid_labels)
        if not raw_answer.strip() or not labels:
            return ""

        label_text = ", ".join(labels)
        raw_full = raw_answer.strip()

        # Single, strict extraction prompt aligned with the paper
        # (Extended Data Fig. 4: "Respond with JSON ONLY", schema {"option_letter":"A"}).
        # No multi-strategy fallback, no truncation: the full raw output is
        # submitted to the auxiliary extractor.
        prompt = (
            f"Valid options: [{label_text}]. Extract the option letter from the OUTPUT.\n"
            "Respond with JSON ONLY (no explanation, no code fences).\n"
            'Schema: {"option_letter":"one of the letters shown in Valid options"}\n\n'
            f"OUTPUT:\n{raw_full}"
        )

        attempts = 1 + max(0, int(max_retries))
        for i in range(attempts):
            repaired_raw = self._call_extractor(prompt)
            pred = parse_option_letter(repaired_raw, labels)
            if pred:
                return pred
            if backoff_s > 0:
                time.sleep(float(backoff_s) * (i + 1))
        return ""
