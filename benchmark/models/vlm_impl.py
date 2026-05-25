from __future__ import annotations

import base64
import mimetypes
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Literal

from benchmark.engine.post_processors import OptionLetterRepairAgent
from benchmark.core.logging import get_logger
from benchmark.core.registry import register_model
from benchmark.models.base import BaseInference, Prediction
from benchmark.utils.parsers import parse_option_letter

log = get_logger(__name__)


def _to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _build_prompt(question: str, options: List[Dict[str, Any]]) -> str:
    option_lines = [f"{str(o.get('label', '')).strip().upper()}. {str(o.get('text', '')).strip()}" for o in options]
    option_text = "\n".join(option_lines)
    labels = [str(o.get("label", "")).strip().upper() for o in options if str(o.get("label", "")).strip()]
    label_str = ", ".join(labels) if labels else "A, B, C, D"
    return (
        f"Question: {question}\n"
        f"Options:\n{option_text}\n\n"
        "Please analyze the pathology image and answer the question. "
        f"Respond with JSON ONLY (no explanation, no code fences). "
        f"Schema: {{\"option_letter\":\"<ONE_OF [{label_str}]>\"}}.\n"
        "Example: {\"option_letter\":\"A\"}"
    )


@register_model("vlm")
class VLMAdapter(BaseInference):
    """
    Unified VLM adapter.
    Supports OpenRouter and local/self-hosted endpoints (via OpenAI-compatible API).
    """

    def __init__(
        self,
        device: str,
        model: str = "gpt-4o",
        provider: Literal["openrouter", "local"] = "openrouter",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
        temperature: float = 0.0,
        max_retries: int = 5,
        retry_backoff_s: float = 1.5,
        **kwargs: Any,
    ):
        super().__init__(device=device, **kwargs)

        from openai import OpenAI

        self.model_name = model
        self.provider = provider.lower()
        self.timeout = int(timeout)
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        self.retry_backoff_s = float(retry_backoff_s)

        if self.provider not in {"openrouter", "local"}:
            raise RuntimeError(f"Unsupported provider '{self.provider}'. Use 'openrouter' or 'local'.")

        if self.provider == "openrouter":
            key = api_key or os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OpenRouter provider requires api_key or OPENROUTER_API_KEY.")
            router_base_url = base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.client = OpenAI(api_key=key, base_url=router_base_url)
        else:
            # Local/Self-hosted OpenAI-compatible endpoint.
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                key = "empty"
            url = base_url or os.environ.get("OPENAI_BASE_URL")
            self.client = OpenAI(api_key=key, base_url=url)
            log.info(
                "Initialized VLMAdapter (provider=%s, model=%s, url=%s)",
                self.provider,
                self.model_name,
                url,
            )

        self.option_repair_agent = OptionLetterRepairAgent(text_repair_call=self._call_text_api)

    def _call_api(self, system_prompt: str, prompt: str, image_paths: List[Path]) -> str:
        content: List[dict] = [{"type": "text", "text": prompt}]
        for p in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": _to_data_url(p)},
            })

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        last_error: Exception | None = None
        log_name = image_paths[0].name if image_paths else "unknown"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature if 'gpt-5' not in self.model_name else 1, #  Only the default (1) value is supported for gpt-5
                    timeout=self.timeout,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                log.warning(
                    "VLM request failed (attempt %s/%s) sample=%s err=%s",
                    attempt,
                    self.max_retries,
                    log_name,
                    exc,
                )
                time.sleep(self.retry_backoff_s * attempt)

        raise RuntimeError(f"VLM request failed after retries: {last_error}")

    def _call_text_api(self, system_prompt: str, prompt: str) -> str:
        """Text-only call for answer extraction/repair."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    timeout=self.timeout,
                    max_tokens=64,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                log.warning("Text-only request failed: %s", exc)
                time.sleep(self.retry_backoff_s * attempt)
        return ""

    def predict(
        self,
        samples: List[List[Path]],
        question_text: str,
        options: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> List[Prediction]:
        if not samples:
            return []

        workers = int(kwargs.get("num_workers", 4))
        system_prompt = str(kwargs.get("system_prompt", "")).strip()
        prompt = _build_prompt(question_text, options)
        valid_labels = [str(o.get("label", "")).strip().upper() for o in options if str(o.get("label", "")).strip()]
        extractor_repair = bool(kwargs.get("extractor_repair", True))
        extractor_max_retries = int(kwargs.get("extractor_max_retries", 1))
        extractor_backoff_s = float(kwargs.get("extractor_retry_backoff_s", 0.0))

        rows: List[Prediction] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_sample = {
                executor.submit(self._call_api, system_prompt, prompt, s): s
                for s in samples
            }
            for future in as_completed(future_to_sample):
                sample_paths = future_to_sample[future]
                if len(sample_paths) > 1:
                    identifier = str(sample_paths[0].parent)
                else:
                    identifier = str(sample_paths[0])

                raw = ""
                pred = ""
                try:
                    raw = future.result()
                    pred = parse_option_letter(raw, valid_labels)
                    if (not pred) and extractor_repair:
                        pred = self.option_repair_agent.repair(
                            raw,
                            valid_labels,
                            max_retries=extractor_max_retries,
                            backoff_s=extractor_backoff_s,
                        ) or ""
                except Exception as exc:
                    log.error("VLM inference failed for %s: %s", identifier, exc)
                    if not raw:
                        raw = f"Error: {str(exc)}"

                # Fallback Strategy:
                # If prediction is missing (due to API failure or parsing failure),
                # randomly sample from valid labels to ensure the CSV is complete.
                if not pred and valid_labels:
                    pred = random.choice(valid_labels)
                    raw = (raw or "") + f" [FALLBACK: Randomly selected '{pred}' from {valid_labels} due to failure]"
                    log.warning("Sample %s: Prediction failed, fallback to random label %s.", identifier, pred)

                rows.append((identifier, pred, raw))

        return rows
