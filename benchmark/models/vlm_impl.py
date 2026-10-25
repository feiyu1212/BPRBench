from __future__ import annotations

import base64
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

from benchmark.engine.post_processors import OptionLetterRepairAgent
from benchmark.core.logging import get_logger
from benchmark.core.registry import register_model
from benchmark.models.base import BaseInference, PredictionRecord
from benchmark.utils.parsers import parse_option_letter
from benchmark.utils.io import truncate_for_log

log = get_logger(__name__)

# Paper (Extended Data Fig. 4): if a response cannot be parsed, the evaluated
# model is queried again using the same prompt, for up to three attempts in
# total. Only after all three attempts fail is GPT-5 used as the auxiliary
# extractor on the final response.
PAPER_MAX_QUERY_ATTEMPTS = 3


def _to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _build_prompt(question: str, options: List[Dict[str, Any]]) -> str:
    """
    Prompt template aligned with the paper (Extended Data Fig. 4).

    Schema: {"option_letter":"one of the letters shown in Options"}
    """
    option_lines = [
        f"{str(o.get('label', '')).strip().upper()}. {str(o.get('text', '')).strip()}"
        for o in options
    ]
    option_text = "\n".join(option_lines)
    # Verbatim layout of the paper's prompt (Extended Data Fig. 4). The
    # "Input Image: {pathology image}" line maps to the attached image content
    # parts, which are sent ahead of this text (see _call_api).
    return (
        f"Question: {question}\n"
        f"Options:\n{option_text}\n\n"
        "Please analyze the pathology image and answer the question.\n"
        "Respond with JSON ONLY (no explanation, no code fences).\n\n"
        'Schema:\n{"option_letter":"one of the letters shown in Options"}\n\n'
        'Output Example:\n{"option_letter": "B"}'
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
        # Independent extractor configuration. Defaults to GPT-5 as the paper's
        # auxiliary extractor, overridable per-config for users without GPT-5 access.
        extractor_model: str = "gpt-5",
        extractor_api_key: str | None = None,
        extractor_base_url: str | None = None,
        extractor_timeout: int = 60,
        extractor_max_retries: int = 3,
        extractor_retry_backoff_s: float = 1.5,
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
            raise RuntimeError(
                f"Unsupported provider '{self.provider}'. Use 'openrouter' or 'local'."
            )

        # Pure dependency injection: no os.environ sniffing here. The CLI /
        # config layer is the single source of truth for credentials and
        # endpoints, and must have already populated api_key / base_url
        # (via _apply_model_env_defaults in cli.py or explicit YAML config).
        if self.provider == "openrouter":
            if not api_key:
                raise RuntimeError(
                    "OpenRouter provider requires an explicitly configured "
                    "api_key (set model.args.api_key or the OPENROUTER_API_KEY "
                    "env var before launching the CLI)."
                )
            router_base_url = base_url or "https://openrouter.ai/api/v1"
            self.client = OpenAI(api_key=api_key, base_url=router_base_url)
        else:
            # Local/Self-hosted OpenAI-compatible endpoint.
            self.client = OpenAI(api_key=api_key or "empty", base_url=base_url)
            log.info(
                "Initialized VLMAdapter (provider=%s, model=%s, url=%s)",
                self.provider,
                self.model_name,
                base_url,
            )

        # Unified, independent repair extractor (paper: GPT-5 as auxiliary extractor).
        # It owns its own client so the model under test is never asked to repair itself.
        self.option_repair_agent = OptionLetterRepairAgent(
            api_key=extractor_api_key,
            model_name=extractor_model,
            base_url=extractor_base_url,
            timeout=extractor_timeout,
            max_retries=extractor_max_retries,
            retry_backoff_s=extractor_retry_backoff_s,
        )

    def _call_api(
        self,
        system_prompt: str,
        prompt: str,
        image_paths: List[Path],
    ) -> Tuple[str, int, int, float]:
        """
        Returns: (content, prompt_tokens, completion_tokens, latency_s)
        """
        # Paper layout: the pathology image comes first (the "Input Image:"
        # line of the prompt template), followed by the Question/Options text.
        content: List[dict] = [
            {
                "type": "image_url",
                "image_url": {"url": _to_data_url(p)},
            }
            for p in image_paths
        ]
        content.append({"type": "text", "text": prompt})

        messages = []
        # Omit the system role entirely when no system prompt is configured
        # (empty or "N/A"), so the model only sees the paper's User Prompt
        # (Question + Options) per Extended Data Fig. 4.
        sys_p = (system_prompt or "").strip()
        if sys_p and sys_p.upper() != "N/A":
            messages.append({"role": "system", "content": sys_p})
        messages.append({"role": "user", "content": content})

        last_error: Exception | None = None
        log_name = image_paths[0].name if image_paths else "unknown"
        for attempt in range(1, self.max_retries + 1):
            start_time = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    # No hardcoded overrides: temperature is fully controlled
                    # by config (defaults to 0.0 for greedy/deterministic decoding).
                    temperature=self.temperature,
                    timeout=self.timeout,
                )
                latency = time.time() - start_time

                # Capture token usage for cost/efficiency reporting.
                prompt_tokens = 0
                completion_tokens = 0
                if resp.usage is not None:
                    prompt_tokens = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
                    completion_tokens = int(
                        getattr(resp.usage, "completion_tokens", 0) or 0
                    )

                content_str = resp.choices[0].message.content or ""
                # Print the model's raw answer to the original question as each
                # API call lands, so a long API-based run is observable instead
                # of a silent wait. Truncated to one scannable line; the full
                # text is preserved in the PredictionRecord / CSV downstream.
                log.info(
                    "VLM response | model=%s sample=%s latency=%.2fs raw=%s",
                    self.model_name,
                    log_name,
                    latency,
                    truncate_for_log(content_str),
                )
                return (
                    content_str,
                    prompt_tokens,
                    completion_tokens,
                    latency,
                )
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

    def _predict_single(
        self,
        system_prompt: str,
        prompt: str,
        valid_labels: List[str],
        sample_paths: List[Path],
        extractor_repair: bool,
        extractor_max_retries: int,
        extractor_backoff_s: float,
    ) -> PredictionRecord:
        if len(sample_paths) > 1:
            identifier = str(sample_paths[0].parent)
        else:
            identifier = str(sample_paths[0])

        raw = ""
        pred = ""
        p_tok, c_tok, lat = 0, 0, 0.0
        try:
            # Paper protocol: query the model with the same prompt, re-query on
            # parse failure for up to three attempts in total, and only then
            # fall back to the GPT-5 auxiliary extractor on the final response.
            for attempt in range(1, PAPER_MAX_QUERY_ATTEMPTS + 1):
                a_raw, a_p, a_c, a_lat = self._call_api(system_prompt, prompt, sample_paths)
                # Token/latency accounting is cumulative across attempts: failed
                # attempts still consumed API budget and must not be dropped.
                raw = a_raw
                p_tok += a_p
                c_tok += a_c
                lat += a_lat
                pred = parse_option_letter(raw, valid_labels)
                if pred:
                    break
                if attempt < PAPER_MAX_QUERY_ATTEMPTS:
                    log.warning(
                        "Sample %s: attempt %s/%s unparseable; re-querying the model with the same prompt.",
                        identifier,
                        attempt,
                        PAPER_MAX_QUERY_ATTEMPTS,
                    )

            if (not pred) and extractor_repair:
                pred = (
                    self.option_repair_agent.repair(
                        raw,
                        valid_labels,
                        max_retries=extractor_max_retries,
                        backoff_s=extractor_backoff_s,
                    )
                    or ""
                )
        except Exception as exc:
            log.error("VLM inference failed for %s: %s", identifier, exc)
            if not raw:
                raw = f"Error: {str(exc)}"

        # Strict evaluation: no random fallback. A failure to produce a
        # valid option letter is recorded as INVALID and scored as wrong,
        # matching the paper's correctness criterion.
        if not pred:
            pred = "INVALID"
            raw = (raw or "") + " [FAILED_TO_PARSE]"
            log.warning(
                "Sample %s: no valid option letter after %s attempts; marked INVALID.",
                identifier,
                PAPER_MAX_QUERY_ATTEMPTS,
            )

        return PredictionRecord(
            sample_id=identifier,
            pred_label=pred,
            raw_output=raw,
            input_tokens=p_tok,
            output_tokens=c_tok,
            latency_s=lat,
        )

    def predict(
        self,
        samples: List[List[Path]],
        question_text: str,
        options: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> List[PredictionRecord]:
        if not samples:
            return []

        workers = int(kwargs.get("num_workers", 4))
        system_prompt = str(kwargs.get("system_prompt", "")).strip()
        prompt = _build_prompt(question_text, options)
        valid_labels = [
            str(o.get("label", "")).strip().upper()
            for o in options
            if str(o.get("label", "")).strip()
        ]
        extractor_repair = bool(kwargs.get("extractor_repair", True))
        extractor_max_retries = int(kwargs.get("extractor_max_retries", 1))
        extractor_backoff_s = float(kwargs.get("extractor_retry_backoff_s", 0.0))

        rows: List[PredictionRecord] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_sample = {
                executor.submit(
                    self._predict_single,
                    system_prompt,
                    prompt,
                    valid_labels,
                    s,
                    extractor_repair,
                    extractor_max_retries,
                    extractor_backoff_s,
                ): s
                for s in samples
            }
            for future in as_completed(future_to_sample):
                rows.append(future.result())

        return rows
