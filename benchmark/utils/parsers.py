from __future__ import annotations

import ast
import json
import re
from typing import Iterable

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
# Matches a flat (non-nested) object containing the "option_letter" key,
# aligned with the paper's schema {"option_letter":"X"}. Tolerates single or
# double quotes around the key (some open-source models emit Python literals).
_OPTION_OBJ_RE = re.compile(r"""\{[^{}]*["']option_letter["'][^{}]*\}""", re.IGNORECASE | re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Keep only the part after the closing </think> tag (DeepSeek-R1 style)."""
    if not text or "</think>" not in text:
        return text
    return text.split("</think>")[-1]


def _strip_code_fences(text: str) -> str:
    """Remove surrounding triple-backtick fences, keeping inner content intact."""
    t = (text or "").strip()
    if t.startswith("```") and t.endswith("```"):
        t = _CODE_FENCE_RE.sub("", t).strip()
    return t


def extract_json_object(text: str) -> dict:
    """
    Minimal, robust JSON object extractor.

    Fast path: json.loads on the stripped text. Targeted fallback: a regex
    that pulls out a flat object containing "option_letter", tolerant of
    single-quoted Python literals emitted by some open-source models.
    Returns the first dict-like object found, or {} if none can be parsed.
    """
    if not text:
        return {}

    candidate = _strip_code_fences(text).strip()

    def _parse(s: str) -> dict | None:
        # Try strict JSON first, then lenient Python literal (single quotes).
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(s)
            except (json.JSONDecodeError, SyntaxError, ValueError):
                continue
            if isinstance(obj, dict):
                return obj
        return None

    if (res := _parse(candidate)) is not None:
        return res

    match = _OPTION_OBJ_RE.search(candidate)
    if match and (res := _parse(match.group(0))) is not None:
        return res

    return {}


def parse_option_letter(raw: str, valid_labels: Iterable[str]) -> str:
    """
    Strict MCQ extractor aligned with the paper (Extended Data Fig. 4).

    Accepts ONLY a JSON object containing the 'option_letter' key. No
    single-letter regex, no alternative keys (answer/choice/label), no
    key-value pattern fallback. Anything that cannot be parsed this way
    is returned as "" so the caller can hand it to the GPT-5 repair agent.
    """
    valid = [str(v).strip().upper() for v in valid_labels if str(v).strip()]
    valid_set = set(valid)
    if not raw or not valid_set:
        return ""

    text = _strip_think_tags(raw)
    text = _strip_code_fences(text).strip()
    if not text:
        return ""

    obj = extract_json_object(text)
    if obj and "option_letter" in obj:
        c = str(obj.get("option_letter", "")).strip().upper()
        if c in valid_set:
            return c

    return ""
