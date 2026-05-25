from __future__ import annotations

import ast
import json
import re
from typing import Iterable

_KV_LETTER_RE = re.compile(
    r"""(?ix)
    \b(option_letter|answer|choice|label)\b
    \s*[:=]\s*
    ["']?\s*([A-Za-z])\s*["']?
    """
)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    """
    Remove content before the closing </think> tag.
    Some VLMs output </think>...</think>actual_answer; we keep only the part after </think>.
    """
    if not text or "</think>" not in text:
        return text
    return text.split("</think>")[-1]


def _strip_code_fences(text: str) -> str:
    """
    Remove surrounding triple-backtick fences if present.
    Keeps inner content intact.
    """
    t = (text or "").strip()
    if t.startswith("```") and t.endswith("```"):
        t = _CODE_FENCE_RE.sub("", t).strip()
    return t


def _extract_first_balanced_object(text: str) -> str:
    """
    Extract the first balanced {...} JSON-like object, accounting for quoted strings.
    Returns "" if none found.
    """
    s = text or ""
    start = -1
    depth = 0
    in_str = False
    esc = False

    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return s[start : i + 1]
    return ""


def extract_json_object(text: str) -> dict:
    """
    Backward-compatible: returns the first dict-like JSON object found in the text.
    Strict by default (json.loads). Falls back to ast.literal_eval to handle single quotes.
    """
    t = _strip_code_fences(text)
    snippet = _extract_first_balanced_object(t)
    if not snippet:
        return {}

    # Strict JSON first
    try:
        obj = json.loads(snippet)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # Safe fallback for Python-literal dicts (single quotes, etc.)
    try:
        obj = ast.literal_eval(snippet)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def parse_option_letter(raw: str, valid_labels: Iterable[str]) -> str:
    """
    Strict MCQ extractor (mainstream approach when no logprobs):
    - Accept ONLY: exact single-letter OR JSON with known key OR explicit key-value pattern.
    - Never fall back to "any standalone capital letter" (avoids matching "I", "A common...", etc).
    Returns "" if invalid.
    """
    valid = [str(v).strip().upper() for v in valid_labels if str(v).strip()]
    valid_set = set(valid)
    if not raw or not valid_set:
        return ""

    text = _strip_think_tags(raw)
    print(text)
    text = _strip_code_fences(text).strip()
    if not text:
        return ""

    # 1) Exact single-letter answer (allow trailing punctuation like "A." / "A)")
    m = re.fullmatch(r"\s*([A-Za-z])\s*[\.\)\]]?\s*$", text)
    if m:
        c = m.group(1).upper()
        return c if c in valid_set else ""

    # 2) JSON object with a known key
    obj = extract_json_object(text)
    if obj:
        for key in ("option_letter", "answer", "choice", "label"):
            if key in obj:
                c = str(obj.get(key, "")).strip().upper()
                if c in valid_set:
                    return c

    # 3) Explicit key-value (anchored) pattern, e.g. "answer: B"
    m = _KV_LETTER_RE.search(text)
    if m:
        c = m.group(2).strip().upper()
        return c if c in valid_set else ""

    return ""
