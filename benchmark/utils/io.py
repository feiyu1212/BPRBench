from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def truncate_for_log(text: Any, max_len: int = 200) -> str:
    """Compact, single-line representation of an LLM response for log lines.

    Collapses internal whitespace (newlines/tabs) so a multi-line JSON or
    reasoning trace stays on one scannable line, and truncates to ``max_len``
    chars with an ellipsis tail showing how much was elided. Used by the
    intermediate LLM-output logs so long API-based runs stay observable
    without flooding the terminal.
    """
    s = "" if text is None else str(text)
    s = " ".join(s.split())
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"...<+{len(s) - max_len} chars>"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any, indent: int = 2) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
