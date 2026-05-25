from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from benchmark.utils.io import read_json, write_json


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]


def convert(input_path: Path, output_path: Path) -> None:
    raw = read_json(input_path)
    if not isinstance(raw, list):
        raise ValueError("Input questions JSON must be a list.")

    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        qid = int(item.get("id"))
        question = str(item.get("question") or item.get("text") or item.get("question_text") or "").strip()
        if not question:
            continue

        correct_type = str(item.get("correct_type", "max")).strip().lower()
        if correct_type not in ("max", "min"):
            correct_type = "max"

        tags = _as_list(item.get("tags")) or ["default"]

        options: List[Dict[str, Any]] = []
        for opt in item.get("options", []):
            if not isinstance(opt, dict):
                continue
            label = str(opt.get("label", "")).strip().upper()
            text = str(opt.get("text", "")).strip()
            folders = _as_list(opt.get("folders"))
            classnames = _as_list(opt.get("classnames") or opt.get("prompt_list"))
            if not label or not text:
                continue
            options.append(
                {
                    "label": label,
                    "text": text,
                    "folders": folders,
                    "classnames": classnames,
                }
            )

        if not options:
            continue

        out.append(
            {
                "id": qid,
                "question": question,
                "correct_type": correct_type,
                "tags": tags,
                "options": options,
            }
        )

    out.sort(key=lambda x: x["id"])
    write_json(output_path, out, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser("convert_questions")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    convert(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
