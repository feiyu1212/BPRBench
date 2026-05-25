from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from benchmark.data.schemas import Option, Question
from benchmark.utils.io import read_json


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    v = str(value).strip()
    return [v] if v else []


def _expand_synonyms(name: str, prompt_map: Dict[str, Any]) -> List[str]:
    """
    Look up the classname in the prompt_map and flatten the synonyms.
    Handles the structure: {"Key": {"Key": ["synonym1", "synonym2"]}}
    Performs case-insensitive lookup to handle mismatches like "calcification" vs "Calcification".
    """
    if not name:
        return [name]
    
    # First try exact match (fast path)
    if name in prompt_map:
        entry = prompt_map[name]
    else:
        # Try case-insensitive lookup
        name_lower = name.lower()
        matched_key = None
        for key in prompt_map.keys():
            if key.lower() == name_lower:
                matched_key = key
                break
        
        if matched_key is None:
            print(f"Warning: {name} not found in prompt_map (case-insensitive search also failed)")
            return [name]
        
        entry = prompt_map[matched_key]
    synonyms: List[str] = []

    # Handle nested dict structure from JSON (e.g. prompt -> Carcinoma -> Carcinoma -> [list])
    if isinstance(entry, dict):
        for val in entry.values():
            if isinstance(val, list):
                synonyms.extend([str(v) for v in val if str(v).strip()])
            elif isinstance(val, str) and val.strip():
                synonyms.append(val.strip())
    # Handle direct list structure
    elif isinstance(entry, list):
        synonyms.extend([str(v) for v in entry if str(v).strip()])

    # If expansion yielded nothing, return original name
    return sorted(list(set(synonyms))) if synonyms else [name]


def load_questions(path: Path) -> List[Question]:
    raw = read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Questions file must be a list, got {type(raw)}")

    questions: List[Question] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        qid = int(item.get("id"))
        qtext = str(item.get("question") or item.get("text") or item.get("question_text") or "").strip()
        correct_type = str(item.get("correct_type") or "max").strip().lower()
        if correct_type not in {"max", "min"}:
            correct_type = "max"

        tags = _as_list(item.get("tags")) or ["default"]
        
        # FIX: Extract the prompt dictionary to use for expansion
        prompt_data = item.get("prompt", {})

        options: List[Option] = []
        for opt in item.get("options", []):
            if not isinstance(opt, dict):
                continue
            
            # 1. Get initial classnames
            raw_classnames = _as_list(opt.get("classnames") or opt.get("prompt_list"))
            if not raw_classnames:
                # Fallback to text if no classnames provided
                raw_classnames = _as_list(opt.get("text", ""))

            # 2. FIX: Expand classnames using the prompt_data lookup
            expanded_classnames: List[str] = []
            for name in raw_classnames:
                expanded_classnames.extend(_expand_synonyms(name, prompt_data))
            
            # Deduplicate while preserving order roughly
            seen = set()
            final_classnames = []
            for x in expanded_classnames:
                if x not in seen:
                    final_classnames.append(x)
                    seen.add(x)

            payload = {
                "label": opt.get("label", ""),
                "text": opt.get("text", ""),
                "folders": _as_list(opt.get("folders")),
                "classnames": final_classnames, # FIX: Pass the expanded list here
            }
            options.append(Option.model_validate(payload))

        payload = {
            "id": qid,
            "question": qtext,
            "correct_type": correct_type,
            "tags": tags,
            "options": [o.model_dump() for o in options],
            "prompt": prompt_data, # FIX: Store raw prompt data
        }
        questions.append(Question.model_validate(payload))

    questions.sort(key=lambda x: x.id)
    return questions