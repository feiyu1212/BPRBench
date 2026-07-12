from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from benchmark.utils.io import ensure_dir


@dataclass
class JsonlCache:
    """
    Minimal JSONL cache: key -> value
    - Load all data into memory at startup
    - Append line on new key write
    """

    path: Path
    _mem: Dict[str, Any]

    @classmethod
    def open(cls, path: Path) -> "JsonlCache":
        ensure_dir(path.parent)
        mem: Dict[str, Any] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        mem[obj["key"]] = obj["value"]
                    except Exception:
                        continue
        return cls(path=path, _mem=mem)

    def get(self, key: str) -> Optional[Any]:
        return self._mem.get(key)

    def set(self, key: str, value: Any) -> None:
        if key in self._mem:
            return
        self._mem[key] = value
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
