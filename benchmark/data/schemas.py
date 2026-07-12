from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator


class Option(BaseModel):
    label: str
    text: str
    folders: List[str] = Field(default_factory=list)
    classnames: List[str] = Field(default_factory=list)

    @field_validator("label")
    @classmethod
    def normalize_label(cls, v: str) -> str:
        out = str(v).strip().upper()
        if not out:
            raise ValueError("Option label must not be empty")
        return out

    @field_validator("text")
    @classmethod
    def normalize_text(cls, v: str) -> str:
        out = str(v).strip()
        if not out:
            raise ValueError("Option text must not be empty")
        return out


class Question(BaseModel):
    id: int
    question: str
    correct_type: Literal["max", "min"] = "max"
    tags: List[str] = Field(default_factory=lambda: ["default"])
    options: List[Option]
    # Raw synonym dictionary from JSON, used for classname expansion at load time.
    prompt: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, v: str) -> str:
        out = str(v).strip()
        if not out:
            raise ValueError("Question text must not be empty")
        return out

    def valid_labels(self) -> List[str]:
        return [o.label for o in self.options]
