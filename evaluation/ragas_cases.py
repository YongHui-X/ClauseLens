"""Load Ragas judge-based quality cases for QFind."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from evaluation.answer_cases import AnswerMessage

DEFAULT_RAGAS_CASE_FILE = Path(__file__).parent / "ragas_cases.jsonl"


class RagasCase(BaseModel):
    """One case for semantic RAG evaluation."""

    case_id: str = Field(min_length=1)
    messages: list[AnswerMessage] = Field(min_length=1)
    reference: str = Field(min_length=1)
    expected_clause_type: str | None
    critical: bool = False


def load_ragas_cases(path: str | Path = DEFAULT_RAGAS_CASE_FILE) -> list[RagasCase]:
    """Load Ragas cases from JSONL."""

    case_path = Path(path)
    cases: list[RagasCase] = []
    with case_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc
            cases.append(RagasCase(**data))
    return cases
