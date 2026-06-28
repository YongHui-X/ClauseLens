"""Helpers for turning raw CUAD files into clean QFind records.

This file does not do embedding or vector search. Its job is data preparation:
read CUAD's CSV/text files, extract useful clause evidence, and write a simpler
JSONL dataset that the RAG ingestion step can embed later.
"""

from __future__ import annotations

import ast
import csv
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# A constant list: these are the first clause categories we want to support.
# The ALL_CAPS name is a Python convention for "this value should not change".
STARTER_CLAUSE_TYPES = [
    "Anti-Assignment",
    "Cap On Liability",
    "License Grant",
    "Audit Rights",
    "Termination For Convenience",
]


@dataclass(frozen=True)
class CuadPaths:
    """Central place for CUAD file paths.

    @dataclass automatically creates an __init__ method for this class.
    frozen=True means instances are read-only after creation, so code cannot
    accidentally change paths.root halfway through a run.
    """

    # The syntax "root: Path" is a type hint. It says root should be a Path.
    # Type hints help editors and readers, but Python does not enforce them by
    # default at runtime.
    root: Path = Path("data/cuad/CUAD_v1")

    @property
    def master_clauses_csv(self) -> Path:
        # @property lets callers write paths.master_clauses_csv instead of
        # paths.master_clauses_csv(). The "-> Path" means this method returns
        # a pathlib.Path object.
        #
        # The "/" operator is overloaded by Path. It joins path pieces safely,
        # like "data/cuad/CUAD_v1" / "master_clauses.csv".
        return self.root / "master_clauses.csv"

    @property
    def text_dirs(self) -> tuple[Path, Path]:
        # tuple[Path, Path] means this returns exactly two Path objects.
        return (
            self.root / "full_contract_txt" / "Part_I",
            self.root / "full_contract_txt" / "Part_II",
        )


def filename_stem(filename: str) -> str:
    # "filename: str" means the parameter should be a string.
    # "-> str" means the function returns a string.
    # Path(...).stem removes the file extension:
    # "Example Agreement.pdf" becomes "Example Agreement".
    return Path(filename).stem


def load_text_file_lookup(text_dirs: Iterable[Path]) -> dict[str, Path]:
    # Iterable[Path] means any loopable collection of Path objects is accepted:
    # a list, tuple, generator, etc.
    #
    # dict[str, Path] means the return value is a dictionary where keys are
    # strings and values are Path objects.
    lookup: dict[str, Path] = {}
    for text_dir in text_dirs:
        if not text_dir.exists():
            # continue skips this loop iteration and moves to the next text_dir.
            continue

        # glob("*.txt") finds all .txt files directly inside this folder.
        for text_file in text_dir.glob("*.txt"):
            # Store a mapping from filename-without-extension to the full path.
            # This lets us match "Example.pdf" from the CSV to "Example.txt".
            lookup[text_file.stem] = text_file
    return lookup


def load_master_rows(csv_path: Path) -> list[dict[str, str]]:
    # list[dict[str, str]] means a list of dictionaries. Each dictionary maps
    # CSV column names (str) to cell values (str).
    #
    # "with ... as file" is a context manager. It closes the file automatically,
    # even if an error occurs while reading.
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        # csv.DictReader returns each CSV row as a dictionary keyed by column.
        return list(csv.DictReader(file))


def parse_evidence_list(raw_value: str | None) -> list[str]:
    # "str | None" means raw_value can be either a string or None.
    # This is newer Python syntax for Optional[str].
    #
    # CUAD stores clause evidence in CSV cells that often look like Python lists,
    # for example: "['first clause', 'second clause']". This function converts
    # those cells into a normal list[str].
    if not raw_value:
        return []

    value = raw_value.strip()
    if not value or value == "[]":
        return []

    try:
        # ast.literal_eval safely parses Python literals such as lists/strings.
        # It is safer than eval because it does not execute arbitrary code.
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        # If the cell is not a valid Python literal, keep it as one text item.
        return [value]

    if isinstance(parsed, list):
        # This list comprehension strips every item and drops empty strings.
        return [str(item).strip() for item in parsed if str(item).strip()]

    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]

    return []


def answer_column_for(clause_type: str, row: dict[str, str]) -> str | None:
    # Some CUAD columns have inconsistent spacing in their names. This function
    # checks both possible answer-column spellings and returns the one present.
    candidates = [f"{clause_type}-Answer", f"{clause_type}- Answer"]
    for candidate in candidates:
        if candidate in row:
            return candidate
    return None


def matching_labeled_rows(
    rows: Iterable[dict[str, str]],
    text_lookup: dict[str, Path],
) -> list[dict[str, str]]:
    # This keeps only CSV rows whose PDF filename has a matching .txt file.
    # The multi-line "return [...]" below is a list comprehension:
    # "give me every row in rows where this condition is true".
    return [
        row
        for row in rows
        if filename_stem(row["Filename"]) in text_lookup
    ]


def build_clause_evidence_records(
    rows: Iterable[dict[str, str]],
    text_lookup: dict[str, Path],
    clause_types: Iterable[str],
) -> list[dict[str, object]]:
    # dict[str, object] means keys are strings and values can be mixed types.
    # We use object because fields include strings now, and future metadata may
    # include numbers, booleans, or lists.
    records: list[dict[str, object]] = []

    for row in rows:
        document_stem = filename_stem(row["Filename"])

        # dict.get(key) returns the value if present, otherwise None.
        # That is why we check "if text_path is None" below.
        text_path = text_lookup.get(document_stem)
        if text_path is None:
            continue

        for clause_type in clause_types:
            # row.get(clause_type) fetches the CSV cell for this clause category.
            # If the column is missing, .get returns None instead of crashing.
            evidence_items = parse_evidence_list(row.get(clause_type))
            if not evidence_items:
                continue

            answer_column = answer_column_for(clause_type, row)

            # Python's conditional expression:
            # value_if_true if condition else value_if_false
            answer = row.get(answer_column, "") if answer_column else ""

            # enumerate(..., start=1) gives both a counter and each item.
            # Here index is 1, 2, 3... and text is the evidence clause.
            for index, text in enumerate(evidence_items, start=1):
                records.append(
                    {
                        # f"...{value}..." is an f-string. It inserts variable
                        # values into the string.
                        "id": f"{document_stem}::{clause_type}::{index}",
                        "document_id": document_stem,
                        "source_pdf": row["Filename"],
                        "source_txt": str(text_path),
                        "clause_type": clause_type,
                        "answer": answer,
                        "text": text,
                    }
                )

    return records


def select_starter_records(
    records: list[dict[str, object]],
    max_contracts: int,
) -> list[dict[str, object]]:
    # Counter counts how many records each document_id has.
    document_scores = Counter(str(record["document_id"]) for record in records)

    # This set comprehension keeps the document IDs for the top N documents.
    # Sets are useful for fast membership checks with "in".
    selected_documents = {
        document_id
        for document_id, _ in document_scores.most_common(max_contracts)
    }

    return [
        record
        for record in records
        if str(record["document_id"]) in selected_documents
    ]


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    # "-> None" means this function is used for its side effect: writing a file.
    # It does not return a useful value.
    path.parent.mkdir(parents=True, exist_ok=True)

    # JSONL means "one JSON object per line". It is convenient for datasets
    # because you can stream/read one record at a time.
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_records(records: Iterable[dict[str, object]]) -> dict[str, object]:
    # Convert to list so we can loop over the records more than once.
    record_list = list(records)

    clause_counts = Counter(str(record["clause_type"]) for record in record_list)

    # The {... for ... in ...} syntax creates a set. Because sets remove
    # duplicates, len(...) gives the number of unique document IDs.
    document_count = len({str(record["document_id"]) for record in record_list})

    return {
        "documents": document_count,
        "clause_evidence_records": len(record_list),
        "clause_counts": dict(sorted(clause_counts.items())),
    }
