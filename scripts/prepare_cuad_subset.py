from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.cuad import (  # noqa: E402
    STARTER_CLAUSE_TYPES,
    CuadPaths,
    build_clause_evidence_records,
    load_master_rows,
    load_text_file_lookup,
    matching_labeled_rows,
    select_starter_records,
    summarize_records,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a clean CUAD starter subset for QFind."
    )
    parser.add_argument(
        "--max-contracts",
        type=int,
        default=30,
        help="Maximum number of matched contracts to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/starter_clause_evidence.jsonl"),
        help="Path for the prepared JSONL evidence records.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/processed/starter_summary.json"),
        help="Path for the summary JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = CuadPaths()

    text_lookup = load_text_file_lookup(paths.text_dirs)
    master_rows = load_master_rows(paths.master_clauses_csv)
    labeled_rows = matching_labeled_rows(master_rows, text_lookup)

    all_records = build_clause_evidence_records(
        rows=labeled_rows,
        text_lookup=text_lookup,
        clause_types=STARTER_CLAUSE_TYPES,
    )
    starter_records = select_starter_records(
        records=all_records,
        max_contracts=args.max_contracts,
    )
    summary = summarize_records(starter_records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, starter_records)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote {summary['clause_evidence_records']} evidence records")
    print(f"Documents: {summary['documents']}")
    print(f"Output: {args.output}")
    print(f"Summary: {args.summary_output}")


if __name__ == "__main__":
    main()
