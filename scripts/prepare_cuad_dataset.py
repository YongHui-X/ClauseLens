from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_CUAD_ROOT = Path("data/cuad/CUAD_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and repair the CUAD v1 dataset layout."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_CUAD_ROOT,
        help="Path to the extracted CUAD_v1 directory.",
    )
    return parser.parse_args()


def load_master_stems(master_clauses_csv: Path) -> set[str]:
    with master_clauses_csv.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            Path(row["Filename"]).stem.strip()
            for row in csv.DictReader(file)
            if row.get("Filename")
        }


def document_stem(document: dict[str, object]) -> str:
    title = str(document.get("title", "")).strip()
    return Path(title).stem if Path(title).suffix else title


def load_json_contexts(cuad_json: Path) -> dict[str, str]:
    payload = json.loads(cuad_json.read_text(encoding="utf-8"))
    contexts_by_stem: dict[str, str] = {}

    for document in payload.get("data", []):
        stem = document_stem(document)
        if not stem:
            continue

        contexts = [
            str(paragraph.get("context", "")).strip()
            for paragraph in document.get("paragraphs", [])
            if str(paragraph.get("context", "")).strip()
        ]
        if contexts:
            contexts_by_stem[stem] = "\n\n".join(contexts)

    return contexts_by_stem


def text_dirs(cuad_root: Path) -> tuple[Path, Path]:
    text_root = cuad_root / "full_contract_txt"
    return text_root / "Part_I", text_root / "Part_II"


def matched_text_count(cuad_root: Path, master_stems: set[str]) -> int:
    count = 0
    for directory in text_dirs(cuad_root):
        if not directory.exists():
            continue
        for text_file in directory.glob("*.txt"):
            if text_file.stem in master_stems:
                count += 1
    return count


def repair_cuad_dataset(cuad_root: Path) -> int:
    master_clauses_csv = cuad_root / "master_clauses.csv"
    cuad_json = cuad_root / "CUAD_v1.json"
    required_files = [master_clauses_csv, cuad_json]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "CUAD dataset is missing required files: " + ", ".join(missing_files)
        )

    master_stems = load_master_stems(master_clauses_csv)
    part_i, part_ii = text_dirs(cuad_root)
    needs_repair = (
        not part_i.exists()
        or not part_ii.exists()
        or matched_text_count(cuad_root, master_stems) == 0
    )

    part_i.mkdir(parents=True, exist_ok=True)
    part_ii.mkdir(parents=True, exist_ok=True)

    if not needs_repair:
        return 0

    contexts_by_stem = load_json_contexts(cuad_json)
    written = 0
    for stem in sorted(master_stems):
        text = contexts_by_stem.get(stem)
        if not text:
            continue

        output_path = part_i / f"{stem}.txt"
        if output_path.exists():
            continue

        output_path.write_text(text, encoding="utf-8")
        written += 1

    return written


def validate_cuad_dataset(cuad_root: Path) -> None:
    master_clauses_csv = cuad_root / "master_clauses.csv"
    part_i, part_ii = text_dirs(cuad_root)
    required_paths = [
        master_clauses_csv,
        cuad_root / "CUAD_v1.json",
        part_i,
        part_ii,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        extracted = []
        if cuad_root.exists():
            extracted = sorted(
                str(path.relative_to(cuad_root))
                for path in cuad_root.rglob("*")
                if path.is_dir() or path.name in {"master_clauses.csv", "CUAD_v1.json"}
            )
        raise RuntimeError(
            "CUAD extraction missing required paths: "
            f"{missing}. Extracted paths: {extracted[:40]}"
        )

    master_stems = load_master_stems(master_clauses_csv)
    if matched_text_count(cuad_root, master_stems) == 0:
        raise RuntimeError("CUAD dataset has no text files matching master_clauses.csv.")


def main() -> None:
    args = parse_args()
    written = repair_cuad_dataset(args.root)
    validate_cuad_dataset(args.root)
    if written:
        print(f"Generated {written} CUAD text files from CUAD_v1.json.")
    print("CUAD v1 dataset layout validated.")


if __name__ == "__main__":
    main()
