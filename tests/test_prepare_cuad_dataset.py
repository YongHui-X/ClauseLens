import csv
import json

from scripts.prepare_cuad_dataset import repair_cuad_dataset, validate_cuad_dataset


def test_repair_cuad_dataset_creates_text_layout_from_json(tmp_path) -> None:
    cuad_root = tmp_path / "CUAD_v1"
    cuad_root.mkdir()

    with (cuad_root / "master_clauses.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["Filename"])
        writer.writeheader()
        writer.writerow({"Filename": "Example Agreement.pdf"})

    (cuad_root / "CUAD_v1.json").write_text(
        json.dumps(
            {
                "data": [
                    {
                        "title": "Example Agreement",
                        "paragraphs": [
                            {"context": "The first paragraph."},
                            {"context": "The second paragraph."},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    written = repair_cuad_dataset(cuad_root)
    validate_cuad_dataset(cuad_root)

    part_i = cuad_root / "full_contract_txt" / "Part_I"
    part_ii = cuad_root / "full_contract_txt" / "Part_II"
    text_file = part_i / "Example Agreement.txt"

    assert written == 1
    assert part_i.is_dir()
    assert part_ii.is_dir()
    assert text_file.read_text(encoding="utf-8") == (
        "The first paragraph.\n\nThe second paragraph."
    )
