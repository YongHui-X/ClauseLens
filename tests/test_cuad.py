from app.cuad import filename_stem, parse_evidence_list, select_starter_records


def test_filename_stem_removes_extension() -> None:
    assert filename_stem("Example Agreement.pdf") == "Example Agreement"


def test_parse_empty_evidence_list() -> None:
    assert parse_evidence_list("[]") == []
    assert parse_evidence_list("") == []


def test_parse_multiple_evidence_items() -> None:
    raw = "['first clause', 'second clause']"
    assert parse_evidence_list(raw) == ["first clause", "second clause"]


def test_parse_plain_text_fallback() -> None:
    assert parse_evidence_list("plain evidence") == ["plain evidence"]


def test_select_starter_records_keeps_top_documents() -> None:
    records = [
        {"document_id": "a", "clause_type": "x"},
        {"document_id": "a", "clause_type": "y"},
        {"document_id": "b", "clause_type": "x"},
        {"document_id": "c", "clause_type": "x"},
    ]

    selected = select_starter_records(records, max_contracts=2)

    assert {record["document_id"] for record in selected} == {"a", "b"}
