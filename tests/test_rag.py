from types import SimpleNamespace

import pytest

from app.rag import make_clause_type_filter, search_clause_evidence


class FakeVector:
    def tolist(self) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, query: str, *, normalize_embeddings: bool) -> FakeVector:
        self.calls.append(
            {
                "query": query,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        return FakeVector()


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def query_points(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=0.91,
                    payload={
                        "clause_type": "Audit Rights",
                        "source_pdf": "Example.pdf",
                        "text": "Audit evidence",
                    },
                )
            ]
        )


def test_make_clause_type_filter_returns_none_without_clause_type() -> None:
    assert make_clause_type_filter(None) is None
    assert make_clause_type_filter("") is None


def test_make_clause_type_filter_matches_clause_type() -> None:
    query_filter = make_clause_type_filter("Audit Rights")

    assert query_filter is not None
    condition = query_filter.must[0]
    assert condition.key == "clause_type"
    assert condition.match.value == "Audit Rights"


def test_search_clause_evidence_embeds_and_queries_qdrant() -> None:
    client = FakeClient()
    model = FakeModel()

    results = search_clause_evidence(
        client=client,
        model=model,
        query="  audit rights  ",
        clause_type="Audit Rights",
        limit=3,
        collection_name="contracts_clause_evidence",
    )

    assert model.calls == [
        {
            "query": "audit rights",
            "normalize_embeddings": True,
        }
    ]
    assert len(client.calls) == 1
    assert client.calls[0]["collection_name"] == "contracts_clause_evidence"
    assert client.calls[0]["query"] == [0.1, 0.2, 0.3]
    assert client.calls[0]["limit"] == 3
    assert client.calls[0]["with_payload"] is True
    assert client.calls[0]["query_filter"] is not None

    assert len(results) == 1
    assert results[0].score == 0.91
    assert results[0].clause_type == "Audit Rights"
    assert results[0].source_pdf == "Example.pdf"
    assert results[0].text == "Audit evidence"


def test_search_clause_evidence_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        search_clause_evidence(
            client=FakeClient(),
            model=FakeModel(),
            query=" ",
        )


def test_search_clause_evidence_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be at least 1"):
        search_clause_evidence(
            client=FakeClient(),
            model=FakeModel(),
            query="assignment",
            limit=0,
        )
