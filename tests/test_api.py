from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api import SearchEngine, create_app


class FakeVector:
    def tolist(self) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeModel:
    def encode(self, query: str, *, normalize_embeddings: bool) -> FakeVector:
        assert query == "audit rights"
        assert normalize_embeddings is True
        return FakeVector()


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def collection_exists(self, *, collection_name: str) -> bool:
        assert collection_name == "contracts_clause_evidence"
        return True

    def query_points(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=0.92,
                    payload={
                        "clause_type": "Audit Rights",
                        "source_pdf": "Example.pdf",
                        "source_txt": "Example.txt",
                        "document_id": "Example",
                        "answer": "Yes",
                        "text": "Customer may audit records during business hours.",
                    },
                )
            ]
        )


def build_test_client() -> tuple[TestClient, FakeClient]:
    app = create_app()
    fake_client = FakeClient()
    app.dependency_overrides[app.state.get_search_engine] = lambda: SearchEngine(
        client=fake_client,
        model=FakeModel(),
    )
    return TestClient(app), fake_client


def test_health_reports_collection_status() -> None:
    client, _ = build_test_client()

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["collection_ready"] is True


def test_clause_types_lists_supported_filters() -> None:
    client, _ = build_test_client()

    response = client.get("/clause-types")

    assert response.status_code == 200
    assert "Audit Rights" in response.json()["clause_types"]


def test_search_returns_cited_clause_results() -> None:
    client, fake_client = build_test_client()

    response = client.post(
        "/search",
        json={
            "query": "  audit rights  ",
            "clause_type": "Audit Rights",
            "limit": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "audit rights"
    assert body["clause_type"] == "Audit Rights"
    assert body["result_count"] == 1
    assert body["results"][0]["score"] == 0.92
    assert body["results"][0]["source_pdf"] == "Example.pdf"
    assert body["results"][0]["source_txt"] == "Example.txt"
    assert body["results"][0]["document_id"] == "Example"
    assert body["results"][0]["answer"] == "Yes"
    assert body["results"][0]["text"].startswith("Customer may audit")
    assert fake_client.calls[0]["limit"] == 3
    assert fake_client.calls[0]["query_filter"] is not None


def test_search_rejects_blank_query() -> None:
    client, _ = build_test_client()

    response = client.post("/search", json={"query": " ", "limit": 5})

    assert response.status_code == 422


def test_search_rejects_invalid_limit() -> None:
    client, _ = build_test_client()

    response = client.post("/search", json={"query": "audit rights", "limit": 0})

    assert response.status_code == 422


def test_search_accepts_blank_clause_type_as_no_filter() -> None:
    client, fake_client = build_test_client()

    response = client.post(
        "/search",
        json={"query": "audit rights", "clause_type": " ", "limit": 1},
    )

    assert response.status_code == 200
    assert response.json()["clause_type"] is None
    assert fake_client.calls[0]["query_filter"] is None
