"""FastAPI service for the ClauseLens retrieval demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.cuad import STARTER_CLAUSE_TYPES
from app.rag import (
    COLLECTION,
    EMBEDDING_MODEL,
    QDRANT_PATH,
    QdrantClient,
    SentenceTransformer,
    create_qdrant_client,
    load_embedding_model,
    search_clause_evidence,
    serialize_search_result,
)


class SearchRequest(BaseModel):
    """Request body for semantic clause search."""

    query: str = Field(..., description="Natural-language contract question")
    clause_type: str | None = Field(
        default=None,
        description="Optional CUAD clause type filter",
    )
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("query must not be empty")
        return clean_value

    @field_validator("clause_type")
    @classmethod
    def blank_clause_type_becomes_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        clean_value = value.strip()
        return clean_value or None


class SearchResult(BaseModel):
    """Public result shape returned by the API."""

    score: float
    clause_type: str | None
    source_pdf: str | None
    source_txt: str | None
    document_id: str | None
    answer: str | None
    text: str


class SearchResponse(BaseModel):
    """Search response with query metadata and ranked results."""

    query: str
    clause_type: str | None
    limit: int
    result_count: int
    results: list[SearchResult]


@dataclass
class SearchEngine:
    """Loaded retrieval dependencies shared across API requests."""

    client: QdrantClient
    model: SentenceTransformer


def create_app(
    *,
    qdrant_path: Path = QDRANT_PATH,
    collection_name: str = COLLECTION,
    model_name: str = EMBEDDING_MODEL,
) -> FastAPI:
    app = FastAPI(
        title="ClauseLens API",
        description="Semantic clause-evidence search over CUAD contract records.",
        version="0.1.0",
    )

    def get_search_engine() -> SearchEngine:
        engine = getattr(app.state, "search_engine", None)
        if engine is None:
            engine = SearchEngine(
                client=create_qdrant_client(path=qdrant_path),
                model=load_embedding_model(model_name),
            )
            app.state.search_engine = engine
        return engine

    search_engine_dependency = Depends(get_search_engine)

    @app.get("/")
    def root() -> dict[str, object]:
        return {
            "name": "ClauseLens API",
            "purpose": "Search CUAD contract clause evidence with citations.",
            "docs": "/docs",
            "health": "/health",
            "clause_types": "/clause-types",
            "search": {
                "method": "POST",
                "path": "/search",
                "example": {
                    "query": "Does the contract restrict assignment?",
                    "clause_type": "Anti-Assignment",
                    "limit": 5,
                },
            },
        }

    @app.get("/health")
    def health(engine: SearchEngine = search_engine_dependency) -> dict[str, object]:
        collection_ready = False
        try:
            collection_ready = engine.client.collection_exists(
                collection_name=collection_name
            )
        except Exception:
            collection_ready = False

        return {
            "status": "ok",
            "collection": collection_name,
            "collection_ready": collection_ready,
            "qdrant_path": str(qdrant_path),
            "model": model_name,
        }

    @app.get("/clause-types")
    def clause_types() -> dict[str, list[str]]:
        return {"clause_types": STARTER_CLAUSE_TYPES}

    @app.post("/search", response_model=SearchResponse)
    def search(
        request: SearchRequest,
        engine: SearchEngine = search_engine_dependency,
    ) -> SearchResponse:
        try:
            results = search_clause_evidence(
                client=engine.client,
                model=engine.model,
                query=request.query,
                clause_type=request.clause_type,
                limit=request.limit,
                collection_name=collection_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        serialized = [serialize_search_result(result) for result in results]
        return SearchResponse(
            query=request.query,
            clause_type=request.clause_type,
            limit=request.limit,
            result_count=len(serialized),
            results=serialized,
        )

    app.dependency_overrides_provider = app
    app.state.get_search_engine = get_search_engine
    return app


app = create_app()
