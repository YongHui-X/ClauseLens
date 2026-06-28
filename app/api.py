"""FastAPI service for the QFind retrieval demo."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.chat import (
    ChatEngine,
    ChatRequest,
    ChatResult,
    answer_chat_turn,
    create_chat_client,
    stream_chat_turn,
)
from app.cuad import STARTER_CLAUSE_TYPES
from app.rag import (
    COLLECTION,
    DEFAULT_EVIDENCE_PATH,
    EMBEDDING_MODEL,
    QDRANT_PATH,
    RERANKER_MODEL,
    CrossEncoder,
    LexicalIndex,
    QdrantClient,
    SentenceTransformer,
    create_qdrant_client,
    load_embedding_model,
    load_lexical_index,
    load_qdrant_payload_records,
    load_reranker_model,
    search_clause_evidence,
    serialize_search_result,
)
from app.security import (
    SlidingWindowRateLimiter,
    issue_session_cookie,
    make_session_dependency,
)

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


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
    vector_score: float | None = None
    reranker_score: float | None = None
    lexical_score: float | None = None
    fused_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
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
    reranker: CrossEncoder | None = None
    reranking_enabled: bool = False
    reranker_loader: Callable[[], CrossEncoder] | None = None
    lexical_index: object | None = None
    lexical_source: str = "none"
    lexical_record_count: int = 0
    _reranker_lock: threading.Lock = dataclass_field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def get_reranker(self) -> CrossEncoder:
        """Load the cross-encoder only when a request actually needs it."""

        if self.reranker is None:
            with self._reranker_lock:
                if self.reranker is None:
                    if self.reranker_loader is None:
                        raise ValueError("reranker is not configured")
                    self.reranker = self.reranker_loader()
        return self.reranker


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _qdrant_server_url(override: str | None = None) -> str | None:
    return (
        override
        or os.getenv("QDRANT_CLOUD_URL")
        or os.getenv("QDRANT_URL")
        or os.getenv("QDRANT_LOCAL_URL")
    )


def create_app(
    *,
    qdrant_path: Path = QDRANT_PATH,
    qdrant_mode: str | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    collection_name: str = COLLECTION,
    model_name: str = EMBEDDING_MODEL,
    reranker_model_name: str | None = None,
    reranking_enabled: bool | None = None,
    warmup_enabled: bool | None = None,
) -> FastAPI:
    effective_reranker_model = (
        reranker_model_name
        or os.getenv("RERANKER_MODEL", "").strip()
        or RERANKER_MODEL
    )
    use_reranking = (
        _env_flag("RERANKING_ENABLED", default=False)
        if reranking_enabled is None
        else reranking_enabled
    )
    use_warmup = (
        _env_flag("MODEL_WARMUP_ENABLED", default=True)
        if warmup_enabled is None
        else warmup_enabled
    )
    engine_lock = threading.RLock()
    rate_limiter = SlidingWindowRateLimiter()

    def build_lexical_index(
        client: QdrantClient,
        effective_qdrant_mode: str,
    ) -> tuple[object, str, int]:
        if effective_qdrant_mode == "server":
            records = load_qdrant_payload_records(
                client,
                collection_name=collection_name,
            )
            return LexicalIndex(records), "qdrant_payloads", len(records)
        lexical_index = load_lexical_index()
        return lexical_index, str(DEFAULT_EVIDENCE_PATH), len(lexical_index.records)

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        app_instance.state.ready = False
        app_instance.state.warmup_error = None
        app_instance.state.warmup_latency_ms = 0.0
        if use_warmup:
            started = time.perf_counter()
            try:
                search_engine = get_search_engine()
                search_engine.model.encode(
                    "contract clause retrieval warmup",
                    normalize_embeddings=True,
                )
                reranker = search_engine.get_reranker()
                reranker.predict(
                    [("contract rights", "The agreement grants contract rights.")]
                )
                client_factory = getattr(get_chat_engine().llm, "_client", None)
                if callable(client_factory):
                    client_factory()
                app_instance.state.ready = True
            except Exception as exc:
                app_instance.state.warmup_error = str(exc)
            finally:
                app_instance.state.warmup_latency_ms = (
                    time.perf_counter() - started
                ) * 1000
        else:
            app_instance.state.ready = True
        yield

    app = FastAPI(
        title="QFind API",
        description="Semantic clause-evidence search over CUAD contract records.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.ready = not use_warmup
    app.state.warmup_error = None
    app.state.warmup_latency_ms = 0.0
    app.state.lexical_error = None

    def get_search_engine() -> SearchEngine:
        # Tests can inject a fake engine on app.state without patching the
        # dependency graph. In production this stays unset and we build the
        # actual Qdrant + embedding stack once per process.
        override = getattr(app.state, "search_engine_override", None)
        if override is not None:
            return override

        engine = getattr(app.state, "search_engine", None)
        if engine is None:
            with engine_lock:
                engine = getattr(app.state, "search_engine", None)
                if engine is None:
                    engine = SearchEngine(
                        client=(
                            create_qdrant_client(
                                url=qdrant_url,
                                api_key=qdrant_api_key,
                            )
                            if (
                                qdrant_mode
                                or os.getenv("QDRANT_MODE", "server")
                            ).lower()
                            == "server"
                            else create_qdrant_client(path=qdrant_path)
                        ),
                        model=load_embedding_model(model_name),
                        reranking_enabled=use_reranking,
                        reranker_loader=lambda: load_reranker_model(
                            effective_reranker_model
                        ),
                    )
                    effective_qdrant_mode = (
                        qdrant_mode or os.getenv("QDRANT_MODE", "server")
                    ).lower()
                    try:
                        (
                            engine.lexical_index,
                            engine.lexical_source,
                            engine.lexical_record_count,
                        ) = build_lexical_index(engine.client, effective_qdrant_mode)
                        app.state.lexical_error = None
                    except Exception as exc:
                        engine.lexical_index = None
                        engine.lexical_source = "error"
                        engine.lexical_record_count = 0
                        app.state.lexical_error = str(exc)
                    app.state.search_engine = engine
        return engine

    def get_chat_engine() -> ChatEngine:
        # Chat depends on retrieval, so it reuses the search engine instead of
        # constructing a second Qdrant client or embedding model.
        override = getattr(app.state, "chat_engine_override", None)
        if override is not None:
            return override

        engine = getattr(app.state, "chat_engine", None)
        if engine is None:
            with engine_lock:
                engine = getattr(app.state, "chat_engine", None)
                if engine is None:
                    llm = create_chat_client()
                    engine = ChatEngine(
                        search_engine=get_search_engine(),
                        llm=llm,
                        model_name=llm.model,
                    )
                    app.state.chat_engine = engine
        return engine

    search_engine_dependency = Depends(get_search_engine)
    chat_engine_dependency = Depends(get_chat_engine)
    browser_session_dependency = Depends(make_session_dependency(rate_limiter))

    @app.get("/api")
    def api_info() -> dict[str, object]:
        return {
            "name": "QFind API",
            "purpose": "Search CUAD contract clause evidence with citations.",
            "docs": "/docs",
            "health": "/health",
            "clause_types": "/clause-types",
            "chat": {
                "method": "POST",
                "path": "/chat",
                "example": {
                    "messages": [
                        {"role": "user", "content": "Can a party walk away after notice?"}
                    ],
                    "clause_type": "Termination For Convenience",
                    "limit": 5,
                },
            },
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

    @app.get("/api/session")
    def api_session(response: Response) -> dict[str, object]:
        return issue_session_cookie(response)

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
            "status": "ok" if app.state.ready else "warming",
            "ready": bool(app.state.ready),
            "warmup_latency_ms": round(float(app.state.warmup_latency_ms), 3),
            "warmup_error": app.state.warmup_error,
            "collection": collection_name,
            "collection_ready": collection_ready,
            "qdrant_path": str(qdrant_path),
            "qdrant_mode": (
                qdrant_mode or os.getenv("QDRANT_MODE", "server")
            ),
            "qdrant_url": _qdrant_server_url(qdrant_url),
            "qdrant_api_key_configured": bool(
                qdrant_api_key or os.getenv("QDRANT_API_KEY")
            ),
            "lexical_ready": engine.lexical_index is not None,
            "lexical_source": engine.lexical_source,
            "lexical_record_count": engine.lexical_record_count,
            "lexical_error": app.state.lexical_error,
            "model": model_name,
            "reranking_enabled": engine.reranking_enabled,
            "reranker_model": (
                effective_reranker_model
            ),
            "answer_model": getattr(
                getattr(app.state, "chat_engine", None),
                "model_name",
                os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            ),
            "service_tier": os.getenv("OPENAI_SERVICE_TIER", "standard"),
        }

    @app.get("/clause-types")
    def clause_types(_session_id: str = browser_session_dependency) -> dict[str, list[str]]:
        return {"clause_types": STARTER_CLAUSE_TYPES}

    @app.post("/search", response_model=SearchResponse)
    def search(
        request: SearchRequest,
        engine: SearchEngine = search_engine_dependency,
        _session_id: str = browser_session_dependency,
    ) -> SearchResponse:
        if engine.lexical_source == "error":
            raise HTTPException(
                status_code=503,
                detail=f"Lexical retrieval is not ready: {app.state.lexical_error}",
            )
        try:
            results = search_clause_evidence(
                client=engine.client,
                model=engine.model,
                query=request.query,
                clause_type=request.clause_type,
                limit=request.limit,
                collection_name=collection_name,
                reranker=engine.reranker,
                rerank=engine.reranking_enabled,
                lexical_index=engine.lexical_index,
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

    @app.post("/chat", response_model=ChatResult)
    def chat(
        request: ChatRequest,
        engine: ChatEngine = chat_engine_dependency,
        _session_id: str = browser_session_dependency,
    ) -> ChatResult:
        if engine.search_engine.lexical_source == "error":
            raise HTTPException(
                status_code=503,
                detail=f"Lexical retrieval is not ready: {app.state.lexical_error}",
            )
        try:
            # Chat failures should surface as a clear HTTP error instead of a
            # silent 500, because the frontend needs to distinguish config
            # issues from empty retrieval results.
            return answer_chat_turn(engine=engine, request=request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Chat generation failed: {exc}",
            ) from exc

    @app.post("/chat/stream")
    def chat_stream(
        request: ChatRequest,
        engine: ChatEngine = chat_engine_dependency,
        _session_id: str = browser_session_dependency,
    ) -> StreamingResponse:
        def event_stream():
            try:
                if engine.search_engine.lexical_source == "error":
                    raise ValueError(
                        f"Lexical retrieval is not ready: {app.state.lexical_error}"
                    )
                yield from stream_chat_turn(engine=engine, request=request)
            except Exception as exc:
                yield json.dumps(
                    {
                        "event": "error",
                        "detail": f"Chat generation failed: {exc}",
                    },
                    ensure_ascii=False,
                ) + "\n"

        return StreamingResponse(
            event_stream(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    app.dependency_overrides_provider = app
    app.state.get_search_engine = get_search_engine
    app.state.get_chat_engine = get_chat_engine
    app.state.rate_limiter = rate_limiter
    if FRONTEND_DIST.exists():
        app.mount(
            "/",
            StaticFiles(directory=FRONTEND_DIST, html=True),
            name="frontend",
        )
    return app


app = create_app()
