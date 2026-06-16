"""Shared RAG/Qdrant helpers for ClauseLens.

RAG means Retrieval-Augmented Generation. In this project, the retrieval part is:
1. turn contract/clause text into embeddings,
2. store those embeddings in Qdrant,
3. search Qdrant later to find relevant contract evidence.

This module intentionally does not create a global Qdrant client or load the
embedding model at import time. Those operations can be slow or fail if Qdrant
is not running, so we expose functions that callers can run explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

# Name of the Qdrant collection where contract clause vectors are stored.
COLLECTION = "contracts_clause_evidence"

# SentenceTransformer model used to convert text into 384-number vectors.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Embedded/local Qdrant path. This works without Docker or a Qdrant server.
QDRANT_PATH = Path("data/qdrant_local")

# Server URL used when Qdrant is running separately, usually via Docker.
QDRANT_URL = "http://localhost:6333"

# __all__ is optional. It documents which names this module expects other code
# to import when someone writes "from app.rag import ...".
__all__ = [
    "ClauseSearchResult",
    "COLLECTION",
    "EMBEDDING_MODEL",
    "QDRANT_PATH",
    "QDRANT_URL",
    "Distance",
    "FieldCondition",
    "Filter",
    "MatchValue",
    "PointStruct",
    "QdrantClient",
    "SentenceTransformer",
    "VectorParams",
    "create_qdrant_client",
    "ensure_collection",
    "load_jsonl_records",
    "make_clause_type_filter",
    "load_embedding_model",
    "search_clause_evidence",
    "serialize_search_result",
    "stable_point_id",
]


@dataclass(frozen=True)
class ClauseSearchResult:
    """One retrieved clause evidence result from Qdrant."""

    score: float
    payload: dict[str, Any]

    @property
    def clause_type(self) -> str | None:
        value = self.payload.get("clause_type")
        return str(value) if value is not None else None

    @property
    def source_pdf(self) -> str | None:
        value = self.payload.get("source_pdf")
        return str(value) if value is not None else None

    @property
    def text(self) -> str:
        return str(self.payload.get("text", ""))

    @property
    def source_txt(self) -> str | None:
        value = self.payload.get("source_txt")
        return str(value) if value is not None else None

    @property
    def document_id(self) -> str | None:
        value = self.payload.get("document_id")
        return str(value) if value is not None else None

    @property
    def answer(self) -> str | None:
        value = self.payload.get("answer")
        return str(value) if value is not None else None


def stable_point_id(raw_id: str) -> int:
    """Turn a text record ID into a stable integer ID for Qdrant."""

    digest = hashlib.sha256(raw_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def load_jsonl_records(path: Path) -> list[dict[str, object]]:
    """Load prepared ClauseLens evidence records from JSONL."""

    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if not record.get("id") or not record.get("text"):
                raise ValueError(f"Record on line {line_number} must include id and text")
            records.append(record)

    return records


def create_qdrant_client(
    *,
    url: str | None = None,
    path: str | Path | None = None,
) -> QdrantClient:
    """Create a Qdrant client.

    The "*" in the function signature makes url and path keyword-only
    arguments. That means callers must write:

        create_qdrant_client(path=QDRANT_PATH)

    instead of passing values positionally. This makes calls clearer because
    url and path mean very different modes.

    Return type: "-> QdrantClient" means this function returns a QdrantClient
    object from the qdrant-client library.
    """

    if path is not None:
        # Embedded mode: Qdrant stores files in a local folder and no server
        # needs to be running on localhost:6333.
        return QdrantClient(path=str(path))

    # Server mode: connect to a running Qdrant service.
    # os.getenv("QDRANT_URL", QDRANT_URL) means:
    # use the environment variable QDRANT_URL if it exists, otherwise use the
    # default constant "http://localhost:6333".
    return QdrantClient(url=url or os.getenv("QDRANT_URL", QDRANT_URL))


def ensure_collection(
    client: QdrantClient,
    *,
    collection_name: str = COLLECTION,
    vector_size: int = 384,
    distance: Distance = Distance.COSINE,
) -> None:
    """Create the Qdrant collection if it does not already exist.

    client: QdrantClient means the caller passes an already-created client.
    collection_name: str = COLLECTION means the default collection name is the
    COLLECTION constant unless the caller overrides it.
    vector_size: int = 384 matches BAAI/bge-small-en-v1.5, which outputs
    384-dimensional embeddings.
    distance controls how Qdrant compares vectors. COSINE is common for
    normalized text embeddings.
    """

    if client.collection_exists(collection_name=collection_name):
        # Return early if the collection is already there. This makes the
        # function safe to call more than once.
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=distance),
    )


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load the sentence-transformers embedding model.

    This is a separate function because loading model weights can take time.
    Keeping it explicit helps notebooks/scripts control when the slow work
    happens.
    """

    return SentenceTransformer(model_name)


def make_clause_type_filter(clause_type: str | None) -> Filter | None:
    """Build a Qdrant payload filter for a CUAD clause type."""

    if not clause_type:
        return None

    return Filter(
        must=[
            FieldCondition(
                key="clause_type",
                match=MatchValue(value=clause_type),
            )
        ]
    )


def search_clause_evidence(
    *,
    client: QdrantClient,
    model: SentenceTransformer,
    query: str,
    clause_type: str | None = None,
    limit: int = 5,
    collection_name: str = COLLECTION,
) -> list[ClauseSearchResult]:
    """Search indexed clause evidence using a natural-language query.

    The caller provides the Qdrant client and embedding model so this function
    stays easy to reuse from scripts, APIs, notebooks, or a future UI.
    """

    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query must not be empty")

    if limit < 1:
        raise ValueError("limit must be at least 1")

    query_vector = model.encode(
        clean_query,
        normalize_embeddings=True,
    )
    if hasattr(query_vector, "tolist"):
        query_vector = query_vector.tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=make_clause_type_filter(clause_type),
        limit=limit,
        with_payload=True,
    )

    return [
        ClauseSearchResult(
            score=float(point.score),
            payload=dict(point.payload or {}),
        )
        for point in results.points
    ]


def serialize_search_result(result: ClauseSearchResult) -> dict[str, object]:
    """Convert a Qdrant result into the public API/UI result shape."""

    return {
        "score": result.score,
        "clause_type": result.clause_type,
        "source_pdf": result.source_pdf,
        "source_txt": result.source_txt,
        "document_id": result.document_id,
        "answer": result.answer,
        "text": result.text,
    }
