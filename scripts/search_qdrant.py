"""Search the QFind Qdrant collection from the terminal."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Find and import app.rag.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag import (  # noqa: E402
    COLLECTION,
    EMBEDDING_MODEL,
    QDRANT_PATH,
    QDRANT_URL,
    RERANKER_MODEL,
    create_qdrant_client,
    load_embedding_model,
    load_reranker_model,
    search_clause_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the QFind Qdrant collection."
    )
    parser.add_argument("query", help="Natural language search query.")
    parser.add_argument("--clause-type")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--qdrant-mode", choices=["server", "embedded"], default="server")
    parser.add_argument(
        "--qdrant-url",
        default=(
            os.getenv("QDRANT_CLOUD_URL")
            or os.getenv("QDRANT_URL")
            or os.getenv("QDRANT_LOCAL_URL")
            or QDRANT_URL
        ),
    )
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument(
        "--qdrant-timeout",
        type=float,
        default=float(os.getenv("QDRANT_TIMEOUT_SECONDS", "60")),
    )
    parser.add_argument("--qdrant-path", type=Path, default=QDRANT_PATH)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=RERANKER_MODEL)
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable cross-encoder reranking.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = parse_args()
    client = (
        create_qdrant_client(
            url=args.qdrant_url,
            api_key=args.qdrant_api_key,
            timeout=args.qdrant_timeout,
        )
        if args.qdrant_mode == "server"
        else create_qdrant_client(path=args.qdrant_path)
    )
    model = load_embedding_model(args.model)
    reranker = load_reranker_model(args.reranker_model) if args.rerank else None

    results = search_clause_evidence(
        client=client,
        model=model,
        query=args.query,
        clause_type=args.clause_type,
        limit=args.limit,
        collection_name=args.collection,
        reranker=reranker,
        rerank=args.rerank,
    )

    for index, result in enumerate(results, start=1):
        print(f"Result {index}: score={result.score:.3f}")
        print(f"Clause type: {result.clause_type}")
        print(f"Source: {result.source_pdf}")
        print(result.text[:800])
        print("-" * 80)


if __name__ == "__main__":
    main()
