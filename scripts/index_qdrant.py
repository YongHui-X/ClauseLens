"""Index prepared ClauseLens evidence records into Qdrant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client.models import PointStruct  # noqa: E402

from app.rag import (  # noqa: E402
    COLLECTION,
    EMBEDDING_MODEL,
    QDRANT_URL,
    create_qdrant_client,
    ensure_collection,
    load_embedding_model,
    load_jsonl_records,
    stable_point_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index prepared ClauseLens evidence records into Qdrant."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/starter_clause_evidence.jsonl"),
    )
    parser.add_argument(
        "--url",
        default=QDRANT_URL,
        help="Qdrant server URL. Ignored when --qdrant-path is provided.",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        help="Use embedded local Qdrant storage instead of a running Qdrant server.",
    )
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the target collection before indexing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl_records(args.input)
    if not records:
        raise ValueError(f"No records found in {args.input}")

    model = load_embedding_model(args.model)
    client = create_qdrant_client(path=args.qdrant_path) if args.qdrant_path else (
        create_qdrant_client(url=args.url)
    )

    if args.recreate and client.collection_exists(collection_name=args.collection):
        client.delete_collection(collection_name=args.collection)

    ensure_collection(client, collection_name=args.collection)

    texts = [str(record["text"]) for record in records]
    print(f"Embedding {len(texts)} records with {args.model}")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    points = [
        PointStruct(
            id=stable_point_id(str(record["id"])),
            vector=embedding.tolist(),
            payload=record,
        )
        for record, embedding in zip(records, embeddings, strict=True)
    ]

    client.upsert(collection_name=args.collection, points=points)
    count = client.count(collection_name=args.collection, exact=True).count
    print(f"Indexed {len(points)} records into {args.collection}")
    print(f"Collection count: {count}")


if __name__ == "__main__":
    main()
