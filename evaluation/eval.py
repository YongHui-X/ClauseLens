"""Retrieval evaluation for ClauseLens.

This evaluates the current retrieval layer only. It intentionally does not
score generated answers yet because ClauseLens does not have answer generation.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag import (  # noqa: E402
    COLLECTION,
    EMBEDDING_MODEL,
    QDRANT_PATH,
    ClauseSearchResult,
    create_qdrant_client,
    load_embedding_model,
    search_clause_evidence,
)
from evaluation.cases import (  # noqa: E402
    DEFAULT_TEST_FILE,
    RetrievalTestCase,
    load_tests,
)

RESULT_COLUMNS = [
    "question",
    "expected_clause_type",
    "category",
    "top_k",
    "result_count",
    "expected_clause_type_rank",
    "clause_type_mrr",
    "top1_clause_hit",
    "topk_clause_hit",
    "keyword_hit_rate",
    "keywords_found",
    "total_keywords",
    "ndcg",
    "passed",
]


@dataclass(frozen=True)
class RetrievalEvalResult:
    """Metrics for one retrieval test case."""

    question: str
    expected_clause_type: str
    category: str
    top_k: int
    result_count: int
    expected_clause_type_rank: int | None
    clause_type_mrr: float
    top1_clause_hit: bool
    topk_clause_hit: bool
    keyword_hit_rate: float
    keywords_found: int
    total_keywords: int
    ndcg: float

    @property
    def passed(self) -> bool:
        # "Passed" is intentionally strict: the expected clause type should be
        # first, and every keyword should be recoverable from the retrieved set.
        return self.top1_clause_hit and self.keyword_hit_rate == 1.0

    @property
    def keyword_coverage(self) -> float:
        # Backward-compatible alias. Older code and notes still use this name.
        return self.keyword_hit_rate


def result_text(result: ClauseSearchResult) -> str:
    """Combine searchable text from a retrieval result."""

    # We evaluate against all useful retrieval payload fields, not just the
    # displayed snippet, so the keyword check matches what a user could inspect.
    payload = result.payload
    parts = [
        result.clause_type or "",
        result.source_pdf or "",
        result.text,
        str(payload.get("answer", "")),
        str(payload.get("document_id", "")),
    ]
    return "\n".join(parts).lower()


def find_expected_clause_type_rank(
    results: list[ClauseSearchResult],
    expected_clause_type: str,
) -> int | None:
    """Return one-based rank of the first result with the expected clause type."""

    for index, result in enumerate(results, start=1):
        if result.clause_type == expected_clause_type:
            return index
    return None


def calculate_ndcg_for_clause_type(
    results: list[ClauseSearchResult],
    expected_clause_type: str,
    k: int,
) -> float:
    """Calculate binary nDCG for expected clause-type relevance."""

    relevances = [
        1 if result.clause_type == expected_clause_type else 0
        for result in results[:k]
    ]

    dcg = sum(
        relevance / math.log2(index + 2)
        for index, relevance in enumerate(relevances)
    )
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = sum(
        relevance / math.log2(index + 2)
        for index, relevance in enumerate(ideal_relevances)
    )

    return dcg / idcg if idcg > 0 else 0.0


def count_keywords_found(
    results: list[ClauseSearchResult],
    keywords: list[str],
) -> int:
    """Count expected keywords found anywhere in the retrieved result set."""

    combined_text = "\n".join(result_text(result) for result in results)
    return sum(1 for keyword in keywords if keyword.lower() in combined_text)


def result_rows(results: list[RetrievalEvalResult]) -> list[dict[str, Any]]:
    """Convert evaluation objects into a file-friendly row format."""

    return [
        {
            "question": result.question,
            "expected_clause_type": result.expected_clause_type,
            "category": result.category,
            "top_k": result.top_k,
            "result_count": result.result_count,
            "expected_clause_type_rank": result.expected_clause_type_rank,
            "clause_type_mrr": round(result.clause_type_mrr, 4),
            "top1_clause_hit": result.top1_clause_hit,
            "topk_clause_hit": result.topk_clause_hit,
            "keyword_hit_rate": round(result.keyword_hit_rate, 4),
            "keywords_found": result.keywords_found,
            "total_keywords": result.total_keywords,
            "ndcg": round(result.ndcg, 4),
            "passed": result.passed,
        }
        for result in results
    ]


def write_results(path: Path, results: list[RetrievalEvalResult]) -> None:
    """Write detailed evaluation rows to JSON or CSV."""

    rows = result_rows(results)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".json":
        import json

        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return

    if path.suffix.lower() != ".csv":
        raise ValueError("output path must end in .json or .csv")

    with path.open("w", encoding="utf-8", newline="") as file:
        # Use a fixed header order so CSV output stays stable across runs.
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_case(
    test_case: RetrievalTestCase,
    results: list[ClauseSearchResult],
    *,
    top_k: int,
) -> RetrievalEvalResult:
    """Evaluate one retrieval case from already-retrieved results."""

    expected_rank = find_expected_clause_type_rank(
        results,
        test_case.expected_clause_type,
    )
    top1_clause_hit = expected_rank == 1
    topk_clause_hit = expected_rank is not None
    keywords_found = count_keywords_found(results, test_case.keywords)
    total_keywords = len(test_case.keywords)
    keyword_hit_rate = (keywords_found / total_keywords) if total_keywords else 0.0

    return RetrievalEvalResult(
        question=test_case.question,
        expected_clause_type=test_case.expected_clause_type,
        category=test_case.category,
        top_k=top_k,
        result_count=len(results),
        expected_clause_type_rank=expected_rank,
        clause_type_mrr=(1.0 / expected_rank) if expected_rank else 0.0,
        top1_clause_hit=top1_clause_hit,
        topk_clause_hit=topk_clause_hit,
        keyword_hit_rate=keyword_hit_rate,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        ndcg=calculate_ndcg_for_clause_type(
            results,
            test_case.expected_clause_type,
            top_k,
        ),
    )


def evaluate_all(
    *,
    tests_path: Path = DEFAULT_TEST_FILE,
    qdrant_path: Path = QDRANT_PATH,
    collection_name: str = COLLECTION,
    model_name: str = EMBEDDING_MODEL,
    top_k: int = 5,
) -> list[RetrievalEvalResult]:
    """Run retrieval evaluation for all test cases."""

    test_cases = load_tests(tests_path)
    client = create_qdrant_client(path=qdrant_path)
    model = load_embedding_model(model_name)

    results: list[RetrievalEvalResult] = []
    for test_case in test_cases:
        retrieved = search_clause_evidence(
            client=client,
            model=model,
            query=test_case.question,
            limit=top_k,
            collection_name=collection_name,
        )
        results.append(evaluate_case(test_case, retrieved, top_k=top_k))

    return results


def print_summary(results: list[RetrievalEvalResult]) -> None:
    """Print a compact CLI report."""

    if not results:
        print("No evaluation cases found.")
        return

    pass_count = sum(1 for result in results if result.passed)
    avg_mrr = sum(result.clause_type_mrr for result in results) / len(results)
    avg_ndcg = sum(result.ndcg for result in results) / len(results)
    avg_keyword_hit_rate = (
        sum(result.keyword_hit_rate for result in results) / len(results)
    )
    top1_hit_rate = (
        sum(1 for result in results if result.top1_clause_hit) / len(results)
    )
    topk_hit_rate = sum(1 for result in results if result.topk_clause_hit) / len(results)

    print("ClauseLens Retrieval Evaluation")
    print("=" * 36)
    print(f"Cases: {len(results)}")
    print(f"Passed: {pass_count}/{len(results)}")
    print(f"Average clause-type MRR: {avg_mrr:.3f}")
    print(f"Average clause-type nDCG: {avg_ndcg:.3f}")
    print(f"Top-1 clause hit rate: {top1_hit_rate:.1%}")
    print(f"Top-k clause hit rate: {topk_hit_rate:.1%}")
    print(f"Average keyword hit rate: {avg_keyword_hit_rate:.1%}")
    print()

    for index, result in enumerate(results, start=1):
        rank = result.expected_clause_type_rank
        rank_text = str(rank) if rank is not None else "not found"
        status = "PASS" if result.passed else "FAIL"
        print(f"{index}. {status} [{result.category}] {result.question}")
        print(f"   expected clause: {result.expected_clause_type} | rank: {rank_text}")
        print(
            "   keywords: "
            f"{result.keywords_found}/{result.total_keywords} | "
            f"nDCG: {result.ndcg:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ClauseLens retrieval against JSONL test cases."
    )
    parser.add_argument("--tests", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--qdrant-path", type=Path, default=QDRANT_PATH)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for detailed results as JSON or CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate_all(
        tests_path=args.tests,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection,
        model_name=args.model,
        top_k=args.top_k,
    )
    print_summary(results)
    if args.output:
        write_results(args.output, results)
        print(f"Wrote detailed results to {args.output}")


if __name__ == "__main__":
    main()
