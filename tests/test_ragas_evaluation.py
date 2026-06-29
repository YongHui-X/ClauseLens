import json
from pathlib import Path

from evaluation.ragas_cases import load_ragas_cases
from evaluation.ragas_eval import (
    RagasCollectedCase,
    ragas_dataset_rows,
    summarize_results,
)


def test_load_ragas_cases_reads_seed_dataset() -> None:
    cases = load_ragas_cases()

    assert len(cases) == 12
    assert cases[0].case_id == "assignment-unauthorized-consequence"
    assert cases[0].critical is True
    assert cases[-1].expected_clause_type is None
    assert cases[-1].messages[-1].content == (
        "Is either party required to indemnify the other?"
    )


def test_ragas_dataset_rows_use_expected_columns() -> None:
    row = RagasCollectedCase(
        case_id="case",
        critical=False,
        user_input="Question?",
        response="Answer.",
        reference="Reference.",
        retrieved_contexts=["Context."],
        expected_clause_type="Audit Rights",
        resolved_clause_type="Audit Rights",
        abstained=False,
    )

    assert ragas_dataset_rows([row]) == [
        {
            "user_input": "Question?",
            "response": "Answer.",
            "reference": "Reference.",
            "retrieved_contexts": ["Context."],
        }
    ]


def test_summarize_results_reports_threshold_failures_and_json_shape(
    tmp_path: Path,
) -> None:
    rows = [
        RagasCollectedCase(
            case_id="critical-case",
            critical=True,
            user_input="Question?",
            response="Answer.",
            reference="Reference.",
            retrieved_contexts=["Context."],
            expected_clause_type="Anti-Assignment",
            resolved_clause_type="Anti-Assignment",
            abstained=False,
        )
    ]

    report = summarize_results(
        rows=rows,
        scores=[
            {
                "faithfulness": 0.70,
                "answer_relevancy": 0.82,
                "context_precision": 0.90,
                "context_recall": 0.80,
            }
        ],
        judge_model="mock-judge",
    )

    assert report["passed"] is False
    assert report["judge_model"] == "mock-judge"
    assert report["aggregate_means"]["faithfulness"] == 0.7
    assert report["cases"][0]["scores"]["answer_relevancy"] == 0.82
    assert any("faithfulness mean" in failure for failure in report["failures"])
    assert any("critical faithfulness" in failure for failure in report["failures"])

    output = tmp_path / "ragas_eval_results.json"
    output.write_text(json.dumps(report), encoding="utf-8")
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["thresholds"]["critical_faithfulness_min"] == 0.75
