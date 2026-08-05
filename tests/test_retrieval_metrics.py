import json
from pathlib import Path

from backend.app.retrieval_metrics import (
    RetrievalMetricCase,
    aggregate_retrieval_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = (
    PROJECT_ROOT
    / "backend"
    / "evaluation"
    / "retrieval_comparison_v1.json"
)


def test_retrieval_metrics_calculate_recall_and_mrr() -> None:
    cases = (
        RetrievalMetricCase(
            case_id="case-a",
            relevant_document_paths=("a", "b"),
            ranked_document_paths=("b", "x", "a"),
        ),
        RetrievalMetricCase(
            case_id="case-b",
            relevant_document_paths=("c",),
            ranked_document_paths=("x", "c"),
        ),
    )

    summary = aggregate_retrieval_metrics(cases, ks=(1, 3))

    assert summary.model_dump(mode="json") == {
        "case_count": 2,
        "recall_at_k": [
            {"k": 1, "value": 0.25},
            {"k": 3, "value": 1.0},
        ],
        "mrr": 0.75,
    }


def test_recorded_comparison_is_metric_consistent_and_scope_limited() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["evaluation_scope"]["embedding_source"] == "scripted_fake"
    assert report["evaluation_scope"]["top_k"] == [1, 3]
    decision = report["decision"]
    assert decision["status"] == "retain_minimal_experiment"
    assert decision["default_retrieval"] == "keyword"
    assert decision["evidence"] == {
        "keyword_gap_confirmed": True,
        "offline_vector_gain_observed": True,
        "hybrid_gain_over_vector_observed": False,
        "real_embedding_quality_established": False,
    }
    assert "external_vector_database" in decision["not_adopted"]
    assert "hybrid_search_as_default" in decision["not_adopted"]

    for method in ("keyword", "vector", "hybrid"):
        cases = tuple(
            RetrievalMetricCase(
                case_id=case["case_id"],
                relevant_document_paths=tuple(
                    case["relevant_document_paths"]
                ),
                ranked_document_paths=tuple(case["rankings"][method]),
            )
            for case in report["cases"]
        )
        calculated = aggregate_retrieval_metrics(
            cases,
            ks=tuple(report["evaluation_scope"]["top_k"]),
        ).model_dump(mode="json")
        assert calculated == report["metrics"][method]

    assert report["metrics"]["keyword"]["mrr"] < report["metrics"]["vector"]["mrr"]
    assert report["metrics"]["hybrid"] == report["metrics"]["vector"]
    assert report["cases"][0]["rankings"]["keyword"] == []
