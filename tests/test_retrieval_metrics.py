import json
from pathlib import Path

import pytest

from backend.app.retrieval_metrics import (
    RetrievalMetricCase,
    RetrievalMetricError,
    aggregate_retrieval_metrics,
    measure_latency_ms,
    summarize_latency_ms,
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
        "precision_at_k": [
            {"k": 1, "value": 0.5},
            {"k": 3, "value": 0.5},
        ],
        "mrr": 0.75,
    }


def test_recorded_comparison_is_metric_consistent_and_scope_limited() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["evaluation_scope"]["embedding_source"] == "scripted_fake"
    assert report["evaluation_scope"]["top_k"] == [1, 3]
    assert [case["case_id"] for case in report["cases"]] == [
        "approval-consent",
        "sqlite-persistence",
        "approval-file",
        "document-formats",
        "retrieval-baseline",
    ]
    decision = report["decision"]
    assert decision["status"] == "retain_minimal_experiment"
    assert decision["default_retrieval"] == "keyword"
    assert decision["hybrid_default_enabled"] is False
    assert decision["hybrid_mode"] == "comparison_only"
    assert decision["hybrid_adoption_gate"] == {
        "requires_real_embedding_evidence": True,
        "required_ordering": "Hybrid > Vector > Keyword",
    }
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


def test_latency_measurement_and_summary_keep_runtime_observation() -> None:
    result, latency_ms = measure_latency_ms(lambda: "done")

    assert result == "done"
    summary = summarize_latency_ms((latency_ms, 1.25, 2.5))

    assert summary.sample_count == 3
    assert summary.min_ms == pytest.approx(min(latency_ms, 1.25, 2.5))
    assert summary.max_ms == pytest.approx(max(latency_ms, 1.25, 2.5))
    assert summary.mean_ms >= 0


def test_latency_summary_rejects_empty_or_negative_measurements() -> None:
    with pytest.raises(RetrievalMetricError, match="must not be empty"):
        summarize_latency_ms(())

    with pytest.raises(RetrievalMetricError, match="non-negative"):
        summarize_latency_ms((-0.1,))
