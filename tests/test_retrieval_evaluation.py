from pathlib import Path

import pytest

from backend.app.fake_embedding_client import FakeEmbeddingClient
from backend.app.keyword_baseline import load_keyword_baseline
from backend.app.retrieval_evaluation import (
    InMemoryVectorIndex,
    RetrievalEvaluationCase,
    RetrievalEvaluationError,
    build_keyword_ranker,
    load_retrieval_cases,
    run_retrieval_evaluation,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "backend" / "evaluation" / "keyword_retrieval_baseline_v1.json"
COMPARISON_PATH = PROJECT_ROOT / "backend" / "evaluation" / "retrieval_comparison_v1.json"


def test_real_evaluation_cases_compare_rankers_and_record_model_version() -> None:
    dataset = load_keyword_baseline(BASELINE_PATH)
    cases = load_retrieval_cases(COMPARISON_PATH)
    vector_client = FakeEmbeddingClient(
        {
            document.content: (1.0, 0.0) if index == 0 else (0.0, 1.0)
            for index, document in enumerate(dataset.documents)
        }
        | {
            case.vector_query: (1.0, 0.0)
            for case in cases
        }
    )
    vector_index = InMemoryVectorIndex(
        dataset.documents,
        vector_client,
        model_version="fake-test-v1",
    )

    report = run_retrieval_evaluation(
        cases,
        dataset="keyword_retrieval_baseline_v1",
        embedding_source="test_fake",
        embedding_model_version="fake-test-v1",
        keyword_ranker=build_keyword_ranker(dataset),
        vector_ranker=vector_index.rank,
    )

    assert report.quality_evidence is False
    assert report.embedding_model_version == "fake-test-v1"
    assert len(report.cases) == 5
    assert set(report.metrics) == {"keyword", "vector"}
    assert report.metrics["keyword"].precision_at_k[0].value == pytest.approx(0.8)
    assert report.metrics["vector"].case_count == 5
    assert all(
        summary.sample_count == 5 and summary.mean_ms >= 0
        for summary in report.latency.values()
    )


def test_evaluation_rejects_duplicate_rankings_and_missing_model_version() -> None:
    case = RetrievalEvaluationCase(
        case_id="case-one",
        question_id="question-one",
        keyword_query="关键词",
        vector_query="完整语义问题",
        relevant_document_paths=("notes/one.md",),
    )

    with pytest.raises(RetrievalEvaluationError, match="unique"):
        run_retrieval_evaluation(
            (case,),
            dataset="test",
            embedding_source="test",
            embedding_model_version="v1",
            keyword_ranker=lambda _query, _top_k: ("notes/one.md",),
            vector_ranker=lambda _query, _top_k: ("notes/one.md", "notes/one.md"),
        )

    with pytest.raises(RetrievalEvaluationError, match="embedding_model_version"):
        run_retrieval_evaluation(
            (case,),
            dataset="test",
            embedding_source="test",
            embedding_model_version="",
            keyword_ranker=lambda _query, _top_k: (),
            vector_ranker=lambda _query, _top_k: (),
        )
