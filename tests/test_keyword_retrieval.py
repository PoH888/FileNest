import json
from pathlib import Path

import pytest

from backend.app.keyword_baseline import load_keyword_baseline
from backend.app.keyword_retrieval import (
    KeywordSearchError,
    search_keyword_documents,
    search_keyword_results,
)


BASELINE_PATH = (
    Path(__file__).parents[1]
    / "backend"
    / "evaluation"
    / "keyword_retrieval_baseline_v1.json"
)
RESULTS_PATH = (
    Path(__file__).parents[1]
    / "backend"
    / "evaluation"
    / "keyword_retrieval_baseline_results_v1.json"
)


def test_search_keyword_returns_content_matches_in_fixed_document_order() -> None:
    dataset = load_keyword_baseline(BASELINE_PATH)

    results = search_keyword_documents(dataset, "sqlite")

    assert [document.relative_path for document in results] == [
        "guides/persistence.md",
        "archive/legacy-plan.md",
    ]


def test_search_keyword_results_rank_matches_and_return_sources() -> None:
    dataset = load_keyword_baseline(BASELINE_PATH)

    results = search_keyword_results(dataset, "文件")

    assert [result.document.relative_path for result in results] == [
        "guides/approval-flow.md",
        "guides/persistence.md",
        "archive/legacy-plan.md",
    ]
    assert [result.score for result in results] == [2, 1, 1]
    assert [source.matched_text for source in results[0].sources] == [
        "文件",
        "文件",
    ]
    assert all(
        source.source_relative_path == results[0].document.relative_path
        and source.start_line == source.end_line == 1
        and results[0].document.content[
            source.start_offset : source.end_offset
        ]
        == source.matched_text
        for source in results[0].sources
    )


def test_recorded_baseline_samples_match_current_search_results() -> None:
    dataset = load_keyword_baseline(BASELINE_PATH)
    report = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    questions = {
        question.question_id: question for question in dataset.questions
    }

    assert report["schema_version"] == "1.0"
    assert report["dataset"] == "keyword_retrieval_baseline_v1"
    assert [sample["sample_id"] for sample in report["samples"]] == [
        "hit-approval-file",
        "hit-sqlite",
        "miss-vector-database",
        "invalid-blank-keyword",
    ]

    for sample in report["samples"]:
        question_id = sample["question_id"]
        if question_id is not None:
            assert sample["expected_related_document_paths"] == list(
                questions[question_id].related_document_paths
            )

        if sample["outcome"] == "invalid_query":
            with pytest.raises(KeywordSearchError) as error:
                search_keyword_results(dataset, sample["keyword"])
            assert str(error.value) == sample["error"]
            continue

        results = search_keyword_results(dataset, sample["keyword"])
        observed_results = [
            {
                "relative_path": result.document.relative_path,
                "score": result.score,
                "sources": [
                    source.model_dump(mode="json")
                    for source in result.sources
                ],
            }
            for result in results
        ]
        assert observed_results == sample["observed_results"]


def test_search_keyword_rejects_blank_keyword() -> None:
    dataset = load_keyword_baseline(BASELINE_PATH)

    with pytest.raises(KeywordSearchError, match="must not be blank"):
        search_keyword_documents(dataset, "   ")
