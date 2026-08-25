import json
from pathlib import Path

import pytest

from backend.app.citation_evaluation import (
    CitationDatasetError,
    evaluate_citation_correctness,
    evaluate_faithfulness,
    evaluate_source_relevance,
    load_citation_evaluation_dataset,
)


DATASET_PATH = (
    Path(__file__).parents[2]
    / "backend"
    / "evaluation"
    / "citation_correctness_v1.json"
)


def test_citation_dataset_has_normal_case_and_all_required_dimensions() -> None:
    dataset = load_citation_evaluation_dataset(DATASET_PATH)
    cases = {case.case_id: case for case in dataset.cases}

    assert dataset.schema_version == "1.0"
    assert set(cases) == {
        "release-date-supported",
        "irrelevant-source",
        "unfaithful-answer",
        "uncited-fact",
    }

    normal = cases["release-date-supported"]
    relevance = evaluate_source_relevance(normal)
    faithfulness = evaluate_faithfulness(normal)
    correctness = evaluate_citation_correctness(normal)

    assert (relevance.total, relevance.passed, relevance.failed) == (1, 1, 0)
    assert (faithfulness.total, faithfulness.passed, faithfulness.failed) == (
        1,
        1,
        0,
    )
    assert (correctness.total_facts, correctness.correct_facts) == (1, 1)
    assert correctness.incorrect_fact_ids == ()


def test_citation_dimensions_report_key_failure_paths() -> None:
    dataset = load_citation_evaluation_dataset(DATASET_PATH)
    cases = {case.case_id: case for case in dataset.cases}

    irrelevant = evaluate_source_relevance(cases["irrelevant-source"])
    assert irrelevant.failed_ids == ("cite-team-budget",)

    unfaithful = evaluate_faithfulness(cases["unfaithful-answer"])
    assert unfaithful.failed_ids == ("fact-wrong-budget",)

    uncited = evaluate_citation_correctness(cases["uncited-fact"])
    assert uncited.incorrect_fact_ids == ("fact-weekly-audit",)
    assert uncited.fact_results[-1].has_citation is False


def test_citation_dataset_rejects_unknown_fact_citation(tmp_path: Path) -> None:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    payload["cases"][0]["facts"][0]["citation_ids"] = ["missing-citation"]
    invalid_path = tmp_path / "invalid-citation-dataset.json"
    invalid_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(CitationDatasetError, match="格式无效"):
        load_citation_evaluation_dataset(invalid_path)
