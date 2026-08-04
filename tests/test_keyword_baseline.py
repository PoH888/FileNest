from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.keyword_baseline import (
    KeywordBaselineDataset,
    KeywordBaselineDocument,
    KeywordBaselineQuestion,
    load_keyword_baseline,
)


BASELINE_PATH = (
    Path(__file__).parents[1]
    / "backend"
    / "evaluation"
    / "keyword_retrieval_baseline_v1.json"
)


def test_keyword_baseline_has_stable_documents_and_questions() -> None:
    dataset = load_keyword_baseline(BASELINE_PATH)

    assert dataset.schema_version == "1.0"
    assert dataset.name == "FileNest 关键词检索固定基线"
    assert [document.relative_path for document in dataset.documents] == [
        "guides/approval-flow.md",
        "guides/persistence.md",
        "notes/document-tracking.md",
        "notes/retrieval-baseline.txt",
        "archive/legacy-plan.md",
    ]
    assert [question.question_id for question in dataset.questions] == [
        "approval-before-move",
        "sqlite-persistence",
        "document-formats",
        "retrieval-baseline",
        "no-vector-database",
    ]
    assert dataset.questions[-1].related_document_paths == ()


def test_keyword_baseline_rejects_unknown_related_document() -> None:
    with pytest.raises(ValidationError, match="does not exist"):
        KeywordBaselineDataset(
            schema_version="1.0",
            name="固定基线",
            documents=(
                KeywordBaselineDocument(
                    relative_path="notes/known.md",
                    content="已知文档。",
                ),
            ),
            questions=(
                KeywordBaselineQuestion(
                    question_id="unknown-reference",
                    question="查找未知文档",
                    related_document_paths=("notes/missing.md",),
                ),
            ),
        )
