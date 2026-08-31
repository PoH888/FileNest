from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.app.document_contracts import RetrievalContext
from backend.app.retrieval_context import (
    RetrievalContextError,
    build_retrieval_context_from_items,
    retrieval_chunk_to_mapping,
)


def _item(*, workspace_id: int = 7) -> dict[str, object]:
    return {
        "workspace_id": workspace_id,
        "file_id": 11,
        "document_id": "doc-1",
        "chunk_id": "chunk-1",
        "source_relative_path": "docs/guide.md",
        "text": "审批流程需要留存记录。",
        "chunk_index": 0,
        "source_version": "a" * 64,
        "source_updated_at": "2026-09-01T00:00:00+00:00",
        "indexed_at": "2026-09-02T00:00:00+00:00",
        "score": 2,
        "start_offset": 0,
        "end_offset": 10,
        "start_line": 1,
        "end_line": 1,
    }


def test_retrieval_context_is_hashed_and_projects_verified_chunk() -> None:
    context = build_retrieval_context_from_items(
        workspace_id=7,
        query="审批",
        items=[_item()],
        total=1,
        top_k=5,
        has_more=False,
        retrieved_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
    )

    assert context.snapshot_hash is not None
    assert context.has_complete_source_versions is True
    projected = retrieval_chunk_to_mapping(context.chunks[0])
    assert projected["citation_id"] == "cite_chunk-1"
    assert projected["source_relative_path"] == "docs/guide.md"
    assert projected["source_version"] == "a" * 64

    tampered = context.model_dump(mode="json")
    tampered["snapshot_hash"] = "0" * 64
    with pytest.raises(ValidationError):
        RetrievalContext.model_validate(tampered)


def test_retrieval_context_rejects_cross_workspace_tool_item() -> None:
    with pytest.raises(RetrievalContextError):
        build_retrieval_context_from_items(
            workspace_id=7,
            query="审批",
            items=[_item(workspace_id=8)],
            total=1,
            top_k=5,
            has_more=False,
        )
