import pytest

from backend.app.citation_runtime import bind_citations
from backend.app.retrieval_context import build_retrieval_context_from_items


def _context(*, workspace_id: int = 7, source_version: str | None = "a" * 64):
    return build_retrieval_context_from_items(
        workspace_id=workspace_id,
        query="审批",
        items=[
            {
                "workspace_id": workspace_id,
                "file_id": 11,
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "source_relative_path": "docs/guide.md",
                "text": "审批流程需要留存记录。",
                "chunk_index": 0,
                "source_version": source_version,
                "score": 2,
                "start_offset": 0,
                "end_offset": 10,
                "start_line": 1,
                "end_line": 1,
            }
        ],
        total=1,
        top_k=5,
        has_more=False,
    )


def test_bind_citations_whitelists_context_source() -> None:
    binding = bind_citations(
        "审批需要留存记录 [[cite_chunk-1]]。",
        _context(),
        workspace_id=7,
    )

    assert binding.status == "bound"
    assert binding.citation_ids == ("cite_chunk-1",)
    assert binding.invalid_reasons == ()


def test_bind_citations_rejects_unknown_and_cross_workspace_source() -> None:
    unknown = bind_citations(
        "事实 [[cite_missing]]。",
        _context(),
        workspace_id=7,
    )
    cross_workspace = bind_citations(
        "事实 [[cite_chunk-1]]。",
        _context(workspace_id=8),
        workspace_id=7,
    )

    assert unknown.status == "invalid"
    assert "unknown_citation_id" in unknown.invalid_reasons
    assert cross_workspace.status == "invalid"
    assert "cross_workspace_context" in cross_workspace.invalid_reasons


def test_bind_citations_rejects_missing_or_mismatched_source_version() -> None:
    missing = bind_citations(
        "事实 [[cite_chunk-1]]。",
        _context(source_version=None),
        workspace_id=7,
    )
    mismatched = bind_citations(
        "事实 [[cite_chunk-1]]。",
        _context(),
        workspace_id=7,
        current_source_versions={"doc-1": "b" * 64},
    )

    assert missing.status == "invalid"
    assert "missing_source_version" in missing.invalid_reasons
    assert mismatched.status == "invalid"
    assert "source_version_mismatch" in mismatched.invalid_reasons


@pytest.mark.parametrize(
    ("navigation_only", "expected_status"),
    [(True, "unbound"), (False, "invalid")],
)
def test_bind_citations_distinguishes_navigation_without_citations(
    navigation_only: bool,
    expected_status: str,
) -> None:
    binding = bind_citations(
        "文件位于 docs/guide.md。",
        _context(),
        workspace_id=7,
        require_citations=True,
        navigation_only=navigation_only,
    )

    assert binding.status == expected_status
