from backend.app.citation_evaluation import (
    evaluate_agent_citation_observation,
)
from backend.app.document_contracts import DocumentPosition, RetrievalContext
from backend.app.retrieval_context import build_retrieval_context_from_items


def _context(
    *,
    path: str = "report.pdf",
    page_start: int | None = 1,
    page_end: int | None = 1,
    source_positions: tuple[DocumentPosition, ...] = (),
) -> RetrievalContext:
    return build_retrieval_context_from_items(
        workspace_id=7,
        query="审批",
        items=[
            {
                "workspace_id": 7,
                "file_id": 11,
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "source_relative_path": path,
                "text": "审批流程需要留存记录。",
                "chunk_index": 0,
                "source_version": "a" * 64,
                "score": 2,
                "start_offset": 0,
                "end_offset": 10,
                "start_line": 1,
                "end_line": 1,
                "page_start": page_start,
                "page_end": page_end,
                "source_positions": source_positions,
            }
        ],
        total=1,
        top_k=5,
        has_more=False,
    )


def test_evaluation_adapter_reports_real_pdf_citation_scores() -> None:
    observation = evaluate_agent_citation_observation(
        run_id=12,
        workspace_id=7,
        workspace_fixture="citation-pdf-fixture",
        request_text="审批流程是什么？",
        final_answer="审批流程需要留存记录 [[cite_chunk-1]]。",
        retrieval_context=_context(),
        prompt_version="agent-system-v1",
    )

    assert observation.status == "valid"
    assert observation.source_snapshot_hash is not None
    assert observation.parser_version == "citation-parser-v1"
    assert observation.source_relevance.passed == 1
    assert observation.faithfulness.passed == 1
    assert observation.correctness.correct_facts == 1
    assert observation.sources[0].page_start == 1


def test_evaluation_adapter_fails_closed_for_stale_or_structurally_incomplete_source() -> None:
    stale = evaluate_agent_citation_observation(
        run_id=13,
        workspace_id=7,
        workspace_fixture="citation-stale-fixture",
        request_text="审批流程是什么？",
        final_answer="审批流程需要留存记录 [[cite_chunk-1]]。",
        retrieval_context=_context(),
        prompt_version="agent-system-v1",
        current_source_versions={"doc-1": "b" * 64},
    )
    incomplete_docx = evaluate_agent_citation_observation(
        run_id=14,
        workspace_id=7,
        workspace_fixture="citation-docx-fixture",
        request_text="审批流程是什么？",
        final_answer="审批流程需要留存记录 [[cite_chunk-1]]。",
        retrieval_context=_context(
            path="guide.docx",
            page_start=None,
            page_end=None,
        ),
        prompt_version="agent-system-v1",
    )

    assert stale.status == "invalid_observation"
    assert "source_version_mismatch" in stale.invalid_reasons
    assert incomplete_docx.status == "invalid_observation"
    assert "docx_citation_without_structure" in incomplete_docx.invalid_reasons


def test_evaluation_adapter_records_missing_citations_as_invalid_observation() -> None:
    observation = evaluate_agent_citation_observation(
        run_id=15,
        workspace_id=7,
        workspace_fixture="citation-missing-fixture",
        request_text="审批流程是什么？",
        final_answer="审批流程需要留存记录。",
        retrieval_context=_context(),
        prompt_version="agent-system-v1",
    )

    assert observation.status == "invalid_observation"
    assert "citations_required" in observation.invalid_reasons
