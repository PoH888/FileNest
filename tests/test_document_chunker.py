from datetime import datetime, timezone
from uuid import UUID

import pytest

from backend.app.document_chunker import chunk_document
from backend.app.document_contracts import Document, DocumentPage, DocumentPosition


DOCUMENT_ID = UUID("7b3c83de-31d9-44e2-8ab7-2d00d1606a45")


def _document(text: str) -> Document:
    return Document(
        document_id=DOCUMENT_ID,
        workspace_id=3,
        file_entry_id=7,
        source_relative_path="notes/project.md",
        source_format="markdown",
        normalized_text=text,
        source_version="b" * 64,
        source_updated_at=datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc),
    )


def _pdf_document() -> Document:
    text = "first page\n\nsecond page"
    first_page_end = len("first page")
    second_page_start = first_page_end + 2
    return Document(
        document_id=DOCUMENT_ID,
        workspace_id=3,
        file_entry_id=7,
        source_relative_path="reports/summary.pdf",
        source_format="pdf",
        normalized_text=text,
        pages=(
            DocumentPage(
                page_number=1,
                start_offset=0,
                end_offset=first_page_end,
            ),
            DocumentPage(
                page_number=2,
                start_offset=second_page_start,
                end_offset=len(text),
            ),
        ),
        source_version="b" * 64,
        source_updated_at=datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc),
    )


def _docx_document() -> Document:
    text = "Title\n\nBody text\n\nCell text"
    return Document(
        document_id=DOCUMENT_ID,
        workspace_id=3,
        file_entry_id=7,
        source_relative_path="reports/summary.docx",
        source_format="docx",
        normalized_text=text,
        source_positions=(
            DocumentPosition(
                element_type="paragraph",
                start_offset=0,
                end_offset=5,
                section_index=0,
                heading_level=1,
                paragraph_index=0,
            ),
            DocumentPosition(
                element_type="paragraph",
                start_offset=7,
                end_offset=16,
                section_index=0,
                paragraph_index=1,
            ),
            DocumentPosition(
                element_type="table_cell",
                start_offset=18,
                end_offset=27,
                section_index=1,
                table_index=0,
                row_index=0,
                cell_index=0,
            ),
        ),
        source_version="b" * 64,
        source_updated_at=datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc),
    )


def test_chunk_document_packs_lines_and_preserves_traceable_ranges() -> None:
    document = _document("one\ntwo\nthree\nfour")

    chunks = chunk_document(document, max_chars=8)

    assert [chunk.text for chunk in chunks] == [
        "one\ntwo\n",
        "three\n",
        "four",
    ]
    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [
        (1, 2),
        (3, 3),
        (4, 4),
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert all(
        document.normalized_text[chunk.start_offset : chunk.end_offset]
        == chunk.text
        and chunk.document_id == document.document_id
        and chunk.file_entry_id == document.file_entry_id
        and chunk.source_relative_path == document.source_relative_path
        for chunk in chunks
    )
    assert all(
        (chunk.page_start, chunk.page_end) == (None, None)
        for chunk in chunks
    )


def test_chunk_document_preserves_pdf_page_range() -> None:
    chunks = chunk_document(_pdf_document(), max_chars=100)

    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_chunk_document_preserves_docx_structure_provenance() -> None:
    document = _docx_document()

    chunks = chunk_document(document, max_chars=100)

    assert len(chunks) == 1
    assert chunks[0].source_positions == document.source_positions
    assert [
        (
            position.heading_level,
            position.section_index,
            position.paragraph_index,
            position.table_index,
            position.row_index,
            position.cell_index,
        )
        for position in chunks[0].source_positions
    ] == [
        (1, 0, 0, None, None, None),
        (None, 0, 1, None, None, None),
        (None, 1, None, 0, 0, 0),
    ]


def test_chunk_document_splits_a_line_that_exceeds_the_limit() -> None:
    document = _document("abcdefghij\n")

    chunks = chunk_document(document, max_chars=4)

    assert [chunk.text for chunk in chunks] == ["abcd", "efgh", "ij\n"]
    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        (0, 4),
        (4, 8),
        (8, 11),
    ]
    assert all(
        (chunk.start_line, chunk.end_line) == (1, 1) for chunk in chunks
    )


def test_chunk_document_rejects_non_positive_max_chars() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        chunk_document(_document("text"), max_chars=0)
