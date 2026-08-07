from datetime import datetime, timezone
from uuid import uuid4

from backend.app.document_contracts import (
    Document,
    DocumentPage,
    DocumentPosition,
)
from backend.app.document_sync import sync_document_snapshot


SOURCE_UPDATED_AT = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)


def _document(
    *,
    file_entry_id: int,
    source_version: str,
    source_format: str = "markdown",
    pages: tuple[DocumentPage, ...] = (),
    source_positions: tuple[DocumentPosition, ...] = (),
) -> Document:
    return Document(
        document_id=uuid4(),
        workspace_id=1,
        file_entry_id=file_entry_id,
        source_relative_path=f"notes/{file_entry_id}.{source_format}",
        source_format=source_format,
        normalized_text="one\ntwo",
        pages=pages,
        source_positions=source_positions,
        source_version=source_version,
        source_updated_at=SOURCE_UPDATED_AT,
    )


def test_sync_document_snapshot_classifies_all_change_types() -> None:
    previous_documents = (
        _document(file_entry_id=1, source_version="a" * 64),
        _document(file_entry_id=2, source_version="b" * 64),
        _document(file_entry_id=3, source_version="c" * 64),
    )
    current_documents = (
        _document(file_entry_id=1, source_version="d" * 64),
        _document(file_entry_id=2, source_version="b" * 64),
        _document(file_entry_id=4, source_version="e" * 64),
    )

    result = sync_document_snapshot(current_documents, previous_documents)

    assert [document.file_entry_id for document in result.new_documents] == [4]
    assert [document.file_entry_id for document in result.modified_documents] == [1]
    assert [document.file_entry_id for document in result.deleted_documents] == [3]
    assert [document.file_entry_id for document in result.duplicate_documents] == [2]


def test_sync_document_snapshot_keeps_current_location_metadata() -> None:
    current_pdf = _document(
        file_entry_id=5,
        source_version="f" * 64,
        source_format="pdf",
        pages=(DocumentPage(page_number=1, start_offset=0, end_offset=7),),
    )
    current_docx = _document(
        file_entry_id=6,
        source_version="1" * 64,
        source_format="docx",
        source_positions=(
            DocumentPosition(
                element_type="paragraph",
                start_offset=0,
                end_offset=7,
                paragraph_index=0,
            ),
        ),
    )
    previous_documents = (
        _document(
            file_entry_id=5,
            source_version="a" * 64,
            source_format="pdf",
            pages=(DocumentPage(page_number=1, start_offset=0, end_offset=7),),
        ),
        _document(
            file_entry_id=6,
            source_version="b" * 64,
            source_format="docx",
            source_positions=(
                DocumentPosition(
                    element_type="paragraph",
                    start_offset=0,
                    end_offset=7,
                    paragraph_index=0,
                ),
            ),
        ),
    )

    result = sync_document_snapshot(
        (current_pdf, current_docx),
        previous_documents,
    )

    assert result.modified_documents == (current_pdf, current_docx)
    assert result.modified_documents[0].pages == current_pdf.pages
    assert (
        result.modified_documents[1].source_positions
        == current_docx.source_positions
    )
