from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from backend.app.document_contracts import Chunk, Document, DocumentPosition


DOCUMENT_ID = UUID("233eb1b5-4298-4ae0-8d25-19d5555d5f3f")
CHUNK_ID = UUID("8b0f0327-44f4-484a-8a16-73bc15a1994e")
SOURCE_UPDATED_AT = datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc)


def _document(**overrides: object) -> Document:
    values: dict[str, object] = {
        "document_id": DOCUMENT_ID,
        "workspace_id": 3,
        "file_entry_id": 7,
        "source_relative_path": "notes/project.md",
        "source_format": "markdown",
        "normalized_text": "第一行\n第二行",
        "source_version": "a" * 64,
        "source_updated_at": SOURCE_UPDATED_AT,
    }
    values.update(overrides)
    return Document(**values)


def _chunk(**overrides: object) -> Chunk:
    values: dict[str, object] = {
        "chunk_id": CHUNK_ID,
        "document_id": DOCUMENT_ID,
        "file_entry_id": 7,
        "source_relative_path": "notes/project.md",
        "chunk_index": 1,
        "text": "第二行",
        "start_offset": 4,
        "end_offset": 7,
        "start_line": 2,
        "end_line": 2,
    }
    values.update(overrides)
    return Chunk(**values)


def test_document_and_chunk_preserve_file_and_position_traceability() -> None:
    document = _document()
    chunk = _chunk()

    assert document.source_format == "markdown"
    assert chunk.document_id == document.document_id
    assert chunk.file_entry_id == document.file_entry_id
    assert chunk.source_relative_path == document.source_relative_path
    assert (
        document.normalized_text[chunk.start_offset : chunk.end_offset]
        == chunk.text
    )
    assert (chunk.start_line, chunk.end_line) == (2, 2)


def test_document_rejects_unsafe_source_path() -> None:
    with pytest.raises(ValidationError, match="normalized relative path"):
        _document(source_relative_path="../outside.md")


def test_chunk_rejects_invalid_source_position() -> None:
    with pytest.raises(ValidationError, match="greater than start_offset"):
        _chunk(end_offset=4)


def test_document_position_rejects_paragraph_without_index() -> None:
    with pytest.raises(ValidationError, match="paragraph position metadata"):
        DocumentPosition(
            element_type="paragraph",
            start_offset=0,
            end_offset=4,
        )


def test_chunk_rejects_non_overlapping_source_position() -> None:
    position = DocumentPosition(
        element_type="paragraph",
        start_offset=0,
        end_offset=2,
        paragraph_index=0,
    )

    with pytest.raises(ValidationError, match="overlap chunk range"):
        _chunk(source_positions=(position,))
