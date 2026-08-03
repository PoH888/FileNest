from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest

from backend.app.document_chunker import chunk_document
from backend.app.document_contracts import Document


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
