from datetime import datetime, timezone
from uuid import uuid4

from backend.app.document_contracts import Document
from backend.app.document_versioning import (
    classify_document_version,
    document_version_key,
)


SOURCE_UPDATED_AT = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)


def _document(
    *,
    file_entry_id: int = 7,
    source_version: str = "a" * 64,
) -> Document:
    return Document(
        document_id=uuid4(),
        workspace_id=1,
        file_entry_id=file_entry_id,
        source_relative_path="notes/project.md",
        source_format="markdown",
        normalized_text="one\ntwo",
        source_version=source_version,
        source_updated_at=SOURCE_UPDATED_AT,
    )


def test_document_version_key_uses_file_identity_and_content_version() -> None:
    document = _document()

    assert document_version_key(document) == (7, "a" * 64)


def test_classify_document_version_distinguishes_new_modified_and_duplicate() -> None:
    current = _document()

    assert classify_document_version(current, None) == "new"
    assert (
        classify_document_version(
            current,
            _document(source_version="b" * 64),
        )
        == "modified"
    )
    assert classify_document_version(current, _document()) == "duplicate"
