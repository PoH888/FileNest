from hashlib import sha256
from pathlib import Path
from uuid import UUID

import pytest

from backend.app.document_parser import (
    UnsupportedDocumentFormatError,
    parse_document,
)
from backend.app.filesystem_adapter import FileSystemAdapter


DOCUMENT_ID = UUID("8a96e8c4-6ebc-4b5f-83b8-b27c681e8d95")


def test_parse_document_supports_markdown_and_txt_and_normalizes(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    adapter = FileSystemAdapter(workspace_root)
    cases = [
        (
            Path("notes/README.MD"),
            b"\xef\xbb\xbfTitle\r\n\r\nBody\r",
            "markdown",
            "Title\n\nBody\n",
        ),
        (
            Path("notes/plain.TXT"),
            b"Plain\r\ntext",
            "text",
            "Plain\ntext",
        ),
    ]

    for file_entry_id, (
        relative_path,
        raw_bytes,
        source_format,
        expected_text,
    ) in enumerate(cases, start=7):
        file_path = workspace_root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(raw_bytes)

        document = parse_document(
            adapter,
            workspace_id=3,
            file_entry_id=file_entry_id,
            source_relative_path=relative_path,
            document_id=DOCUMENT_ID,
        )

        assert document.source_relative_path == relative_path.as_posix()
        assert document.source_format == source_format
        assert document.normalized_text == expected_text
        assert document.source_version == sha256(raw_bytes).hexdigest()
        assert document.source_updated_at.tzinfo is not None


def test_parse_document_rejects_unsupported_format(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    unsupported_file = workspace_root / "notes" / "report.pdf"
    unsupported_file.parent.mkdir(parents=True)
    unsupported_file.write_bytes(b"not a supported text document")
    adapter = FileSystemAdapter(workspace_root)

    with pytest.raises(
        UnsupportedDocumentFormatError,
        match="unsupported document format",
    ):
        parse_document(
            adapter,
            workspace_id=3,
            file_entry_id=7,
            source_relative_path=Path("notes/report.pdf"),
            document_id=DOCUMENT_ID,
        )
