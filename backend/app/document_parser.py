"""Markdown/TXT 文档读取与规范化。"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from .document_contracts import Document
from .filesystem_adapter import FileSystemAdapter


class DocumentParseError(ValueError):
    """文档无法按当前文本输入契约解析。"""


class UnsupportedDocumentFormatError(DocumentParseError):
    """文档不是当前课程允许的 Markdown/TXT 格式。"""


def normalize_document_text(text: str) -> str:
    """只统一 BOM 与换行符，不改写 Markdown/TXT 的正文空白。"""

    if text.startswith("\ufeff"):
        text = text[1:]
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_document(
    adapter: FileSystemAdapter,
    *,
    workspace_id: int,
    file_entry_id: int,
    source_relative_path: str | Path,
    document_id: UUID | None = None,
) -> Document:
    """通过已授权适配器读取一个 Markdown/TXT 文件并生成 Document。"""

    source_path = Path(source_relative_path)
    source_format = _source_format_for_path(source_path)
    source_metadata = adapter.get_file_metadata(source_path)
    if source_metadata is None:
        raise DocumentParseError("source path must point to a regular file")

    try:
        raw_text = adapter.read_text(source_path, encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentParseError("document must be valid UTF-8 text") from exc

    source_version = adapter.get_file_sha256(source_path)
    if source_version is None:
        raise DocumentParseError("source path must remain a regular file")

    return Document(
        document_id=document_id if document_id is not None else uuid4(),
        workspace_id=workspace_id,
        file_entry_id=file_entry_id,
        source_relative_path=source_path.as_posix(),
        source_format=source_format,
        normalized_text=normalize_document_text(raw_text),
        source_version=source_version,
        source_updated_at=datetime.fromtimestamp(
            source_metadata.mtime_ns / 1_000_000_000,
            tz=timezone.utc,
        ),
    )


def _source_format_for_path(path: Path) -> Literal["markdown", "text"]:
    """按扩展名确定当前课程允许的文本格式。"""

    suffix = path.suffix.casefold()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".txt":
        return "text"
    raise UnsupportedDocumentFormatError(
        f"unsupported document format: {suffix or '<none>'}"
    )
