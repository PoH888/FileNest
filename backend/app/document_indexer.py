"""将工作区文件条目解析、分块并持久化为文档索引。"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from .document_chunker import chunk_document
from .document_parser import parse_document, source_format_for_path
from .filesystem_adapter import FileSystemAdapter
from .models import ChunkRecord, DocumentRecord, FileEntry, Workspace
from .path_policy import PathPolicyError
from .services import (
    WorkspaceNotFoundError,
    require_workspace_read_policy,
)


_PERSISTED_DOCUMENT_EXTENSIONS = frozenset(
    {".md", ".markdown", ".txt", ".pdf", ".docx"}
)


class DocumentIndexWorkspaceNotFoundError(Exception):
    """文档索引所需的工作区不存在。"""


@dataclass(frozen=True, slots=True)
class DocumentIndexResult:
    """一次文档索引 Job 的最小统计结果。"""

    indexed_documents: int
    indexed_chunks: int
    skipped_documents: int


def index_workspace_documents(
    session: Session,
    workspace_id: int,
) -> DocumentIndexResult:
    """将工作区中可持久化格式的文件解析、分块并写入索引。"""

    workspace = session.get(Workspace, workspace_id)
    if workspace is None:
        session.rollback()
        raise DocumentIndexWorkspaceNotFoundError(workspace_id)

    try:
        policy = require_workspace_read_policy(session, workspace_id)
    except WorkspaceNotFoundError as error:
        session.rollback()
        raise DocumentIndexWorkspaceNotFoundError(workspace_id) from error

    file_entries = _find_document_file_entries(session, workspace_id)
    adapter = FileSystemAdapter(
        Path(workspace.root_path),
        workspace_policy=policy,
    )
    logger = logging.getLogger("FileNest")
    indexed_documents = 0
    indexed_chunks = 0
    skipped_documents = 0

    failed_file_entry: tuple[int, str] | None = None
    try:
        for file_entry in file_entries:
            failed_file_entry = (file_entry.id, file_entry.relative_path)
            try:
                adapter.authorized_path(Path(file_entry.relative_path))
            except PathPolicyError as error:
                skipped_documents += 1
                logger.info(
                    "Knowledge 索引排除条目",
                    extra={
                        "workspace_id": workspace_id,
                        "relative_path": file_entry.relative_path,
                        "ignored_reason": error.code.value,
                    },
                )
                failed_file_entry = None
                continue

            document = parse_document(
                adapter,
                workspace_id=workspace_id,
                file_entry_id=file_entry.id,
                source_relative_path=file_entry.relative_path,
            )
            if _document_version_exists(
                session,
                file_entry_id=file_entry.id,
                source_version=document.source_version,
            ):
                skipped_documents += 1
                continue

            chunks = chunk_document(document)
            session.add(DocumentRecord.from_contract(document))
            session.add_all(ChunkRecord.from_contract(chunk) for chunk in chunks)
            indexed_documents += 1
            indexed_chunks += len(chunks)

        session.commit()
    except Exception as error:
        session.rollback()
        if failed_file_entry is not None:
            try:
                _record_failed_ingest(
                    session,
                    workspace_id=workspace_id,
                    file_entry_id=failed_file_entry[0],
                    source_relative_path=failed_file_entry[1],
                    error=error,
                )
                session.commit()
            except Exception:
                session.rollback()
        raise

    return DocumentIndexResult(
        indexed_documents=indexed_documents,
        indexed_chunks=indexed_chunks,
        skipped_documents=skipped_documents,
    )


def _find_document_file_entries(
    session: Session,
    workspace_id: int,
) -> Sequence[FileEntry]:
    """只选择当前持久化模型支持的文档格式，并保持处理顺序稳定。"""

    statement = (
        select(FileEntry)
        .where(
            FileEntry.workspace_id == workspace_id,
            FileEntry.extension.in_(_PERSISTED_DOCUMENT_EXTENSIONS),
        )
        .order_by(FileEntry.relative_path.asc())
    )
    return session.scalars(statement).all()


def _document_version_exists(
    session: Session,
    *,
    file_entry_id: int,
    source_version: str,
) -> bool:
    """避免重复索引同一文件版本，同时保留不同版本的历史记录。"""

    statement = select(DocumentRecord.document_id).where(
        DocumentRecord.file_entry_id == file_entry_id,
        DocumentRecord.source_version == source_version,
    )
    return session.scalar(statement) is not None


def _record_failed_ingest(
    session: Session,
    *,
    workspace_id: int,
    file_entry_id: int,
    source_relative_path: str,
    error: Exception,
) -> None:
    """保存失败证据，但不伪造尚未生成的规范化来源数据。"""

    source_format = source_format_for_path(source_relative_path)
    error_message = str(error).strip() or type(error).__name__
    existing = session.scalar(
        select(DocumentRecord)
        .where(
            DocumentRecord.file_entry_id == file_entry_id,
            DocumentRecord.ingest_status == "failed",
            DocumentRecord.source_version.is_(None),
        )
        .order_by(DocumentRecord.document_id.asc())
    )
    if existing is not None:
        existing.source_relative_path = source_relative_path
        existing.source_format = source_format
        existing.ingest_status = "failed"
        existing.ingest_error = error_message
        return

    session.add(
        DocumentRecord.for_failed_ingest(
            document_id=str(uuid4()),
            workspace_id=workspace_id,
            file_entry_id=file_entry_id,
            source_relative_path=source_relative_path,
            source_format=source_format,
            ingest_error=error_message,
        )
    )
