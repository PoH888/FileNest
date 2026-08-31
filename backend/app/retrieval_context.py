"""Knowledge 检索结果到统一 RetrievalContext 的转换边界。"""

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import json

from pydantic import AwareDatetime, TypeAdapter, ValidationError

from .document_contracts import (
    DocumentPosition,
    RetrievedChunk,
    RetrievalContext,
)
from .models import ChunkRecord, DocumentRecord


class RetrievalContextError(ValueError):
    """检索记录无法安全构造成 workspace 内证据快照。"""


def build_retrieval_context_from_records(
    *,
    workspace_id: int,
    query: str,
    rows: Iterable[tuple[ChunkRecord, DocumentRecord, int]],
    total: int,
    top_k: int,
    has_more: bool,
    retrieved_at: datetime | str | None = None,
    snapshot_hash: str | None = None,
) -> RetrievalContext:
    """将已按 workspace 查询的数据库行转换为统一快照。"""

    observed_at = _as_aware(retrieved_at or datetime.now(timezone.utc))
    chunks = tuple(
        _retrieved_chunk_from_record(
            workspace_id=workspace_id,
            chunk=chunk,
            document=document,
            score=score,
            observed_at=observed_at,
        )
        for chunk, document, score in rows
    )
    try:
        return RetrievalContext(
            workspace_id=workspace_id,
            query=query,
            total=total,
            top_k=top_k,
            has_more=has_more,
            retrieved_at=observed_at,
            chunks=chunks,
            snapshot_hash=snapshot_hash,
        )
    except ValidationError as error:
        raise RetrievalContextError("检索结果快照校验失败") from error


def build_retrieval_context_from_items(
    *,
    workspace_id: int,
    query: str,
    items: Iterable[Mapping[str, object]],
    total: int,
    top_k: int,
    has_more: bool,
    retrieved_at: datetime | str | None = None,
    snapshot_hash: str | None = None,
) -> RetrievalContext:
    """重新校验工具返回的 JSON 后构造成同一快照契约。"""

    observed_at = _as_aware(retrieved_at or datetime.now(timezone.utc))
    chunks: list[RetrievedChunk] = []
    for raw_item in items:
        raw_chunk_id = raw_item.get("chunk_id")
        if not isinstance(raw_chunk_id, str):
            raise RetrievalContextError("工具结果缺少有效 chunk_id")
        data: dict[str, object] = {
            "workspace_id": raw_item.get("workspace_id", workspace_id),
            "document_id": raw_item.get("document_id"),
            "chunk_id": raw_chunk_id,
            "citation_id": raw_item.get(
                "citation_id",
                f"cite_{raw_chunk_id}",
            ),
            "file_id": raw_item.get("file_id"),
            "source_relative_path": raw_item.get("source_relative_path"),
            "text": raw_item.get("text"),
            "chunk_index": raw_item.get("chunk_index"),
            "source_version": raw_item.get("source_version"),
            "source_updated_at": raw_item.get("source_updated_at"),
            "indexed_at": raw_item.get("indexed_at", observed_at),
            "score": raw_item.get("score"),
            "start_offset": raw_item.get("start_offset"),
            "end_offset": raw_item.get("end_offset"),
            "start_line": raw_item.get("start_line"),
            "end_line": raw_item.get("end_line"),
            "page_start": raw_item.get("page_start"),
            "page_end": raw_item.get("page_end"),
            "source_positions": raw_item.get("source_positions", ()),
        }
        try:
            chunks.append(RetrievedChunk.model_validate(data))
        except ValidationError as error:
            raise RetrievalContextError("工具检索结果 provenance 无效") from error

    try:
        return RetrievalContext(
            workspace_id=workspace_id,
            query=query,
            total=total,
            top_k=top_k,
            has_more=has_more,
            retrieved_at=observed_at,
            chunks=tuple(chunks),
            snapshot_hash=snapshot_hash,
        )
    except ValidationError as error:
        raise RetrievalContextError("工具检索结果快照校验失败") from error


def retrieval_chunk_to_mapping(chunk: RetrievedChunk) -> dict[str, object]:
    """把已验证片段投影为工具/API 可安全序列化的字段。"""

    return chunk.model_dump(mode="json")


def _retrieved_chunk_from_record(
    *,
    workspace_id: int,
    chunk: ChunkRecord,
    document: DocumentRecord,
    score: int,
    observed_at: datetime,
) -> RetrievedChunk:
    if (
        document.workspace_id != workspace_id
        or chunk.document_id != document.document_id
        or chunk.file_entry_id != document.file_entry_id
        or chunk.source_relative_path != document.source_relative_path
    ):
        raise RetrievalContextError("检索记录跨越 workspace 或来源关系不一致")

    try:
        source_positions = _source_positions(chunk.source_positions_json)
        return RetrievedChunk(
            workspace_id=workspace_id,
            document_id=document.document_id,
            chunk_id=chunk.chunk_id,
            citation_id=f"cite_{chunk.chunk_id}",
            file_id=chunk.file_entry_id,
            source_relative_path=chunk.source_relative_path,
            text=chunk.text,
            chunk_index=chunk.chunk_index,
            source_version=document.source_version,
            source_updated_at=_as_optional_aware(document.source_updated_at),
            indexed_at=_as_optional_aware(document.indexed_at),
            score=score,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            source_positions=source_positions,
        )
    except (TypeError, ValidationError) as error:
        raise RetrievalContextError("数据库检索 provenance 无效") from error


def _source_positions(
    source_positions_json: str | None,
) -> tuple[DocumentPosition, ...]:
    if source_positions_json is None:
        return ()
    try:
        raw_positions = json.loads(source_positions_json)
        return TypeAdapter(tuple[DocumentPosition, ...]).validate_python(
            raw_positions
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise RetrievalContextError("DOCX provenance JSON 无效") from error


def _as_aware(value: datetime | str) -> datetime:
    if isinstance(value, str):
        try:
            return TypeAdapter(AwareDatetime).validate_python(value)
        except ValidationError as error:
            raise RetrievalContextError("检索时间戳无效") from error
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _as_optional_aware(value: datetime | None) -> datetime | None:
    return None if value is None else _as_aware(value)
