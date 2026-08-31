"""Knowledge 文档查询 API。"""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .database import get_session
from .document_contracts import DocumentPosition
from .models import ChunkEmbeddingRecord, ChunkRecord, DocumentRecord, FileEntry
from .retrieval_context import (
    RetrievalContextError,
    build_retrieval_context_from_records,
)
from .services import get_workspace as get_workspace_service


router = APIRouter(prefix="/api/v1/knowledge")


class KnowledgeDocumentListItem(BaseModel):
    """Knowledge 文档列表中的公开字段。"""

    document_id: UUID
    workspace_id: int
    file_entry_id: int
    source_relative_path: str
    source_format: str
    ingest_status: str
    source_version: str | None = None
    source_updated_at: datetime | None = None


class KnowledgeDocumentMetadata(BaseModel):
    """文档详情中的基础元数据。"""

    document_id: UUID
    workspace_id: int
    file_entry_id: int
    source_relative_path: str
    source_format: str


class KnowledgeDocumentProvenance(BaseModel):
    """文档详情中的来源追踪元数据。"""

    source_relative_path: str
    source_version: str | None = None
    source_updated_at: datetime | None = None


class KnowledgeDocumentDetail(BaseModel):
    """Knowledge 文档详情的公开投影。"""

    document_id: UUID
    metadata: KnowledgeDocumentMetadata
    ingest_status: str
    ingest_error: str | None = None
    provenance: KnowledgeDocumentProvenance


class KnowledgeSearchRequest(BaseModel):
    """Knowledge 搜索请求的受限参数。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: int = Field(ge=1)
    query: str = Field(min_length=1, max_length=200)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class KnowledgeSearchChunk(BaseModel):
    """搜索返回的文档片段。"""

    chunk_id: str
    document_id: UUID
    text: str
    chunk_index: int
    citation_id: str
    source_version: str | None = None
    source_updated_at: AwareDatetime | None = None
    indexed_at: AwareDatetime | None = None


class KnowledgeSearchDocument(BaseModel):
    """搜索结果涉及的去重文档摘要。"""

    document_id: UUID
    workspace_id: int
    source_relative_path: str
    source_format: str
    ingest_status: str


class KnowledgeSearchRelevance(BaseModel):
    """一个片段的确定性关键词相关性分数。"""

    chunk_id: str
    score: int = Field(ge=1)


class KnowledgeSearchProvenance(BaseModel):
    """一个片段在来源文档中的可复核位置。"""

    chunk_id: str
    document_id: UUID
    citation_id: str
    source_relative_path: str
    source_version: str | None = None
    source_updated_at: AwareDatetime | None = None
    indexed_at: AwareDatetime | None = None
    start_offset: int
    end_offset: int
    start_line: int
    end_line: int
    page_start: int | None = None
    page_end: int | None = None
    source_positions: tuple[DocumentPosition, ...] = ()


class KnowledgeSearchResponse(BaseModel):
    """Knowledge 搜索的结构化结果。"""

    query: str
    total: int = Field(ge=0)
    top_k: int = Field(ge=1, le=10)
    has_more: bool
    chunks: list[KnowledgeSearchChunk]
    documents: list[KnowledgeSearchDocument]
    relevance: list[KnowledgeSearchRelevance]
    provenance: list[KnowledgeSearchProvenance]
    retrieved_at: AwareDatetime
    snapshot_hash: str


@router.get("/documents", response_model=list[KnowledgeDocumentListItem])
def list_knowledge_documents(
    workspace_id: Annotated[int | None, Query(ge=1)] = None,
    session: Session = Depends(get_session),
) -> list[DocumentRecord]:
    """返回 Knowledge 文档列表，可按工作区筛选。"""

    statement = select(DocumentRecord).order_by(
        DocumentRecord.source_relative_path.asc(),
        DocumentRecord.document_id.asc(),
    )
    if workspace_id is not None:
        statement = statement.where(DocumentRecord.workspace_id == workspace_id)
    return list(session.scalars(statement).all())


@router.get(
    "/documents/{document_id}",
    response_model=KnowledgeDocumentDetail,
)
def get_knowledge_document(
    document_id: UUID,
    session: Session = Depends(get_session),
) -> KnowledgeDocumentDetail:
    """返回 Knowledge 文档的元数据、摄取状态和来源信息。"""

    document = session.get(DocumentRecord, str(document_id))
    if document is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "document_not_found",
                "message": "Knowledge 文档不存在。",
            },
        )

    metadata = KnowledgeDocumentMetadata(
        document_id=document.document_id,
        workspace_id=document.workspace_id,
        file_entry_id=document.file_entry_id,
        source_relative_path=document.source_relative_path,
        source_format=document.source_format,
    )
    provenance = KnowledgeDocumentProvenance(
        source_relative_path=document.source_relative_path,
        source_version=document.source_version,
        source_updated_at=document.source_updated_at,
    )
    return KnowledgeDocumentDetail(
        document_id=document.document_id,
        metadata=metadata,
        ingest_status=document.ingest_status,
        ingest_error=document.ingest_error,
        provenance=provenance,
    )


@router.post("/search", response_model=KnowledgeSearchResponse)
def search_knowledge(
    request: KnowledgeSearchRequest,
    session: Session = Depends(get_session),
) -> KnowledgeSearchResponse:
    """在指定工作区的已索引片段中执行只读关键词搜索。"""

    if get_workspace_service(session, request.workspace_id) is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workspace_not_found",
                "message": "工作区不存在。",
            },
        )

    statement = (
        select(ChunkRecord, DocumentRecord)
        .join(
            FileEntry,
            FileEntry.id == ChunkRecord.file_entry_id,
        )
        .join(
            DocumentRecord,
            DocumentRecord.document_id == ChunkRecord.document_id,
        )
        .where(
            FileEntry.workspace_id == request.workspace_id,
            DocumentRecord.workspace_id == request.workspace_id,
            DocumentRecord.file_entry_id == FileEntry.id,
            DocumentRecord.source_relative_path == FileEntry.relative_path,
            ChunkRecord.source_relative_path == FileEntry.relative_path,
            ChunkRecord.text.icontains(request.query, autoescape=True),
        )
    )
    rows = list(session.execute(statement).all())

    normalized_query = request.query.casefold()
    ranked: list[tuple[int, str, int, str, ChunkRecord, DocumentRecord]] = []
    for chunk, document in rows:
        score = chunk.text.casefold().count(normalized_query)
        if score > 0:
            ranked.append(
                (
                    -score,
                    chunk.source_relative_path,
                    chunk.chunk_index,
                    chunk.chunk_id,
                    chunk,
                    document,
                )
            )
    ranked.sort(key=lambda item: item[:4])

    selected = ranked[: request.top_k]
    try:
        retrieval_context = build_retrieval_context_from_records(
            workspace_id=request.workspace_id,
            query=request.query,
            rows=[
                (chunk, document, -negative_score)
                for negative_score, _, _, _, chunk, document in selected
            ],
            total=len(ranked),
            top_k=request.top_k,
            has_more=len(ranked) > request.top_k,
        )
    except RetrievalContextError as error:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "invalid_retrieval_provenance",
                "message": "检索结果来源证据无效。",
            },
        ) from error

    documents: dict[str, KnowledgeSearchDocument] = {}
    selected_documents = {
        document.document_id: document
        for _, _, _, _, _, document in selected
    }
    chunks: list[KnowledgeSearchChunk] = []
    relevance: list[KnowledgeSearchRelevance] = []
    provenance: list[KnowledgeSearchProvenance] = []
    for chunk in retrieval_context.chunks:
        document = selected_documents[chunk.document_id]
        chunks.append(
            KnowledgeSearchChunk(
                chunk_id=chunk.chunk_id,
                document_id=UUID(chunk.document_id),
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                citation_id=chunk.citation_id,
                source_version=chunk.source_version,
                source_updated_at=chunk.source_updated_at,
                indexed_at=chunk.indexed_at,
            )
        )
        relevance.append(
            KnowledgeSearchRelevance(
                chunk_id=chunk.chunk_id,
                score=chunk.score,
            )
        )
        provenance.append(
            KnowledgeSearchProvenance(
                chunk_id=chunk.chunk_id,
                document_id=UUID(chunk.document_id),
                citation_id=chunk.citation_id,
                source_relative_path=chunk.source_relative_path,
                source_version=chunk.source_version,
                source_updated_at=chunk.source_updated_at,
                indexed_at=chunk.indexed_at,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                source_positions=chunk.source_positions,
            )
        )
        documents.setdefault(
            str(document.document_id),
            KnowledgeSearchDocument(
                document_id=UUID(str(document.document_id)),
                workspace_id=document.workspace_id,
                source_relative_path=document.source_relative_path,
                source_format=document.source_format,
                ingest_status=document.ingest_status,
            ),
        )

    return KnowledgeSearchResponse(
        query=request.query,
        total=len(ranked),
        top_k=request.top_k,
        has_more=len(ranked) > request.top_k,
        chunks=chunks,
        documents=list(documents.values()),
        relevance=relevance,
        provenance=provenance,
        retrieved_at=retrieval_context.retrieved_at,
        snapshot_hash=retrieval_context.snapshot_hash,
    )


@router.delete("/documents/{document_id}", status_code=204)
def delete_knowledge_document(
    document_id: UUID,
    session: Session = Depends(get_session),
) -> Response:
    """删除知识索引及其派生数据，但保留用户原始文件。"""

    document = session.get(DocumentRecord, str(document_id))
    if document is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "document_not_found",
                "message": "Knowledge 文档不存在。",
            },
        )

    chunk_ids = select(ChunkRecord.chunk_id).where(
        ChunkRecord.document_id == str(document_id),
    )
    session.execute(
        delete(ChunkEmbeddingRecord).where(
            ChunkEmbeddingRecord.chunk_id.in_(chunk_ids),
        )
    )
    session.delete(document)
    session.commit()
    return Response(status_code=204)
