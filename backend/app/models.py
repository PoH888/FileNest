import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.engine.default import DefaultExecutionContext

from sqlalchemy.orm import Mapped, mapped_column, relationship
# Mapped 用来标记：这个类属性不是普通属性，而是需要映射到数据库字段的 ORM 属性。

from .database import Base
from .document_contracts import (
    Chunk,
    Document,
    DocumentPage,
    DocumentPosition,
)
from .embedding_client import (
    EmbeddingVector,
    InvalidEmbeddingVectorError,
    validate_embedding_vector,
)
from .operation_status import OperationStatus


agent_run_sessions = Table(
    "agent_run_sessions",
    Base.metadata,
    Column(
        "agent_run_id",
        ForeignKey("agent_runs.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column(
        "agent_session_id",
        ForeignKey("agent_sessions.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
)


class Workspace(Base): # 继承 FileNest 的 ORM 基类
    """告诉 SQLAlchemy：
    FileNest 有一种叫 Workspace 的数据库对象。"""
    __tablename__ = "workspaces" # 保存在 SQLite 的 workspaces 表

    id: Mapped[int] = mapped_column(primary_key=True)
#       ORM 映射属性；Python 中对应整数 int
#                     该字段是主键

    name: Mapped[str] = mapped_column(String, nullable=False)
#                                     数据库不允许这个字段保存 NULL

    root_path: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True, # 给 root_path 添加唯一约束，SQLite 会拒绝第二条记录
    )


class FileEntry(Base):
    """工作区内一个文件的持久化索引记录。"""

    __tablename__ = "file_entries"
    __table_args__ = (
        # 同一相对路径只能代表工作区内的一个文件，但不同工作区可以使用相同路径。
        UniqueConstraint(
            "workspace_id",
            "relative_path",
            name="uq_file_entries_workspace_relative_path",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int] = mapped_column(
        ForeignKey("workspaces.id"),
        nullable=False,
    )
    relative_path: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    extension: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mtime_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)


class DocumentRecord(Base):
    """规范化文档及其 ingestion 状态的持久化记录。"""

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "source_format IN ('markdown', 'text', 'pdf', 'docx')",
            name="ck_documents_source_format",
        ),
        CheckConstraint(
            "ingest_status IN ('pending', 'parsing', 'indexed', 'failed')",
            name="ck_documents_ingest_status",
        ),
        CheckConstraint(
            "(ingest_status = 'failed' AND ingest_error IS NOT NULL) "
            "OR (ingest_status <> 'failed' AND ingest_error IS NULL)",
            name="ck_documents_ingest_error",
        ),
        UniqueConstraint(
            "file_entry_id",
            "source_version",
            name="uq_documents_file_entry_source_version",
        ),
    )

    document_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace_id: Mapped[int] = mapped_column(
        ForeignKey("workspaces.id"),
        nullable=False,
        index=True,
    )
    file_entry_id: Mapped[int] = mapped_column(
        ForeignKey("file_entries.id"),
        nullable=False,
        index=True,
    )
    source_relative_path: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    ingest_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        server_default="pending",
    )
    source_format: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_version: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    source_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    indexed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    ingest_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    pages: Mapped[list["DocumentPageRecord"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentPageRecord.page_number",
    )
    source_positions: Mapped[list["DocumentPositionRecord"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentPositionRecord.position_index",
    )
    chunks: Mapped[list["ChunkRecord"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="ChunkRecord.chunk_index",
    )

    @classmethod
    def from_contract(cls, document: Document) -> "DocumentRecord":
        """将已验证的 Document 转换为待提交的 ORM 记录。"""

        record = cls(
            document_id=str(document.document_id),
            workspace_id=document.workspace_id,
            file_entry_id=document.file_entry_id,
            source_relative_path=document.source_relative_path,
            ingest_status="indexed",
            source_format=document.source_format,
            normalized_text=document.normalized_text,
            source_version=document.source_version,
            source_updated_at=document.source_updated_at,
            indexed_at=datetime.now(timezone.utc),
        )
        record.pages = [
            DocumentPageRecord.from_contract(
                document_id=str(document.document_id),
                page=page,
            )
            for page in document.pages
        ]
        record.source_positions = [
            DocumentPositionRecord.from_contract(
                document_id=str(document.document_id),
                position_index=position_index,
                position=position,
            )
            for position_index, position in enumerate(document.source_positions)
        ]
        return record

    @classmethod
    def for_failed_ingest(
        cls,
        *,
        document_id: str,
        workspace_id: int,
        file_entry_id: int,
        source_relative_path: str,
        source_format: str,
        ingest_error: str,
    ) -> "DocumentRecord":
        """创建尚未生成规范化正文但已保存失败证据的记录。"""

        return cls(
            document_id=document_id,
            workspace_id=workspace_id,
            file_entry_id=file_entry_id,
            source_relative_path=source_relative_path,
            source_format=source_format,
            normalized_text="",
            source_version=None,
            source_updated_at=None,
            ingest_status="failed",
            ingest_error=ingest_error,
        )


class DocumentPositionRecord(Base):
    """DOCX 结构位置在规范化正文中的持久化记录。"""

    __tablename__ = "document_positions"
    __table_args__ = (
        CheckConstraint(
            "position_index >= 0",
            name="ck_document_positions_index_non_negative",
        ),
        CheckConstraint(
            "element_type IN ('paragraph', 'table_cell')",
            name="ck_document_positions_element_type",
        ),
        CheckConstraint(
            "start_offset >= 0 AND end_offset >= start_offset",
            name="ck_document_positions_offset_order",
        ),
        CheckConstraint(
            "heading_level IS NULL OR heading_level >= 1",
            name="ck_document_positions_heading_level",
        ),
        CheckConstraint(
            "section_index IS NULL OR section_index >= 0",
            name="ck_document_positions_section_index",
        ),
        CheckConstraint(
            "paragraph_index IS NULL OR paragraph_index >= 0",
            name="ck_document_positions_paragraph_index",
        ),
        CheckConstraint(
            "table_index IS NULL OR table_index >= 0",
            name="ck_document_positions_table_index",
        ),
        CheckConstraint(
            "row_index IS NULL OR row_index >= 0",
            name="ck_document_positions_row_index",
        ),
        CheckConstraint(
            "cell_index IS NULL OR cell_index >= 0",
            name="ck_document_positions_cell_index",
        ),
        UniqueConstraint(
            "document_id",
            "position_index",
            name="uq_document_positions_document_index",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id"),
        nullable=False,
        index=True,
    )
    position_index: Mapped[int] = mapped_column(nullable=False)
    element_type: Mapped[str] = mapped_column(String(20), nullable=False)
    start_offset: Mapped[int] = mapped_column(nullable=False)
    end_offset: Mapped[int] = mapped_column(nullable=False)
    section_index: Mapped[int | None] = mapped_column(nullable=True)
    heading_level: Mapped[int | None] = mapped_column(nullable=True)
    paragraph_index: Mapped[int | None] = mapped_column(nullable=True)
    table_index: Mapped[int | None] = mapped_column(nullable=True)
    row_index: Mapped[int | None] = mapped_column(nullable=True)
    cell_index: Mapped[int | None] = mapped_column(nullable=True)
    document: Mapped["DocumentRecord"] = relationship(
        back_populates="source_positions",
    )

    @classmethod
    def from_contract(
        cls,
        *,
        document_id: str,
        position_index: int,
        position: DocumentPosition,
    ) -> "DocumentPositionRecord":
        """将已验证的 DOCX 位置转换为待提交的 ORM 记录。"""

        return cls(
            document_id=document_id,
            position_index=position_index,
            element_type=position.element_type,
            start_offset=position.start_offset,
            end_offset=position.end_offset,
            section_index=position.section_index,
            heading_level=position.heading_level,
            paragraph_index=position.paragraph_index,
            table_index=position.table_index,
            row_index=position.row_index,
            cell_index=position.cell_index,
        )


class DocumentPageRecord(Base):
    """PDF 页码及其在规范化正文中的可追踪区间。"""

    __tablename__ = "document_pages"
    __table_args__ = (
        CheckConstraint(
            "page_number >= 1",
            name="ck_document_pages_number_positive",
        ),
        CheckConstraint(
            "start_offset >= 0 AND end_offset >= start_offset",
            name="ck_document_pages_offset_order",
        ),
        UniqueConstraint(
            "document_id",
            "page_number",
            name="uq_document_pages_document_number",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id"),
        nullable=False,
        index=True,
    )
    page_number: Mapped[int] = mapped_column(nullable=False)
    start_offset: Mapped[int] = mapped_column(nullable=False)
    end_offset: Mapped[int] = mapped_column(nullable=False)
    document: Mapped["DocumentRecord"] = relationship(
        back_populates="pages",
    )

    @classmethod
    def from_contract(
        cls,
        *,
        document_id: str,
        page: DocumentPage,
    ) -> "DocumentPageRecord":
        """将已验证的 PDF 页元数据转换为待提交的 ORM 记录。"""

        return cls(
            document_id=document_id,
            page_number=page.page_number,
            start_offset=page.start_offset,
            end_offset=page.end_offset,
        )


class ChunkRecord(Base):
    """文档片段的来源位置与顺序持久化记录。"""

    __tablename__ = "document_chunks"
    __table_args__ = (
        CheckConstraint(
            "chunk_index >= 0",
            name="ck_document_chunks_index_non_negative",
        ),
        CheckConstraint(
            "start_offset >= 0 AND end_offset > start_offset",
            name="ck_document_chunks_offset_order",
        ),
        CheckConstraint(
            "start_line >= 1 AND end_line >= start_line",
            name="ck_document_chunks_line_order",
        ),
        CheckConstraint(
            "(page_start IS NULL AND page_end IS NULL) "
            "OR (page_start >= 1 AND page_end >= page_start)",
            name="ck_document_chunks_page_range",
        ),
        CheckConstraint(
            "source_positions_json IS NULL OR length(source_positions_json) > 0",
            name="ck_document_chunks_source_positions_non_empty",
        ),
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunks_document_index",
        ),
        UniqueConstraint(
            "chunk_id",
            name="uq_document_chunks_chunk_id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String(36), nullable=False)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id"),
        nullable=False,
        index=True,
    )
    file_entry_id: Mapped[int] = mapped_column(
        ForeignKey("file_entries.id"),
        nullable=False,
        index=True,
    )
    source_relative_path: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    start_offset: Mapped[int] = mapped_column(nullable=False)
    end_offset: Mapped[int] = mapped_column(nullable=False)
    start_line: Mapped[int] = mapped_column(nullable=False)
    end_line: Mapped[int] = mapped_column(nullable=False)
    source_positions_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    page_start: Mapped[int | None] = mapped_column(nullable=True)
    page_end: Mapped[int | None] = mapped_column(nullable=True)
    document: Mapped["DocumentRecord"] = relationship(
        back_populates="chunks",
    )

    @classmethod
    def from_contract(cls, chunk: Chunk) -> "ChunkRecord":
        """将已验证的 Chunk 转换为待提交的 ORM 记录。"""

        source_positions_json = (
            json.dumps(
                [
                    position.model_dump(mode="json")
                    for position in chunk.source_positions
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if chunk.source_positions
            else None
        )
        return cls(
            chunk_id=str(chunk.chunk_id),
            document_id=str(chunk.document_id),
            file_entry_id=chunk.file_entry_id,
            source_relative_path=chunk.source_relative_path,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            source_positions_json=source_positions_json,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        )


class ChunkEmbeddingRecord(Base):
    """一个文档片段在指定 Embedding 模型下的最小持久化向量记录。"""

    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        CheckConstraint(
            "dimension > 0",
            name="ck_chunk_embeddings_dimension_positive",
        ),
        CheckConstraint(
            "length(embedding_model) > 0",
            name="ck_chunk_embeddings_model_non_empty",
        ),
        CheckConstraint(
            "length(vector_json) > 0",
            name="ck_chunk_embeddings_vector_non_empty",
        ),
        UniqueConstraint(
            "chunk_id",
            "embedding_model",
            name="uq_chunk_embeddings_chunk_model",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("document_chunks.chunk_id"),
        nullable=False,
        index=True,
    )
    embedding_model: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    dimension: Mapped[int] = mapped_column(nullable=False)
    vector_json: Mapped[str] = mapped_column(Text, nullable=False)

    @classmethod
    def from_vector(
        cls,
        *,
        chunk_id: str,
        embedding_model: str,
        vector: Sequence[float],
    ) -> "ChunkEmbeddingRecord":
        """从已验证向量创建可持久化记录，不改变原始片段记录。"""

        if (
            not isinstance(chunk_id, str)
            or not chunk_id
            or chunk_id != chunk_id.strip()
        ):
            raise ValueError("chunk_id must be non-empty without surrounding whitespace")
        if (
            not isinstance(embedding_model, str)
            or not embedding_model
            or embedding_model != embedding_model.strip()
        ):
            raise ValueError(
                "embedding_model must be non-empty without surrounding whitespace"
            )
        if len(embedding_model) > 200:
            raise ValueError("embedding_model must be at most 200 characters")

        validated_vector = validate_embedding_vector(vector)
        return cls(
            chunk_id=chunk_id,
            embedding_model=embedding_model,
            dimension=len(validated_vector),
            vector_json=json.dumps(
                validated_vector,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    @property
    def vector(self) -> EmbeddingVector:
        """读取并重新校验持久化向量，损坏时明确失败。"""

        try:
            raw_vector = json.loads(self.vector_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise InvalidEmbeddingVectorError(
                "持久化向量 JSON 无效"
            ) from error

        if not isinstance(raw_vector, list):
            raise InvalidEmbeddingVectorError("持久化向量必须是数组")

        vector = validate_embedding_vector(raw_vector)
        if len(vector) != self.dimension:
            raise InvalidEmbeddingVectorError(
                "持久化向量维度与记录不一致"
            )
        return vector


class AgentSession(Base):
    """Agent 多次运行共享的会话级元数据。"""

    __tablename__ = "agent_sessions"
    __table_args__ = (
        CheckConstraint(
            "length(metadata_json) > 0",
            name="ck_agent_sessions_metadata_non_empty",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int | None] = mapped_column(
        ForeignKey("workspaces.id"),
        nullable=True,
    )
    metadata_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="{}",
        server_default="{}",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )

    runs: Mapped[list["AgentRun"]] = relationship(
        "AgentRun",
        secondary=agent_run_sessions,
        back_populates="sessions",
        order_by="AgentRun.started_at",
    )
    steps: Mapped[list["AgentStep"]] = relationship(
        "AgentStep",
        back_populates="agent_session",
        order_by="AgentStep.step_index",
    )
    metrics: Mapped[list["AgentMetric"]] = relationship(
        "AgentMetric",
        back_populates="agent_session",
        order_by="AgentMetric.created_at",
    )


def _default_job_payload(context: DefaultExecutionContext) -> str:
    """兼容旧 ORM 写入，同时只从同一行的 workspace_id 生成参数。"""

    workspace_id = context.get_current_parameters().get("workspace_id")
    if isinstance(workspace_id, bool) or not isinstance(workspace_id, int):
        raise ValueError("JobRecord workspace_id is required for its payload")
    return json.dumps(
        {"workspace_id": workspace_id},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _default_job_payload_hash(context: DefaultExecutionContext) -> str:
    """让旧的 ORM 直接写入也能产生与 payload 对应的摘要。"""

    payload_json = context.get_current_parameters().get("payload_json")
    if not isinstance(payload_json, str):
        payload_json = _default_job_payload(context)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


class AgentStep(Base):
    """Agent 会话内一个可追踪步骤的持久化记录。"""

    __tablename__ = "agent_steps"
    __table_args__ = (
        UniqueConstraint(
            "agent_session_id",
            "step_index",
            name="uq_agent_steps_session_index",
        ),
        CheckConstraint(
            "step_index >= 0",
            name="ck_agent_steps_index_non_negative",
        ),
        CheckConstraint(
            "length(step_type) > 0 AND step_type = trim(step_type)",
            name="ck_agent_steps_type_non_empty",
        ),
        CheckConstraint(
            "length(status) > 0 AND status = trim(status)",
            name="ck_agent_steps_status_non_empty",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_session_id: Mapped[int] = mapped_column(
        ForeignKey("agent_sessions.id"),
        nullable=False,
    )
    step_index: Mapped[int] = mapped_column(nullable=False)
    step_type: Mapped[str] = mapped_column(String(64), nullable=False)
    input: Mapped[str] = mapped_column(Text, nullable=False)
    output_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        server_default="pending",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    agent_session: Mapped[AgentSession] = relationship(
        "AgentSession",
        back_populates="steps",
    )
    messages: Mapped[list["AgentMessage"]] = relationship(
        "AgentMessage",
        back_populates="agent_step",
        order_by="AgentMessage.sequence_no",
    )
    model_runs: Mapped[list["AgentModelRun"]] = relationship(
        "AgentModelRun",
        back_populates="agent_step",
        order_by="AgentModelRun.created_at",
    )
    metrics: Mapped[list["AgentMetric"]] = relationship(
        "AgentMetric",
        back_populates="agent_step",
        order_by="AgentMetric.created_at",
    )
    tool_calls: Mapped[list["AgentToolCall"]] = relationship(
        "AgentToolCall",
        back_populates="agent_step",
        order_by="AgentToolCall.sequence_no",
    )


class AgentMessage(Base):
    """Agent 步骤中的中间消息或工具事件完整快照。"""

    __tablename__ = "agent_messages"
    __table_args__ = (
        UniqueConstraint(
            "agent_step_id",
            "sequence_no",
            name="uq_agent_messages_step_sequence",
        ),
        CheckConstraint(
            "sequence_no >= 0",
            name="ck_agent_messages_sequence_non_negative",
        ),
        CheckConstraint(
            "message_type IN ('user', 'assistant', 'tool_call', 'tool_result')",
            name="ck_agent_messages_type",
        ),
        CheckConstraint(
            "length(payload_json) > 0",
            name="ck_agent_messages_payload_non_empty",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_step_id: Mapped[int] = mapped_column(
        ForeignKey("agent_steps.id"),
        nullable=False,
    )
    sequence_no: Mapped[int] = mapped_column(nullable=False)
    message_type: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    agent_step: Mapped[AgentStep] = relationship(
        "AgentStep",
        back_populates="messages",
    )


class AgentModelRun(Base):
    """一个 Agent 步骤对应的模型调用及其运行信息。"""

    __tablename__ = "agent_model_runs"
    __table_args__ = (
        CheckConstraint(
            "length(model) > 0 AND model = trim(model)",
            name="ck_agent_model_runs_model_non_empty",
        ),
        CheckConstraint(
            "prompt_version IS NULL OR (length(prompt_version) > 0 "
            "AND prompt_version = trim(prompt_version))",
            name="ck_agent_model_runs_prompt_version",
        ),
        CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_agent_model_runs_input_tokens_non_negative",
        ),
        CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_agent_model_runs_output_tokens_non_negative",
        ),
        CheckConstraint(
            "total_tokens IS NULL OR total_tokens >= 0",
            name="ck_agent_model_runs_total_tokens_non_negative",
        ),
        CheckConstraint(
            "(input_tokens IS NULL AND output_tokens IS NULL "
            "AND total_tokens IS NULL) OR "
            "(input_tokens IS NOT NULL AND output_tokens IS NOT NULL "
            "AND total_tokens IS NOT NULL)",
            name="ck_agent_model_runs_token_usage_complete",
        ),
        CheckConstraint(
            "latency_ms >= 0",
            name="ck_agent_model_runs_latency_non_negative",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_step_id: Mapped[int] = mapped_column(
        ForeignKey("agent_steps.id"),
        nullable=False,
    )
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    prompt_version: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    input_tokens: Mapped[int | None] = mapped_column(nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    agent_step: Mapped[AgentStep] = relationship(
        "AgentStep",
        back_populates="model_runs",
    )
    metrics: Mapped[list["AgentMetric"]] = relationship(
        "AgentMetric",
        back_populates="agent_model_run",
        order_by="AgentMetric.created_at",
    )


class AgentMetric(Base):
    """供 Agent 分析与评估使用的可扩展指标记录。"""

    __tablename__ = "agent_metrics"
    __table_args__ = (
        CheckConstraint(
            "length(metric_name) > 0 AND metric_name = trim(metric_name)",
            name="ck_agent_metrics_name_non_empty",
        ),
        CheckConstraint(
            "length(value_json) > 0",
            name="ck_agent_metrics_value_non_empty",
        ),
        CheckConstraint(
            "unit IS NULL OR (length(unit) > 0 AND unit = trim(unit))",
            name="ck_agent_metrics_unit",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_session_id: Mapped[int] = mapped_column(
        ForeignKey("agent_sessions.id"),
        nullable=False,
    )
    agent_step_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_steps.id"),
        nullable=True,
    )
    agent_model_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_model_runs.id"),
        nullable=True,
    )
    metric_name: Mapped[str] = mapped_column(String(128), nullable=False)
    value_json: Mapped[str] = mapped_column(Text, nullable=False)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    agent_session: Mapped[AgentSession] = relationship(
        "AgentSession",
        back_populates="metrics",
    )
    agent_step: Mapped[AgentStep | None] = relationship(
        "AgentStep",
        back_populates="metrics",
    )
    agent_model_run: Mapped[AgentModelRun | None] = relationship(
        "AgentModelRun",
        back_populates="metrics",
    )


class AgentRun(Base):
    """一次 Agent Loop 运行的持久化生命周期与恢复上下文。"""

    __tablename__ = "agent_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'waiting_approval', "
            "'completed', 'max_steps_reached', "
            "'timed_out', 'cancelled', 'failed')",
            name="ck_agent_runs_status",
        ),
        CheckConstraint(
            "model_turns >= 0",
            name="ck_agent_runs_model_turns_non_negative",
        ),
        CheckConstraint(
            "model_provider IS NULL OR (length(model_provider) > 0 "
            "AND model_provider = trim(model_provider))",
            name="ck_agent_runs_model_provider",
        ),
        CheckConstraint(
            "model_name IS NULL OR (length(model_name) > 0 "
            "AND model_name = trim(model_name))",
            name="ck_agent_runs_model_name",
        ),
        CheckConstraint(
            "prompt_version IS NULL OR (length(prompt_version) > 0 "
            "AND prompt_version = trim(prompt_version))",
            name="ck_agent_runs_prompt_version",
        ),
        CheckConstraint(
            "latency_ms IS NULL OR latency_ms >= 0",
            name="ck_agent_runs_latency_non_negative",
        ),
        CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_agent_runs_input_tokens_non_negative",
        ),
        CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_agent_runs_output_tokens_non_negative",
        ),
        CheckConstraint(
            "(input_tokens IS NULL AND output_tokens IS NULL) OR "
            "(input_tokens IS NOT NULL AND output_tokens IS NOT NULL)",
            name="ck_agent_runs_token_usage_complete",
        ),
        CheckConstraint(
            "estimated_cost_usd IS NULL OR estimated_cost_usd >= 0",
            name="ck_agent_runs_estimated_cost_non_negative",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int | None] = mapped_column(
        ForeignKey("workspaces.id"),
        nullable=True,
    )
    request_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    context_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="running",
        server_default="running",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    model_turns: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        server_default="0",
    )
    error_code: Mapped[str | None] = mapped_column(String, nullable=True)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    sources_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_provider: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    model_name: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )
    prompt_version: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    latency_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    input_tokens: Mapped[int | None] = mapped_column(nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(nullable=True)
    estimated_cost_usd: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 10),
        nullable=True,
    )
    sessions: Mapped[list[AgentSession]] = relationship(
        "AgentSession",
        secondary=agent_run_sessions,
        back_populates="runs",
        order_by="AgentSession.created_at",
    )
    tool_calls: Mapped[list["AgentToolCall"]] = relationship(
        "AgentToolCall",
        back_populates="agent_run",
        order_by="AgentToolCall.sequence_no",
    )
    operation_plans: Mapped[list["OperationPlanRecord"]] = relationship(
        back_populates="agent_run",
        order_by="OperationPlanRecord.created_at",
    )


class AgentToolCall(Base):
    """一次 Agent Run 中可观察但不含原始参数和结果的工具调用。"""

    __tablename__ = "agent_tool_calls"
    __table_args__ = (
        UniqueConstraint(
            "agent_run_id",
            "sequence_no",
            name="uq_agent_tool_calls_run_sequence",
        ),
        UniqueConstraint(
            "agent_run_id",
            "model_call_id",
            name="uq_agent_tool_calls_run_model_call_id",
        ),
        CheckConstraint(
            "sequence_no >= 1",
            name="ck_agent_tool_calls_sequence_positive",
        ),
        CheckConstraint(
            "status IN ('requested', 'succeeded', 'rejected', 'failed')",
            name="ck_agent_tool_calls_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_run_id: Mapped[int] = mapped_column(
        ForeignKey("agent_runs.id"),
        nullable=False,
    )
    agent_step_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_steps.id", ondelete="RESTRICT"),
        nullable=True,
    )
    sequence_no: Mapped[int] = mapped_column(nullable=False)
    model_call_id: Mapped[str] = mapped_column(String, nullable=False)
    tool_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="requested",
        server_default="requested",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_code: Mapped[str | None] = mapped_column(String, nullable=True)
    agent_run: Mapped[AgentRun] = relationship(
        "AgentRun",
        back_populates="tool_calls",
    )
    agent_step: Mapped[AgentStep | None] = relationship(
        "AgentStep",
        back_populates="tool_calls",
    )


class OperationPlanRecord(Base):
    """一个独立于 workflow checkpoint 的不可变操作计划记录。"""

    __tablename__ = "operation_plans"
    __table_args__ = (
        CheckConstraint(
            "schema_version = 1",
            name="ck_operation_plans_schema_version",
        ),
        CheckConstraint(
            "operation_type IN ('move', 'quarantine', 'rename')",
            name="ck_operation_plans_operation_type",
        ),
        CheckConstraint(
            "status IN ('WAITING_APPROVAL', 'APPROVED', 'REJECTED', 'CANCELLED', 'SUPERSEDED')",
            name="ck_operation_plans_status",
        ),
        CheckConstraint(
            "parent_plan_id IS NULL OR parent_plan_id <> plan_id",
            name="ck_operation_plans_parent_not_self",
        ),
    )

    plan_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    schema_version: Mapped[int] = mapped_column(
        nullable=False,
        default=1,
        server_default="1",
    )
    workspace_id: Mapped[int] = mapped_column(
        ForeignKey("workspaces.id"),
        nullable=False,
        index=True,
    )
    agent_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_runs.id"),
        nullable=True,
        index=True,
    )
    workflow_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
    )
    operation_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="move",
        server_default="move",
    )
    metadata_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="{}",
        server_default="{}",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="WAITING_APPROVAL",
        server_default="WAITING_APPROVAL",
    )
    parent_plan_id: Mapped[str | None] = mapped_column(
        ForeignKey("operation_plans.plan_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    items: Mapped[list["OperationItemRecord"]] = relationship(
        back_populates="plan",
        cascade="all, delete-orphan",
        order_by="OperationItemRecord.sequence_no",
    )
    agent_run: Mapped["AgentRun | None"] = relationship(
        back_populates="operation_plans",
    )


class OperationItemRecord(Base):
    """一个计划中的文件操作及其生成时的源文件校验信息。"""

    __tablename__ = "operation_items"
    __table_args__ = (
        UniqueConstraint(
            "plan_id",
            "sequence_no",
            name="uq_operation_items_plan_sequence",
        ),
        CheckConstraint(
            "sequence_no >= 1",
            name="ck_operation_items_sequence_positive",
        ),
        CheckConstraint(
            "operation_type IN ('move', 'quarantine', 'rename')",
            name="ck_operation_items_operation_type",
        ),
        CheckConstraint(
            "status IN ('PENDING', 'COMPLETED', 'FAILED', 'UNDONE')",
            name="ck_operation_items_status",
        ),
        CheckConstraint(
            "source_file_id >= 1",
            name="ck_operation_items_source_file_positive",
        ),
        CheckConstraint(
            "source_size_bytes >= 0 AND source_mtime_ns >= 0",
            name="ck_operation_items_source_metadata",
        ),
        CheckConstraint(
            "(source_hash_algorithm IS NULL AND source_sha256 IS NULL) OR "
            "(source_hash_algorithm = 'sha256' AND source_sha256 IS NOT NULL "
            "AND length(source_sha256) = 64)",
            name="ck_operation_items_source_hash_pair",
        ),
        CheckConstraint(
            "reason_kind IN ('matched_candidate', 'manual_selection')",
            name="ck_operation_items_reason_kind",
        ),
        CheckConstraint(
            "length(reason_description) BETWEEN 1 AND 500 "
            "AND reason_description = trim(reason_description)",
            name="ck_operation_items_reason_description",
        ),
        CheckConstraint(
            "(reason_kind = 'matched_candidate' AND reason_match_score IS NOT NULL "
            "AND reason_match_score BETWEEN 0 AND 100) OR "
            "(reason_kind = 'manual_selection' AND reason_match_score IS NULL)",
            name="ck_operation_items_reason_score",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    plan_id: Mapped[str] = mapped_column(
        ForeignKey("operation_plans.plan_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence_no: Mapped[int] = mapped_column(nullable=False)
    operation_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="move",
        server_default="move",
    )
    source_file_id: Mapped[int] = mapped_column(nullable=False)
    source_relative_path: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    target_relative_path: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    source_size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    source_mtime_ns: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    source_hash_algorithm: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
    )
    source_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    reason_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    reason_description: Mapped[str] = mapped_column(String(500), nullable=False)
    reason_match_score: Mapped[int | None] = mapped_column(nullable=True)
    risks_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="[]",
        server_default="[]",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="PENDING",
        server_default="PENDING",
    )
    plan: Mapped[OperationPlanRecord] = relationship(back_populates="items")


class ApprovalRequest(Base):
    """一个等待人工决定的文件操作计划。"""

    __tablename__ = "approval_requests"
    __table_args__ = (
        UniqueConstraint(
            "workflow_id",
            name="uq_approval_requests_workflow_id",
        ),
        CheckConstraint(
            "status IN ('WAITING_APPROVAL', 'APPROVED', 'REJECTED', 'CANCELLED')",
            name="ck_approval_requests_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    workflow_id: Mapped[str] = mapped_column(String(36), nullable=False)
    plan_id: Mapped[str] = mapped_column(String(36), nullable=False)
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="WAITING_APPROVAL",
        server_default="WAITING_APPROVAL",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )


class ApprovalAuditEvent(Base):
    """一次只追加、不保存原始用户文本的审批转换记录。"""

    __tablename__ = "approval_audit_events"
    __table_args__ = (
        CheckConstraint(
            "action IN ('approve', 'edit', 'reject', 'cancel')",
            name="ck_approval_audit_events_action",
        ),
        CheckConstraint(
            "previous_status IN "
            "('WAITING_APPROVAL', 'APPROVED', 'REJECTED', 'CANCELLED')",
            name="ck_approval_audit_events_previous_status",
        ),
        CheckConstraint(
            "next_status IN "
            "('WAITING_APPROVAL', 'APPROVED', 'REJECTED', 'CANCELLED')",
            name="ck_approval_audit_events_next_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    approval_request_id: Mapped[int] = mapped_column(
        ForeignKey("approval_requests.id"),
        nullable=False,
        index=True,
    )
    action: Mapped[str] = mapped_column(String, nullable=False)
    previous_status: Mapped[str] = mapped_column(String, nullable=False)
    next_status: Mapped[str] = mapped_column(String, nullable=False)
    previous_plan_id: Mapped[str] = mapped_column(String(36), nullable=False)
    next_plan_id: Mapped[str] = mapped_column(String(36), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )


class OperationExecution(Base):
    """一个已经进入安全执行边界的确定操作计划。"""

    __tablename__ = "operation_executions"
    __table_args__ = (
        UniqueConstraint(
            "workflow_id",
            name="uq_operation_executions_workflow_id",
        ),
        UniqueConstraint(
            "plan_id",
            name="uq_operation_executions_plan_id",
        ),
        CheckConstraint(
            "status IN "
            "('EXECUTING', 'PARTIALLY_COMPLETED', 'COMPLETED', "
            "'UNDOING', 'UNDONE', 'FAILED')",
            name="ck_operation_executions_status",
        ),
        CheckConstraint(
            "attempt >= 1",
            name="ck_operation_executions_attempt_positive",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    workflow_id: Mapped[str] = mapped_column(String(36), nullable=False)
    plan_id: Mapped[str] = mapped_column(String(36), nullable=False)
    workspace_id: Mapped[int] = mapped_column(
        ForeignKey("workspaces.id"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="EXECUTING",
        server_default="EXECUTING",
    )
    attempt: Mapped[int] = mapped_column(
        nullable=False,
        default=1,
        server_default="1",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    undone_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    @property
    def idempotency_key(self) -> str:
        """复用不可变且唯一的计划标识，避免另一套执行身份发生漂移。"""

        return self.plan_id


class OperationExecutionItem(Base):
    """一个文件操作的 before、after 与 undo 持久化证据。"""

    __tablename__ = "operation_execution_items"
    __table_args__ = (
        UniqueConstraint(
            "execution_id",
            "sequence_no",
            name="uq_operation_execution_items_execution_sequence",
        ),
        CheckConstraint(
            "sequence_no >= 1",
            name="ck_operation_execution_items_sequence_positive",
        ),
        CheckConstraint(
            "operation_type IN ('move', 'quarantine', 'rename')",
            name="ck_operation_execution_items_type",
        ),
        CheckConstraint(
            "before_location IN ('workspace', 'quarantine')",
            name="ck_operation_execution_items_before_location",
        ),
        CheckConstraint(
            "after_location IN ('workspace', 'quarantine')",
            name="ck_operation_execution_items_after_location",
        ),
        CheckConstraint(
            "status IN ('PENDING', 'EXECUTING', 'COMPLETED', 'UNDOING', 'UNDONE', "
            "'FAILED')",
            name="ck_operation_execution_items_status",
        ),
        CheckConstraint(
            "before_size_bytes >= 0 AND before_mtime_ns >= 0",
            name="ck_operation_execution_items_before_metadata",
        ),
        CheckConstraint(
            "(after_size_bytes IS NULL OR after_size_bytes >= 0) AND "
            "(after_mtime_ns IS NULL OR after_mtime_ns >= 0)",
            name="ck_operation_execution_items_after_metadata",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    execution_id: Mapped[int] = mapped_column(
        ForeignKey("operation_executions.id"),
        nullable=False,
        index=True,
    )
    sequence_no: Mapped[int] = mapped_column(nullable=False)
    operation_type: Mapped[str] = mapped_column(String, nullable=False)
    source_file_id: Mapped[int] = mapped_column(nullable=False)
    before_location: Mapped[str] = mapped_column(String, nullable=False)
    before_relative_path: Mapped[str] = mapped_column(String, nullable=False)
    before_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    before_mtime_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    before_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    after_location: Mapped[str] = mapped_column(String, nullable=False)
    after_relative_path: Mapped[str] = mapped_column(String, nullable=False)
    after_size_bytes: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )
    after_mtime_ns: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )
    after_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    undo_source_relative_path: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    undo_target_relative_path: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="PENDING",
        server_default="PENDING",
    )
    error_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    failed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    undone_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class OperationStatusRecord(Base):
    """独立持久化的 Operation 当前总体状态与关联标识。"""

    __tablename__ = "operation_statuses"
    __table_args__ = (
        CheckConstraint(
            "overall_status IN "
            "('PROPOSED', 'WAITING_APPROVAL', 'APPROVED', 'REJECTED', "
            "'CANCELLED', 'EXECUTING', 'PARTIAL_FAILED', 'COMPLETED', "
            "'UNDOING', 'UNDONE', 'COMPENSATED', 'FAILED')",
            name="ck_operation_statuses_overall_status",
        ),
        CheckConstraint(
            "revision >= 0",
            name="ck_operation_statuses_revision_non_negative",
        ),
    )

    workflow_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    plan_id: Mapped[str] = mapped_column(
        ForeignKey("operation_plans.plan_id"),
        nullable=False,
        index=True,
    )
    approval_id: Mapped[int | None] = mapped_column(
        ForeignKey("approval_requests.id"),
        nullable=True,
        index=True,
    )
    execution_id: Mapped[int | None] = mapped_column(
        ForeignKey("operation_executions.id"),
        nullable=True,
        index=True,
    )
    overall_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=OperationStatus.PROPOSED.value,
        server_default=OperationStatus.PROPOSED.value,
    )
    revision: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        server_default="0",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )


class JobRecord(Base):
    """一个可跨进程重启恢复的逻辑后台任务。"""

    __tablename__ = "background_jobs"
    __table_args__ = (
        UniqueConstraint(
            "idempotency_key",
            name="uq_background_jobs_idempotency_key",
        ),
        CheckConstraint(
            "schema_version = 1",
            name="ck_background_jobs_schema_version",
        ),
        CheckConstraint(
            "length(task_version) BETWEEN 1 AND 32 "
            "AND task_version = trim(task_version)",
            name="ck_background_jobs_task_version",
        ),
        CheckConstraint(
            "length(payload_json) > 0",
            name="ck_background_jobs_payload_json",
        ),
        CheckConstraint(
            "length(payload_hash) = 64 AND payload_hash = lower(payload_hash)",
            name="ck_background_jobs_payload_hash",
        ),
        CheckConstraint(
            "kind IN ('workspace_scan', 'document_index')",
            name="ck_background_jobs_kind",
        ),
        CheckConstraint(
            "status IN ('pending', 'running', 'cancel_requested', "
            "'succeeded', 'failed', 'cancelled')",
            name="ck_background_jobs_status",
        ),
        CheckConstraint(
            "length(idempotency_key) BETWEEN 1 AND 128 "
            "AND idempotency_key = trim(idempotency_key)",
            name="ck_background_jobs_idempotency_key",
        ),
        CheckConstraint(
            "max_attempts BETWEEN 1 AND 10",
            name="ck_background_jobs_max_attempts",
        ),
        CheckConstraint(
            "revision >= 0",
            name="ck_background_jobs_revision_non_negative",
        ),
        CheckConstraint(
            "((status IN ('succeeded', 'failed', 'cancelled')) "
            "AND finished_at IS NOT NULL) OR "
            "((status IN ('pending', 'running', 'cancel_requested')) "
            "AND finished_at IS NULL)",
            name="ck_background_jobs_finished_at",
        ),
        CheckConstraint(
            "(status = 'failed' AND error_code IS NOT NULL) OR "
            "(status <> 'failed' AND error_code IS NULL)",
            name="ck_background_jobs_error_code",
        ),
        CheckConstraint(
            "status NOT IN ('cancel_requested', 'cancelled') "
            "OR cancel_requested_at IS NOT NULL",
            name="ck_background_jobs_cancellation",
        ),
    )

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    schema_version: Mapped[int] = mapped_column(
        nullable=False,
        default=1,
        server_default="1",
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    task_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="v1",
        server_default="v1",
    )
    workspace_id: Mapped[int] = mapped_column(
        ForeignKey("workspaces.id"),
        nullable=False,
        index=True,
    )
    payload_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default=_default_job_payload,
    )
    payload_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=_default_job_payload_hash,
    )
    idempotency_key: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        server_default="pending",
    )
    max_attempts: Mapped[int] = mapped_column(nullable=False)
    revision: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        server_default="0",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    cancel_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )


class JobAttemptRecord(Base):
    """一次不可覆盖的后台任务执行尝试及其最新进度。"""

    __tablename__ = "background_job_attempts"
    __table_args__ = (
        UniqueConstraint(
            "job_id",
            "attempt_no",
            name="uq_background_job_attempts_job_number",
        ),
        CheckConstraint(
            "schema_version = 1",
            name="ck_background_job_attempts_schema_version",
        ),
        CheckConstraint(
            "attempt_no >= 1",
            name="ck_background_job_attempts_number_positive",
        ),
        CheckConstraint(
            "status IN ('running', 'succeeded', 'failed', "
            "'cancelled', 'interrupted')",
            name="ck_background_job_attempts_status",
        ),
        CheckConstraint(
            "completed_units >= 0 AND "
            "(total_units IS NULL OR (total_units >= 0 "
            "AND completed_units <= total_units))",
            name="ck_background_job_attempts_progress",
        ),
        CheckConstraint(
            "length(phase_code) BETWEEN 1 AND 64 "
            "AND phase_code = trim(phase_code)",
            name="ck_background_job_attempts_phase_code",
        ),
        CheckConstraint(
            "(status = 'running' AND finished_at IS NULL) OR "
            "(status <> 'running' AND finished_at IS NOT NULL)",
            name="ck_background_job_attempts_finished_at",
        ),
        CheckConstraint(
            "(status IN ('failed', 'interrupted') "
            "AND error_code IS NOT NULL) OR "
            "(status NOT IN ('failed', 'interrupted') "
            "AND error_code IS NULL)",
            name="ck_background_job_attempts_error_code",
        ),
        CheckConstraint(
            "(status = 'interrupted' AND retryable = 1) OR "
            "(status IN ('running', 'succeeded', 'cancelled') "
            "AND retryable = 0) OR status = 'failed'",
            name="ck_background_job_attempts_retryable",
        ),
    )

    attempt_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    schema_version: Mapped[int] = mapped_column(
        nullable=False,
        default=1,
        server_default="1",
    )
    job_id: Mapped[str] = mapped_column(
        ForeignKey("background_jobs.job_id"),
        nullable=False,
        index=True,
    )
    attempt_no: Mapped[int] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="running",
        server_default="running",
    )
    completed_units: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        server_default="0",
    )
    total_units: Mapped[int | None] = mapped_column(nullable=True)
    phase_code: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="starting",
        server_default="starting",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    retryable: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="0",
    )
