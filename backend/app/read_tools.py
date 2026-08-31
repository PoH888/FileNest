"""FileNest Agent 可调用的只读业务工具。"""

from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import cast

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from .document_contracts import DocumentPosition, RetrievedChunk
from .models import ChunkRecord, DocumentRecord, FileEntry
from .path_policy import PathPolicyError
from .services import (
    FileEntryNotFoundError,
    WorkspaceNotFoundError,
    get_authorized_file_metadata as get_file_metadata_service,
    get_workspace as get_workspace_service,
    list_workspaces as list_workspaces_service,
    search_files as search_files_service,
)
from .retrieval_context import (
    RetrievalContextError,
    build_retrieval_context_from_records,
    retrieval_chunk_to_mapping,
)
from .tool_contracts import Tool, ToolResult


class ListWorkspacesArguments(BaseModel):
    """列出工作区工具允许模型提供的参数。"""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        """保留现有精确匹配语义，同时拒绝没有业务意义的空名称。"""

        if value is None:
            return None

        normalized = value.strip()
        if not normalized:
            raise ValueError("name must not be blank")
        return normalized


class WorkspaceToolItem(BaseModel):
    """允许 Agent 看见的工作区标识，不包含绝对根路径。"""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: int
    name: str


class ListWorkspacesData(BaseModel):
    """列出工作区工具的成功数据。"""

    model_config = ConfigDict(extra="forbid")

    items: list[WorkspaceToolItem]
    count: int


class SearchFilesArguments(BaseModel):
    """搜索文件工具允许模型提供的受限参数。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: int = Field(ge=1)
    keyword: str = Field(min_length=1, max_length=200)
    extension: str | None = None
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=20, ge=1, le=20)

    @field_validator("keyword")
    @classmethod
    def normalize_keyword(cls, value: str) -> str:
        """拒绝空搜索，避免 Agent 无意读取整个文件索引。"""

        normalized = value.strip()
        if not normalized:
            raise ValueError("keyword must not be blank")
        return normalized

    @field_validator("extension")
    @classmethod
    def normalize_extension(cls, value: str | None) -> str | None:
        """将扩展名转换为现有搜索 Service 使用的规范形式。"""

        if value is None:
            return None

        normalized = value.strip().casefold()
        if not normalized:
            raise ValueError("extension must not be blank")
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        if len(normalized) > 50:
            raise ValueError("extension must not exceed 50 characters")
        return normalized


class SearchFileToolItem(BaseModel):
    """允许 Agent 看见的一条文件索引摘要。"""

    model_config = ConfigDict(extra="forbid")

    file_id: int
    relative_path: str
    name: str
    extension: str
    size_bytes: int = Field(ge=0)
    modified_at: AwareDatetime


class SearchFilesData(BaseModel):
    """搜索文件工具的分页成功数据。"""

    model_config = ConfigDict(extra="forbid")

    items: list[SearchFileToolItem]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    limit: int = Field(ge=1, le=20)
    has_more: bool


class KnowledgeSearchArguments(BaseModel):
    """知识搜索工具允许模型提供的受限参数。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: int = Field(ge=1)
    query: str = Field(min_length=1, max_length=200)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """拒绝空查询并固定查询文本，避免无边界返回文档片段。"""

        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class KnowledgeSearchToolItem(BaseModel):
    """知识搜索返回的原始片段及其文件出处。"""

    model_config = ConfigDict(extra="forbid")

    file_id: int = Field(ge=1)
    workspace_id: int = Field(ge=1)
    name: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    citation_id: str = Field(min_length=1)
    source_relative_path: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    source_version: str | None = None
    source_updated_at: AwareDatetime | None = None
    indexed_at: AwareDatetime | None = None
    source_positions: tuple[DocumentPosition, ...] = ()
    score: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_page_range(self) -> "KnowledgeSearchToolItem":
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("page_start and page_end must be provided together")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end must not be earlier than page_start")
        return self


class KnowledgeSearchData(BaseModel):
    """知识搜索的有上限结果集合。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    items: list[KnowledgeSearchToolItem]
    total: int = Field(ge=0)
    top_k: int = Field(ge=1, le=10)
    has_more: bool
    retrieved_at: AwareDatetime
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class GetFileMetadataArguments(BaseModel):
    """读取单个文件元数据工具允许模型提供的标识。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: int = Field(ge=1)
    file_id: int = Field(ge=1)


class FileMetadataToolData(SearchFileToolItem):
    """通过工作区和路径授权后的单个文件索引元数据。"""

    workspace_id: int


def build_list_workspaces_tool(session: Session) -> Tool:
    """为当前数据库会话构建只读工作区列表工具。"""

    def handle(arguments: BaseModel) -> ToolResult:
        options = cast(ListWorkspacesArguments, arguments)
        workspaces = list_workspaces_service(session, options.name)
        data = ListWorkspacesData(
            items=[
                WorkspaceToolItem.model_validate(workspace)
                for workspace in workspaces
            ],
            count=len(workspaces),
        )
        return ToolResult.success(data.model_dump(mode="json"))

    return Tool(
        name="list_workspaces",
        description="列出 FileNest 已登记的授权工作区，可按名称精确筛选。",
        arguments_model=ListWorkspacesArguments,
        handler=handle,
    )


def build_search_files_tool(session: Session) -> Tool:
    """为当前数据库会话构建有返回数量上限的文件搜索工具。"""

    def handle(arguments: BaseModel) -> ToolResult:
        options = cast(SearchFilesArguments, arguments)

        try:
            result = search_files_service(
                session,
                options.workspace_id,
                keyword=options.keyword,
                extension=options.extension,
                sort_by="relative_path",
                sort_order="asc",
                page=options.page,
                page_size=options.limit,
            )
        except WorkspaceNotFoundError:
            return ToolResult.failure(
                code="workspace_not_found",
                message="工作区不存在",
                details={"workspace_id": options.workspace_id},
            )

        data = SearchFilesData(
            items=[_file_tool_item(item) for item in result.items],
            total=result.total,
            page=result.page,
            limit=result.page_size,
            has_more=result.page * result.page_size < result.total,
        )
        return ToolResult.success(data.model_dump(mode="json"))

    return Tool(
        name="search_files",
        description=(
            "在指定 FileNest 工作区的持久化索引中搜索文件；"
            "每次最多返回 20 条。"
        ),
        arguments_model=SearchFilesArguments,
        handler=handle,
    )


def build_knowledge_search_tool(session: Session) -> Tool:
    """为当前数据库会话构建有上限的只读知识搜索工具。"""

    def handle(arguments: BaseModel) -> ToolResult:
        options = cast(KnowledgeSearchArguments, arguments)

        with session.no_autoflush:
            if get_workspace_service(session, options.workspace_id) is None:
                return ToolResult.failure(
                    code="workspace_not_found",
                    message="工作区不存在",
                    details={"workspace_id": options.workspace_id},
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
                    FileEntry.workspace_id == options.workspace_id,
                    DocumentRecord.workspace_id == options.workspace_id,
                    DocumentRecord.file_entry_id == FileEntry.id,
                    DocumentRecord.source_relative_path
                    == FileEntry.relative_path,
                    ChunkRecord.source_relative_path
                    == FileEntry.relative_path,
                    ChunkRecord.text.icontains(
                        options.query,
                        autoescape=True,
                    ),
                )
                .order_by(
                    ChunkRecord.source_relative_path.asc(),
                    ChunkRecord.chunk_index.asc(),
                    ChunkRecord.chunk_id.asc(),
                )
            )
            rows = list(session.execute(statement).all())

        normalized_query = options.query.casefold()
        ranked_chunks: list[
            tuple[int, str, int, str, ChunkRecord, DocumentRecord]
        ] = []
        for chunk, document in rows:
            score = chunk.text.casefold().count(normalized_query)
            if score > 0:
                ranked_chunks.append(
                    (
                        -score,
                        chunk.source_relative_path,
                        chunk.chunk_index,
                        chunk.chunk_id,
                        chunk,
                        document,
                    )
                )

        ranked_chunks.sort(key=lambda item: item[:4])
        selected = ranked_chunks[: options.top_k]
        try:
            retrieval_context = build_retrieval_context_from_records(
                workspace_id=options.workspace_id,
                query=options.query,
                rows=[
                    (chunk, document, -negative_score)
                    for negative_score, _, _, _, chunk, document in selected
                ],
                total=len(ranked_chunks),
                top_k=options.top_k,
                has_more=len(ranked_chunks) > options.top_k,
            )
        except RetrievalContextError:
            return ToolResult.failure(
                code="invalid_retrieval_provenance",
                message="检索结果来源证据无效",
                details={"workspace_id": options.workspace_id},
            )

        data = KnowledgeSearchData(
            query=options.query,
            items=[
                _knowledge_search_tool_item(chunk)
                for chunk in retrieval_context.chunks
            ],
            total=len(ranked_chunks),
            top_k=options.top_k,
            has_more=len(ranked_chunks) > options.top_k,
            retrieved_at=retrieval_context.retrieved_at,
            snapshot_hash=retrieval_context.snapshot_hash,
        )
        return ToolResult.success(data.model_dump(mode="json"))

    return Tool(
        name="knowledge_search",
        description=(
            "在指定 FileNest 工作区的已索引文档片段中执行只读关键词检索；"
            "最多返回 10 个片段，并保留文件名、行号和字符偏移出处。"
        ),
        arguments_model=KnowledgeSearchArguments,
        handler=handle,
    )


def build_get_file_metadata_tool(session: Session) -> Tool:
    """构建验证工作区归属和路径策略的文件元数据工具。"""

    def handle(arguments: BaseModel) -> ToolResult:
        options = cast(GetFileMetadataArguments, arguments)

        try:
            file_entry = get_file_metadata_service(
                session,
                options.workspace_id,
                options.file_id,
            )
        except WorkspaceNotFoundError:
            return ToolResult.failure(
                code="workspace_not_found",
                message="工作区不存在",
                details={"workspace_id": options.workspace_id},
            )
        except FileEntryNotFoundError:
            return ToolResult.failure(
                code="file_not_found",
                message="文件索引不存在",
                details={
                    "workspace_id": options.workspace_id,
                    "file_id": options.file_id,
                },
            )
        except PathPolicyError as error:
            return ToolResult.failure(
                code=error.code.value,
                message=error.message,
                details={
                    "workspace_id": options.workspace_id,
                    "file_id": options.file_id,
                },
            )

        item = _file_tool_item(file_entry)
        data = FileMetadataToolData(
            workspace_id=options.workspace_id,
            **item.model_dump(),
        )
        return ToolResult.success(data.model_dump(mode="json"))

    return Tool(
        name="get_file_metadata",
        description=(
            "读取指定工作区内一个文件的索引元数据；"
            "返回前会重新验证路径授权。"
        ),
        arguments_model=GetFileMetadataArguments,
        handler=handle,
    )


def _file_tool_item(file_entry: FileEntry) -> SearchFileToolItem:
    """将数据库文件索引收缩为不含绝对路径的工具结果。"""

    seconds, nanoseconds = divmod(file_entry.mtime_ns, 1_000_000_000)
    modified_at = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
        microsecond=nanoseconds // 1_000,
    )
    return SearchFileToolItem(
        file_id=file_entry.id,
        relative_path=file_entry.relative_path,
        name=file_entry.name,
        extension=file_entry.extension,
        size_bytes=file_entry.size_bytes,
        modified_at=modified_at,
    )


def _knowledge_search_tool_item(
    chunk: RetrievedChunk,
) -> KnowledgeSearchToolItem:
    """仅从已验证的 RetrievalContext 投影知识工具结果。"""

    data = retrieval_chunk_to_mapping(chunk)
    data["name"] = PurePosixPath(chunk.source_relative_path).name
    return KnowledgeSearchToolItem.model_validate(data)
