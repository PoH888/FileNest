"""FileNest Agent 可调用的只读业务工具。"""

from datetime import datetime, timezone
import re
from pathlib import Path, PureWindowsPath
from pathlib import PurePosixPath
from stat import S_ISDIR, S_ISREG
from typing import Literal, cast

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
from .filesystem_adapter import FileSystemAdapter
from .models import ChunkRecord, DocumentRecord, FileEntry
from .path_policy import PathPolicyError
from .services import (
    FileEntryNotFoundError,
    WorkspacePolicyError,
    WorkspaceNotFoundError,
    get_authorized_file_metadata as get_file_metadata_service,
    get_workspace as get_workspace_service,
    list_workspaces as list_workspaces_service,
    require_workspace_read_policy,
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


class ListDirectoryArguments(BaseModel):
    """列目录工具允许模型提供的结构化参数。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: int = Field(ge=1)
    relative_directory: str = Field(default=".", min_length=1, max_length=500)
    page_size: int = Field(default=20, ge=1, le=20)
    cursor: str | None = Field(default=None, min_length=1, max_length=500)

    @field_validator("relative_directory")
    @classmethod
    def validate_relative_directory(cls, value: str) -> str:
        if value != value.strip() or "\\" in value:
            raise ValueError(
                "relative_directory must be a normalized POSIX relative path"
            )
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or PureWindowsPath(value).drive
            or any(part in {"", ".."} for part in path.parts)
            or str(path) != value
        ):
            raise ValueError(
                "relative_directory must be a normalized POSIX relative path"
            )
        return value

    @field_validator("cursor")
    @classmethod
    def validate_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value != value.strip() or "\\" in value:
            raise ValueError("cursor must be a normalized relative path")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or PureWindowsPath(value).drive
            or any(part in {"", ".", ".."} for part in path.parts)
            or str(path) != value
        ):
            raise ValueError("cursor must be a normalized relative path")
        return value


DirectoryEntryType = Literal["file", "directory", "other", "ignored"]


class DirectoryToolItem(BaseModel):
    """目录工具返回的安全元数据，不包含文件内容。"""

    model_config = ConfigDict(extra="forbid")

    relative_path: str = Field(min_length=1)
    entry_type: DirectoryEntryType
    size_bytes: int | None = Field(default=None, ge=0)
    modified_at: AwareDatetime | None = None
    ignored_reason: str | None = None

    @model_validator(mode="after")
    def validate_ignored_metadata(self) -> "DirectoryToolItem":
        if self.entry_type == "ignored":
            if self.ignored_reason is None:
                raise ValueError("ignored entries must contain a reason")
            if self.size_bytes is not None or self.modified_at is not None:
                raise ValueError("ignored entries must not contain metadata")
        elif self.ignored_reason is not None:
            raise ValueError("visible entries must not contain an ignore reason")
        return self


class ListDirectoryData(BaseModel):
    """目录工具的有上限、可恢复分页结果。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: int = Field(ge=1)
    relative_directory: str = Field(min_length=1)
    items: list[DirectoryToolItem]
    next_cursor: str | None = None
    has_more: bool


class FindSimilarFoldersArguments(BaseModel):
    """相似目录工具允许模型提供的结构化参数。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: int = Field(ge=1)
    source_file_id: int = Field(ge=1)
    limit: int = Field(default=5, ge=1, le=10)


class SimilarFolderCandidate(BaseModel):
    """一个可解释的目录候选及其数据库文件引用。"""

    model_config = ConfigDict(extra="forbid")

    relative_directory: str = Field(min_length=1)
    score: int = Field(ge=1, le=100)
    reasons: tuple[str, ...] = Field(min_length=1, max_length=5)
    file_ids: tuple[int, ...] = Field(min_length=1, max_length=20)


class FindSimilarFoldersData(BaseModel):
    """相似目录工具的最小候选结果。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: int = Field(ge=1)
    source_file_id: int = Field(ge=1)
    items: list[SimilarFolderCandidate]
    empty_reason: str | None = None

    @model_validator(mode="after")
    def validate_empty_reason(self) -> "FindSimilarFoldersData":
        if bool(self.items) == (self.empty_reason is not None):
            raise ValueError(
                "empty_reason must describe an empty result only"
            )
        return self


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
        except WorkspacePolicyError as error:
            return ToolResult.failure(
                code=error.code.value,
                message=str(error),
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


def build_list_directory_tool(
    session: Session,
    *,
    user_denylist: tuple[Path, ...] = (),
) -> Tool:
    """构建通过 FileSystemAdapter 和 Path Policy 的目录只读工具。"""

    def handle(arguments: BaseModel) -> ToolResult:
        options = cast(ListDirectoryArguments, arguments)
        workspace = get_workspace_service(session, options.workspace_id)
        if workspace is None:
            return ToolResult.failure(
                code="workspace_not_found",
                message="工作区不存在",
                details={"workspace_id": options.workspace_id},
            )

        try:
            policy = require_workspace_read_policy(
                session,
                options.workspace_id,
            )
        except WorkspacePolicyError as error:
            return ToolResult.failure(
                code=error.code.value,
                message=str(error),
                details={"workspace_id": options.workspace_id},
            )

        adapter = FileSystemAdapter(
            Path(workspace.root_path),
            user_denylist=user_denylist,
            workspace_policy=policy,
        )
        ignored: dict[str, str] = {}

        def record_ignored(path: Path, error: PathPolicyError) -> None:
            ignored[path.name] = error.code.value

        try:
            visible_names = adapter.list_directory(
                Path(options.relative_directory),
                on_ignored=record_ignored,
            )
        except PathPolicyError as error:
            return ToolResult.failure(
                code=error.code.value,
                message=error.message,
                details={
                    "workspace_id": options.workspace_id,
                    "relative_directory": options.relative_directory,
                },
            )
        except FileNotFoundError:
            return ToolResult.failure(
                code="directory_not_found",
                message="目录不存在",
                details={
                    "workspace_id": options.workspace_id,
                    "relative_directory": options.relative_directory,
                },
            )
        except NotADirectoryError:
            return ToolResult.failure(
                code="not_a_directory",
                message="请求路径不是目录",
                details={
                    "workspace_id": options.workspace_id,
                    "relative_directory": options.relative_directory,
                },
            )
        except OSError:
            return ToolResult.failure(
                code="directory_unavailable",
                message="目录当前不可读取",
                details={
                    "workspace_id": options.workspace_id,
                    "relative_directory": options.relative_directory,
                },
            )

        names = sorted(
            set(visible_names).union(ignored),
            key=lambda name: (name.casefold(), name),
        )
        relative_directory = PurePosixPath(options.relative_directory)
        relative_paths = {
            name: (
                PurePosixPath(name)
                if relative_directory == PurePosixPath(".")
                else relative_directory / name
            ).as_posix()
            for name in names
        }
        if options.cursor is not None:
            cursor_key = options.cursor.casefold()
            names = [
                name
                for name in names
                if relative_paths[name].casefold() > cursor_key
            ]

        selected_names = names[: options.page_size]
        has_more = len(names) > len(selected_names)
        items: list[DirectoryToolItem] = []
        for name in selected_names:
            relative_path = relative_paths[name]
            ignored_reason = ignored.get(name)
            if ignored_reason is not None:
                items.append(
                    DirectoryToolItem(
                        relative_path=relative_path,
                        entry_type="ignored",
                        ignored_reason=ignored_reason,
                    )
                )
                continue

            try:
                authorized_path = adapter.authorized_path(Path(relative_path))
                metadata = authorized_path.stat()
            except PathPolicyError as error:
                return ToolResult.failure(
                    code=error.code.value,
                    message=error.message,
                    details={
                        "workspace_id": options.workspace_id,
                        "relative_directory": options.relative_directory,
                    },
                )
            except OSError:
                return ToolResult.failure(
                    code="directory_entry_unavailable",
                    message="目录条目当前不可读取",
                    details={
                        "workspace_id": options.workspace_id,
                        "relative_directory": options.relative_directory,
                    },
                )

            entry_type: DirectoryEntryType
            if S_ISDIR(metadata.st_mode):
                entry_type = "directory"
            elif S_ISREG(metadata.st_mode):
                entry_type = "file"
            else:
                entry_type = "other"
            items.append(
                DirectoryToolItem(
                    relative_path=relative_path,
                    entry_type=entry_type,
                    size_bytes=metadata.st_size,
                    modified_at=_modified_at_from_ns(metadata.st_mtime_ns),
                )
            )

        data = ListDirectoryData(
            workspace_id=options.workspace_id,
            relative_directory=options.relative_directory,
            items=items,
            next_cursor=(relative_paths[selected_names[-1]] if has_more else None),
            has_more=has_more,
        )
        return ToolResult.success(data.model_dump(mode="json"))

    return Tool(
        name="list_directory",
        description=(
            "列出指定 FileNest 工作区目录的直接子项；"
            "只返回相对路径、类型、大小、修改时间和忽略原因，最多 20 条。"
        ),
        arguments_model=ListDirectoryArguments,
        handler=handle,
    )


def build_find_similar_folders_tool(session: Session) -> Tool:
    """构建基于文件名、扩展名和现有索引的可解释目录候选工具。"""

    def handle(arguments: BaseModel) -> ToolResult:
        options = cast(FindSimilarFoldersArguments, arguments)
        try:
            source_file = get_file_metadata_service(
                session,
                options.workspace_id,
                options.source_file_id,
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
                message="文件索引不存在或不属于当前工作区",
                details={
                    "workspace_id": options.workspace_id,
                    "source_file_id": options.source_file_id,
                },
            )
        except WorkspacePolicyError as error:
            return ToolResult.failure(
                code=error.code.value,
                message=str(error),
                details={"workspace_id": options.workspace_id},
            )
        except PathPolicyError as error:
            return ToolResult.failure(
                code=error.code.value,
                message=error.message,
                details={"workspace_id": options.workspace_id},
            )

        workspace = get_workspace_service(session, options.workspace_id)
        if workspace is None:
            return ToolResult.failure(
                code="workspace_not_found",
                message="工作区不存在",
                details={"workspace_id": options.workspace_id},
            )
        try:
            policy = require_workspace_read_policy(
                session,
                options.workspace_id,
            )
        except WorkspacePolicyError as error:
            return ToolResult.failure(
                code=error.code.value,
                message=str(error),
                details={"workspace_id": options.workspace_id},
            )
        adapter = FileSystemAdapter(
            Path(workspace.root_path),
            workspace_policy=policy,
        )
        try:
            source_parent = _safe_file_parent(source_file.relative_path)
            if source_parent is None:
                return ToolResult.failure(
                    code="invalid_indexed_path",
                    message="文件索引路径当前不可用",
                    details={"workspace_id": options.workspace_id},
                )
            adapter.authorized_path(Path(source_parent))
        except PathPolicyError as error:
            return ToolResult.failure(
                code=error.code.value,
                message=error.message,
                details={"workspace_id": options.workspace_id},
            )

        statement = (
            select(FileEntry)
            .where(FileEntry.workspace_id == options.workspace_id)
            .order_by(FileEntry.relative_path.asc(), FileEntry.id.asc())
        )
        with session.no_autoflush:
            file_entries = list(session.scalars(statement).all())

        directories: dict[str, list[FileEntry]] = {}
        for file_entry in file_entries:
            parent = _safe_file_parent(file_entry.relative_path)
            if parent is None:
                continue
            directories.setdefault(parent, []).append(file_entry)

        source_tokens = _similar_name_tokens(
            PurePosixPath(source_file.relative_path).stem
        )
        source_extension = source_file.extension.casefold()
        candidates: list[SimilarFolderCandidate] = []
        skipped_by_policy = False
        for relative_directory, entries in directories.items():
            if relative_directory == source_parent:
                continue
            try:
                if not adapter.is_directory(Path(relative_directory)):
                    continue
            except PathPolicyError:
                skipped_by_policy = True
                continue
            except OSError:
                continue

            directory_tokens = _similar_name_tokens(relative_directory)
            matched_tokens = tuple(
                sorted(source_tokens & directory_tokens)
            )
            matching_file_tokens = set().union(
                *(
                    _similar_name_tokens(PurePosixPath(entry.relative_path).stem)
                    for entry in entries
                )
            )
            file_name_matches = tuple(
                sorted(source_tokens & matching_file_tokens)
            )
            same_extension = any(
                entry.extension.casefold() == source_extension
                for entry in entries
            )
            reasons: list[str] = []
            score = 0
            if same_extension:
                score += 50
                reasons.append("extension_match")
            if matched_tokens:
                score += min(30, 15 * len(matched_tokens))
                reasons.append("directory_name_token_overlap")
            if file_name_matches:
                score += min(20, 10 * len(file_name_matches))
                reasons.append("file_name_token_overlap")
            if score == 0:
                continue

            candidates.append(
                SimilarFolderCandidate(
                    relative_directory=relative_directory,
                    score=min(score, 100),
                    reasons=tuple(reasons),
                    file_ids=tuple(entry.id for entry in entries[:20]),
                )
            )

        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.relative_directory.casefold(),
                candidate.relative_directory,
            )
        )
        selected = candidates[: options.limit]
        empty_reason = None
        if not selected:
            if skipped_by_policy:
                empty_reason = "no_authorized_candidates"
            elif len(directories) <= 1:
                empty_reason = "insufficient_directory_samples"
            else:
                empty_reason = "no_explainable_match"

        data = FindSimilarFoldersData(
            workspace_id=options.workspace_id,
            source_file_id=options.source_file_id,
            items=selected,
            empty_reason=empty_reason,
        )
        return ToolResult.success(data.model_dump(mode="json"))

    return Tool(
        name="find_similar_folders",
        description=(
            "根据现有文件索引的目录名、扩展名和文件名 token，"
            "返回可解释的只读目录候选；不会创建或执行操作计划。"
        ),
        arguments_model=FindSimilarFoldersArguments,
        handler=handle,
    )


def build_knowledge_search_tool(session: Session) -> Tool:
    """为当前数据库会话构建有上限的只读知识搜索工具。"""

    def handle(arguments: BaseModel) -> ToolResult:
        options = cast(KnowledgeSearchArguments, arguments)

        with session.no_autoflush:
            workspace = get_workspace_service(session, options.workspace_id)
            if workspace is None:
                return ToolResult.failure(
                    code="workspace_not_found",
                    message="工作区不存在",
                    details={"workspace_id": options.workspace_id},
                )

            try:
                policy = require_workspace_read_policy(
                    session,
                    options.workspace_id,
                )
            except WorkspacePolicyError as error:
                return ToolResult.failure(
                    code=error.code.value,
                    message=str(error),
                    details={"workspace_id": options.workspace_id},
                )

            adapter = FileSystemAdapter(
                Path(workspace.root_path),
                workspace_policy=policy,
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
            rows = []
            for chunk, document in session.execute(statement).all():
                try:
                    adapter.authorized_path(Path(chunk.source_relative_path))
                except PathPolicyError:
                    continue
                rows.append((chunk, document))

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
        except WorkspacePolicyError as error:
            return ToolResult.failure(
                code=error.code.value,
                message=str(error),
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


def _modified_at_from_ns(mtime_ns: int) -> datetime:
    """将授权后读取的纳秒时间转换成稳定的 UTC 时间。"""

    seconds, nanoseconds = divmod(mtime_ns, 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
        microsecond=nanoseconds // 1_000,
    )


def _safe_file_parent(relative_path: str) -> str | None:
    """从索引相对路径提取规范父目录，不替不可信索引放宽路径规则。"""

    path = PurePosixPath(relative_path)
    if (
        not relative_path
        or relative_path != relative_path.strip()
        or "\\" in relative_path
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != relative_path
    ):
        return None
    return path.parent.as_posix()


_SIMILAR_NAME_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]+")


def _similar_name_tokens(value: str) -> frozenset[str]:
    """以稳定 token 集支持可解释的轻量目录匹配。"""

    return frozenset(
        token.casefold()
        for token in _SIMILAR_NAME_TOKEN_PATTERN.findall(value)
    )


def _knowledge_search_tool_item(
    chunk: RetrievedChunk,
) -> KnowledgeSearchToolItem:
    """仅从已验证的 RetrievalContext 投影知识工具结果。"""

    data = retrieval_chunk_to_mapping(chunk)
    data["name"] = PurePosixPath(chunk.source_relative_path).name
    return KnowledgeSearchToolItem.model_validate(data)
