# FastAPI → Service → Repository → Session → SQLite

"""工作区业务服务层。

负责组织完整的工作区业务流程和事务，
不处理 HTTP 路由、状态码或响应格式。
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .filesystem_adapter import FileSystemAdapter
from .models import FileEntry, Workspace
from .operation_preview import (
    OperationPreviewItem,
    OperationPreviewRequest,
    OperationPreviewResponse,
    rank_preview_candidates,
)
from .path_policy import PathPolicyError
from .repositories import (
    add_file_entry,
    add_workspace,
    count_file_entries,
    delete_file_entry,
    FileEntrySortField,
    find_file_entries,
    find_workspaces,
    get_file_entry_by_id,
    get_workspace_by_id,
)
from .workspace_scanner import ScannedFile, scan_workspace_files


class WorkspacePathConflictError(Exception):
    """工作区根路径已经存在。"""


class WorkspaceNotFoundError(Exception):
    """文件索引操作所需的工作区不存在。"""


class FileEntryNotFoundError(Exception):
    """指定工作区内不存在所需的文件索引。"""


class DuplicateScannedPathError(Exception):
    """一次扫描结果中出现重复的工作区相对路径。"""


class WorkspaceScanUnavailableError(Exception):
    """工作区根目录当前无法安全扫描。"""


class OperationPreviewPathUnavailableError(Exception):
    """整理预览所需的源文件或目标目录当前不可用。"""


@dataclass(frozen=True, slots=True)
class FileIndexSyncResult:
    """一次文件索引同步产生的变化统计。"""

    created: int
    updated: int
    deleted: int
    unchanged: int


@dataclass(frozen=True, slots=True)
class FileSearchResult:
    """一次文件搜索的当前页结果和分页元数据。"""

    items: list[FileEntry]
    total: int
    page: int
    page_size: int


_FILE_SORT_FIELDS: dict[str, FileEntrySortField] = {
    "relative_path": "relative_path",
    "name": "name",
    "size_bytes": "size_bytes",
    "modified_at": "mtime_ns",
}


def create_workspace(
    session: Session,
    name: str,
    root_path: str,
) -> Workspace:
    """创建并保存工作区；根路径重复时抛出业务冲突错误。"""

    
    workspace = Workspace(
        name=name,
        root_path=root_path,
    )

    add_workspace(session, workspace)

    try:
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise WorkspacePathConflictError from error

    return workspace


def list_workspaces(
    session: Session,
    name: str | None = None,
) -> list[Workspace]:
    """查询工作区列表，可按名称筛选。"""

    return find_workspaces(session, name)


def get_workspace(
    session: Session,
    workspace_id: int,
) -> Workspace | None:
    """按 ID 查询工作区，找不到时返回 None。"""

    return get_workspace_by_id(session, workspace_id)


def get_file_detail(
    session: Session,
    workspace_id: int,
    file_id: int,
) -> FileEntry:
    """读取指定工作区内一个文件索引的详情。"""

    if get_workspace_by_id(session, workspace_id) is None:
        raise WorkspaceNotFoundError(workspace_id)

    file_entry = get_file_entry_by_id(session, workspace_id, file_id)
    if file_entry is None:
        raise FileEntryNotFoundError(file_id)

    return file_entry


def get_authorized_file_metadata(
    session: Session,
    workspace_id: int,
    file_id: int,
) -> FileEntry:
    """读取文件索引元数据前验证工作区归属和相对路径授权。"""

    workspace = get_workspace_by_id(session, workspace_id)
    if workspace is None:
        raise WorkspaceNotFoundError(workspace_id)

    file_entry = get_file_entry_by_id(session, workspace_id, file_id)
    if file_entry is None:
        raise FileEntryNotFoundError(file_id)

    # 索引属于数据库数据，也可能因历史版本或人工修改而不可信。
    # 返回给 Agent 前重新经过 Path Policy，防止污染路径越过工作区。
    adapter = FileSystemAdapter(Path(workspace.root_path))
    adapter.authorized_path(Path(file_entry.relative_path))
    return file_entry


def generate_operation_preview(
    session: Session,
    request: OperationPreviewRequest,
) -> OperationPreviewResponse:
    """根据当前索引和真实磁盘状态生成只读候选预览。"""

    workspace = get_workspace_by_id(session, request.workspace_id)
    if workspace is None:
        raise WorkspaceNotFoundError(request.workspace_id)

    adapter = FileSystemAdapter(Path(workspace.root_path))
    for directory in request.target_directories:
        try:
            is_directory = adapter.is_directory(Path(directory))
        except OSError as error:
            raise OperationPreviewPathUnavailableError(directory) from error
        if not is_directory:
            raise OperationPreviewPathUnavailableError(directory)

    items: list[OperationPreviewItem] = []
    for file_id in request.source_file_ids:
        file_entry = get_file_entry_by_id(
            session,
            request.workspace_id,
            file_id,
        )
        if file_entry is None:
            raise FileEntryNotFoundError(file_id)

        try:
            metadata = adapter.get_file_metadata(Path(file_entry.relative_path))
        except OSError as error:
            raise OperationPreviewPathUnavailableError(
                file_entry.relative_path
            ) from error
        if metadata is None:
            raise OperationPreviewPathUnavailableError(file_entry.relative_path)

        items.append(
            OperationPreviewItem(
                source_file_id=file_entry.id,
                source_relative_path=file_entry.relative_path,
                candidates=rank_preview_candidates(
                    file_entry.name,
                    request.target_directories,
                ),
            )
        )

    return OperationPreviewResponse(
        workspace_id=request.workspace_id,
        items=items,
    )


def search_files(
    session: Session,
    workspace_id: int,
    *,
    keyword: str | None = None,
    extension: str | None = None,
    modified_from: datetime | None = None,
    modified_to: datetime | None = None,
    sort_by: str = "relative_path",
    sort_order: str = "asc",
    page: int = 1,
    page_size: int = 50,
) -> FileSearchResult:
    """在一个已授权工作区的持久化索引中搜索文件。"""

    if get_workspace_by_id(session, workspace_id) is None:
        raise WorkspaceNotFoundError(workspace_id)
    if page < 1:
        raise ValueError("page must be at least 1")
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")
    if sort_by not in _FILE_SORT_FIELDS:
        raise ValueError(f"unsupported file sort field: {sort_by}")

    normalized_keyword = _normalize_keyword(keyword)
    normalized_extension = _normalize_extension(extension)
    modified_from_ns = _datetime_to_epoch_ns(modified_from)
    modified_to_ns = _datetime_to_epoch_ns(modified_to)
    filter_options = {
        "keyword": normalized_keyword,
        "extension": normalized_extension,
        "modified_from_ns": modified_from_ns,
        "modified_to_ns": modified_to_ns,
    }

    total = count_file_entries(
        session,
        workspace_id,
        **filter_options,
    )
    items = find_file_entries(
        session,
        workspace_id,
        sort_by=_FILE_SORT_FIELDS[sort_by],
        sort_order=sort_order,
        offset=(page - 1) * page_size,
        limit=page_size,
        **filter_options,
    )

    return FileSearchResult(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


def _normalize_keyword(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip()
    if not normalized:
        raise ValueError("keyword must not be blank")
    return normalized


def _normalize_extension(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip().casefold()
    if not normalized:
        raise ValueError("extension must not be blank")
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized


def _datetime_to_epoch_ns(value: datetime | None) -> int | None:
    """不经过浮点时间戳，将带时区时间精确转换为纳秒。"""

    if value is None:
        return None
    if value.utcoffset() is None:
        raise ValueError("modified time must include timezone information")

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = value.astimezone(timezone.utc) - epoch
    return (
        (delta.days * 86_400 + delta.seconds) * 1_000_000_000
        + delta.microseconds * 1_000
    )


def sync_file_index(
    session: Session,
    workspace_id: int,
    scanned_files: list[ScannedFile],
) -> FileIndexSyncResult:
    """将一次完整扫描结果幂等同步到指定工作区的文件索引。"""

    scanned_by_path: dict[str, ScannedFile] = {}
    for scanned_file in scanned_files:
        if scanned_file.relative_path in scanned_by_path:
            raise DuplicateScannedPathError(scanned_file.relative_path)
        scanned_by_path[scanned_file.relative_path] = scanned_file

    try:
        if get_workspace_by_id(session, workspace_id) is None:
            raise WorkspaceNotFoundError(workspace_id)

        existing_by_path = {
            file_entry.relative_path: file_entry
            for file_entry in find_file_entries(session, workspace_id)
        }
        created = 0
        updated = 0
        unchanged = 0

        for relative_path, scanned_file in scanned_by_path.items():
            file_entry = existing_by_path.get(relative_path)

            if file_entry is None:
                add_file_entry(
                    session,
                    FileEntry(
                        workspace_id=workspace_id,
                        relative_path=relative_path,
                        name=scanned_file.name,
                        extension=scanned_file.extension,
                        size_bytes=scanned_file.size_bytes,
                        mtime_ns=scanned_file.mtime_ns,
                    ),
                )
                created += 1
                continue

            current_metadata = (
                file_entry.name,
                file_entry.extension,
                file_entry.size_bytes,
                file_entry.mtime_ns,
            )
            scanned_metadata = (
                scanned_file.name,
                scanned_file.extension,
                scanned_file.size_bytes,
                scanned_file.mtime_ns,
            )

            if current_metadata == scanned_metadata:
                unchanged += 1
                continue

            file_entry.name = scanned_file.name
            file_entry.extension = scanned_file.extension
            file_entry.size_bytes = scanned_file.size_bytes
            file_entry.mtime_ns = scanned_file.mtime_ns
            updated += 1

        deleted = 0
        for relative_path, file_entry in existing_by_path.items():
            if relative_path not in scanned_by_path:
                delete_file_entry(session, file_entry)
                deleted += 1

        session.commit()
    except Exception:
        session.rollback()
        raise

    return FileIndexSyncResult(
        created=created,
        updated=updated,
        deleted=deleted,
        unchanged=unchanged,
    )


def scan_workspace(
    session: Session,
    workspace_id: int,
) -> FileIndexSyncResult:
    """安全扫描一个已授权工作区，并同步其文件索引。"""

    workspace = get_workspace_by_id(session, workspace_id)
    if workspace is None:
        session.rollback()
        raise WorkspaceNotFoundError(workspace_id)

    workspace_root = Path(workspace.root_path)
    adapter = FileSystemAdapter(workspace_root)

    try:
        _require_scannable_workspace_root(adapter)
        scanned_files = scan_workspace_files(
            workspace_root,
            ignore_patterns=[],
        )
        # 扫描期间根目录失效时，空结果不能覆盖现有索引。
        _require_scannable_workspace_root(adapter)
    except WorkspaceScanUnavailableError:
        session.rollback()
        raise

    return sync_file_index(session, workspace_id, scanned_files)


def _require_scannable_workspace_root(
    adapter: FileSystemAdapter,
) -> None:
    """确认工作区根目录存在、是目录并通过 Path Policy。"""

    try:
        is_directory = adapter.is_directory(Path("."))
    except (OSError, PathPolicyError) as error:
        raise WorkspaceScanUnavailableError from error

    if not is_directory:
        raise WorkspaceScanUnavailableError
