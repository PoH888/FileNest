# FastAPI → Service → Repository → Session → SQLite

"""工作区业务服务层。

负责组织完整的工作区业务流程和事务，
不处理 HTTP 路由、状态码或响应格式。
"""

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .filesystem_adapter import FileSystemAdapter
from .models import FileEntry, Workspace
from .path_policy import PathPolicyError
from .repositories import (
    add_file_entry,
    add_workspace,
    delete_file_entry,
    find_file_entries,
    find_workspaces,
    get_workspace_by_id,
)
from .workspace_scanner import ScannedFile, scan_workspace_files


class WorkspacePathConflictError(Exception):
    """工作区根路径已经存在。"""


class WorkspaceNotFoundError(Exception):
    """文件索引同步所需的工作区不存在。"""


class DuplicateScannedPathError(Exception):
    """一次扫描结果中出现重复的工作区相对路径。"""


class WorkspaceScanUnavailableError(Exception):
    """工作区根目录当前无法安全扫描。"""


@dataclass(frozen=True, slots=True)
class FileIndexSyncResult:
    """一次文件索引同步产生的变化统计。"""

    created: int
    updated: int
    deleted: int
    unchanged: int


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
