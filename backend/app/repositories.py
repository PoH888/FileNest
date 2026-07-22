"""FileNest 数据访问层。

集中封装原本写在 API 路由中的数据库查询与持久化操作，
使路由只负责接收请求、调用业务能力和生成 HTTP 响应。
"""

from typing import Literal

from sqlalchemy import func, or_, select # select()：构造数据库查询。
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from .models import AgentRun, AgentToolCall, FileEntry, Workspace

FileEntrySortField = Literal[
    "relative_path",
    "name",
    "size_bytes",
    "mtime_ns",
]
SortOrder = Literal["asc", "desc"]

_FILE_ENTRY_SORT_COLUMNS = {
    "relative_path": FileEntry.relative_path,
    "name": FileEntry.name,
    "size_bytes": FileEntry.size_bytes,
    "mtime_ns": FileEntry.mtime_ns,
}

def get_workspace_by_id(
        session: Session, # 从外部传入
        workspace_id: int # 要查询的主键
) -> Workspace | None:
    """按 ID 查询一个工作区，找不到时返回 None。"""

    return session.get(Workspace, workspace_id) # 按主键查询


def find_workspaces(
    session: Session,
    name: str | None = None,
) -> list[Workspace]:
    """查询工作区列表；传入名称时只返回同名工作区。"""

    statement = select(Workspace)

    if name is not None:
        statement = statement.where(Workspace.name == name)

    return list(session.scalars(statement).all())

def add_workspace(
    session: Session,
    workspace: Workspace,
) -> None:
    """将工作区加入当前 Session，等待后续提交。"""

    session.add(workspace) # 把对象登记为“等待保存”


def get_file_entry_by_path(
    session: Session,
    workspace_id: int,
    relative_path: str,
) -> FileEntry | None:
    """按工作区和相对路径查询一个文件索引。"""

    statement = select(FileEntry).where(
        FileEntry.workspace_id == workspace_id,
        FileEntry.relative_path == relative_path,
    )
    return session.scalar(statement)


def get_file_entry_by_id(
    session: Session,
    workspace_id: int,
    file_id: int,
) -> FileEntry | None:
    """按工作区和文件 ID 查询一个文件索引。"""

    statement = select(FileEntry).where(
        FileEntry.workspace_id == workspace_id,
        FileEntry.id == file_id,
    )
    return session.scalar(statement)


def find_file_entries(
    session: Session,
    workspace_id: int,
    *,
    sort_by: FileEntrySortField = "relative_path",
    sort_order: SortOrder = "asc",
    offset: int = 0,
    limit: int | None = None,
    keyword: str | None = None,
    extension: str | None = None,
    modified_from_ns: int | None = None,
    modified_to_ns: int | None = None,
) -> list[FileEntry]:
    """稳定排序并分页查询指定工作区的文件索引。"""

    if sort_by not in _FILE_ENTRY_SORT_COLUMNS:
        raise ValueError(f"unsupported file sort field: {sort_by}")
    if sort_order not in {"asc", "desc"}:
        raise ValueError(f"unsupported sort order: {sort_order}")
    if offset < 0:
        raise ValueError("offset must not be negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")

    filters = _file_entry_filters(
        workspace_id,
        keyword=keyword,
        extension=extension,
        modified_from_ns=modified_from_ns,
        modified_to_ns=modified_to_ns,
    )

    sort_column = _FILE_ENTRY_SORT_COLUMNS[sort_by]
    primary_order = (
        sort_column.desc()
        if sort_order == "desc"
        else sort_column.asc()
    )

    statement = (
        select(FileEntry)
        .where(*filters)
        .order_by(primary_order)
    )

    # 非唯一字段出现相同值时固定使用相对路径，避免相邻页面重复或漏项。
    if sort_by != "relative_path":
        statement = statement.order_by(FileEntry.relative_path.asc())

    if offset:
        statement = statement.offset(offset)
    if limit is not None:
        statement = statement.limit(limit)

    return list(session.scalars(statement).all())


def count_file_entries(
    session: Session,
    workspace_id: int,
    *,
    keyword: str | None = None,
    extension: str | None = None,
    modified_from_ns: int | None = None,
    modified_to_ns: int | None = None,
) -> int:
    """返回指定工作区符合过滤条件的文件索引总数。"""

    filters = _file_entry_filters(
        workspace_id,
        keyword=keyword,
        extension=extension,
        modified_from_ns=modified_from_ns,
        modified_to_ns=modified_to_ns,
    )

    statement = select(func.count(FileEntry.id)).where(*filters)
    return session.scalar(statement) or 0


def _file_entry_filters(
    workspace_id: int,
    *,
    keyword: str | None,
    extension: str | None,
    modified_from_ns: int | None,
    modified_to_ns: int | None,
) -> list[ColumnElement[bool]]:
    """为文件列表和总数查询生成完全一致的 SQL 过滤条件。"""

    if (
        modified_from_ns is not None
        and modified_to_ns is not None
        and modified_from_ns > modified_to_ns
    ):
        raise ValueError("modified_from_ns must not exceed modified_to_ns")

    filters: list[ColumnElement[bool]] = [
        FileEntry.workspace_id == workspace_id,
    ]

    if keyword is not None:
        filters.append(
            or_(
                FileEntry.name.icontains(keyword, autoescape=True),
                FileEntry.relative_path.icontains(keyword, autoescape=True),
            )
        )
    if extension is not None:
        filters.append(FileEntry.extension == extension)
    if modified_from_ns is not None:
        filters.append(FileEntry.mtime_ns >= modified_from_ns)
    if modified_to_ns is not None:
        filters.append(FileEntry.mtime_ns <= modified_to_ns)

    return filters


def add_file_entry(
    session: Session,
    file_entry: FileEntry,
) -> None:
    """将文件索引加入当前 Session，事务由调用方提交。"""

    session.add(file_entry)


def delete_file_entry(
    session: Session,
    file_entry: FileEntry,
) -> None:
    """将文件索引标记为待删除，事务由调用方提交。"""

    session.delete(file_entry)


def add_agent_run(
    session: Session,
    agent_run: AgentRun,
) -> None:
    """加入一条 Agent 运行记录，提交时机由记录服务决定。"""

    session.add(agent_run)


def get_agent_run_by_id(
    session: Session,
    agent_run_id: int,
) -> AgentRun | None:
    """按主键读取 Agent 运行记录。"""

    return session.get(AgentRun, agent_run_id)


def add_agent_tool_call(
    session: Session,
    tool_call: AgentToolCall,
) -> None:
    """加入一条工具调用记录，提交时机由记录服务决定。"""

    session.add(tool_call)


def get_agent_tool_call_by_id(
    session: Session,
    tool_call_id: int,
) -> AgentToolCall | None:
    """按主键读取工具调用记录。"""

    return session.get(AgentToolCall, tool_call_id)
