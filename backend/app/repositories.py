"""FileNest 数据访问层。

集中封装原本写在 API 路由中的数据库查询与持久化操作，
使路由只负责接收请求、调用业务能力和生成 HTTP 响应。
"""

from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal

from sqlalchemy import func, or_, select, update # select()：构造数据库查询。
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from .models import (
    AgentRun,
    AgentToolCall,
    ApprovalAuditEvent,
    ApprovalRequest,
    FileEntry,
    OperationExecution,
    OperationExecutionItem,
    Workspace,
)

FileEntrySortField = Literal[
    "relative_path",
    "name",
    "size_bytes",
    "mtime_ns",
]
SortOrder = Literal["asc", "desc"]
ApprovalStatus = Literal["WAITING_APPROVAL", "APPROVED", "REJECTED"]
ApprovalAction = Literal["approve", "edit", "reject"]
OperationExecutionStatus = Literal[
    "EXECUTING",
    "PARTIALLY_COMPLETED",
    "COMPLETED",
    "UNDOING",
    "UNDONE",
    "FAILED",
]
OperationExecutionItemStatus = Literal[
    "PENDING",
    "COMPLETED",
    "UNDOING",
    "UNDONE",
    "FAILED",
]

_FILE_ENTRY_SORT_COLUMNS = {
    "relative_path": FileEntry.relative_path,
    "name": FileEntry.name,
    "size_bytes": FileEntry.size_bytes,
    "mtime_ns": FileEntry.mtime_ns,
}

_OPERATION_EXECUTION_TRANSITIONS: dict[
    OperationExecutionStatus,
    frozenset[OperationExecutionStatus],
] = {
    "EXECUTING": frozenset(
        {"PARTIALLY_COMPLETED", "COMPLETED", "FAILED"}
    ),
    "PARTIALLY_COMPLETED": frozenset({"EXECUTING", "UNDOING"}),
    "COMPLETED": frozenset({"UNDOING"}),
    "UNDOING": frozenset({"UNDONE"}),
    "UNDONE": frozenset(),
    "FAILED": frozenset({"EXECUTING"}),
}

_OPERATION_EXECUTION_ITEM_TRANSITIONS: dict[
    OperationExecutionItemStatus,
    frozenset[OperationExecutionItemStatus],
] = {
    "PENDING": frozenset({"COMPLETED", "FAILED"}),
    "COMPLETED": frozenset({"UNDOING"}),
    "UNDOING": frozenset({"UNDONE"}),
    "UNDONE": frozenset(),
    "FAILED": frozenset({"PENDING"}),
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


def get_approval_request_by_workflow_id(
    session: Session,
    workflow_id: str,
) -> ApprovalRequest | None:
    """按工作流标识读取当前审批业务状态。"""

    statement = select(ApprovalRequest).where(
        ApprovalRequest.workflow_id == workflow_id,
    )
    return session.scalar(statement)


def find_waiting_approval_requests(
    session: Session,
) -> list[ApprovalRequest]:
    """按稳定顺序读取所有等待人工决定的审批任务。"""

    statement = (
        select(ApprovalRequest)
        .where(ApprovalRequest.status == "WAITING_APPROVAL")
        .order_by(ApprovalRequest.id.asc())
    )
    return list(session.scalars(statement).all())


def compare_and_set_approval_request(
    session: Session,
    workflow_id: str,
    expected_plan_id: str,
    *,
    next_status: ApprovalStatus,
    next_plan_id: str,
) -> bool:
    """仅在审批仍等待且计划未变化时原子更新。"""

    statement = (
        update(ApprovalRequest)
        .where(
            ApprovalRequest.workflow_id == workflow_id,
            ApprovalRequest.status == "WAITING_APPROVAL",
            ApprovalRequest.plan_id == expected_plan_id,
        )
        .values(
            status=next_status,
            plan_id=next_plan_id,
        )
    )
    result = session.execute(
        statement,
        execution_options={"synchronize_session": False},
    )
    return result.rowcount == 1


def add_approval_audit_event(
    session: Session,
    audit_event: ApprovalAuditEvent,
) -> None:
    """追加审批历史，提交时机由同一审批事务决定。"""

    session.add(audit_event)


def find_approval_audit_events(
    session: Session,
    approval_request_id: int,
) -> list[ApprovalAuditEvent]:
    """按写入顺序读取一条审批任务的不可变历史。"""

    statement = (
        select(ApprovalAuditEvent)
        .where(
            ApprovalAuditEvent.approval_request_id == approval_request_id
        )
        .order_by(ApprovalAuditEvent.id.asc())
    )
    return list(session.scalars(statement).all())


def add_operation_execution(
    session: Session,
    execution: OperationExecution,
) -> None:
    """加入执行主记录，提交时机由执行服务决定。"""

    session.add(execution)


def add_operation_execution_item(
    session: Session,
    execution_item: OperationExecutionItem,
) -> None:
    """加入一个文件操作的 before、after 与 undo 证据。"""

    session.add(execution_item)


def get_operation_execution_by_id(
    session: Session,
    execution_id: int,
) -> OperationExecution | None:
    """按主键读取一条执行主记录。"""

    return session.get(OperationExecution, execution_id)


def get_operation_execution_by_workflow_id(
    session: Session,
    workflow_id: str,
) -> OperationExecution | None:
    """按工作流标识读取唯一执行记录。"""

    statement = select(OperationExecution).where(
        OperationExecution.workflow_id == workflow_id,
    )
    return session.scalar(statement)


def get_operation_execution_by_plan_id(
    session: Session,
    plan_id: str,
) -> OperationExecution | None:
    """按确定计划标识读取唯一执行记录。"""

    statement = select(OperationExecution).where(
        OperationExecution.plan_id == plan_id,
    )
    return session.scalar(statement)


def find_operation_execution_items(
    session: Session,
    execution_id: int,
) -> list[OperationExecutionItem]:
    """按计划顺序读取一条执行记录的所有文件操作证据。"""

    statement = (
        select(OperationExecutionItem)
        .where(OperationExecutionItem.execution_id == execution_id)
        .order_by(OperationExecutionItem.sequence_no.asc())
    )
    return list(session.scalars(statement).all())


def compare_and_set_operation_execution_status(
    session: Session,
    execution_id: int,
    expected_status: OperationExecutionStatus,
    *,
    next_status: OperationExecutionStatus,
    completed_at: datetime | None = None,
    undone_at: datetime | None = None,
) -> bool:
    """仅按合法状态图原子转换，并在真实重试时增加 attempt。"""

    allowed_next_statuses = _OPERATION_EXECUTION_TRANSITIONS.get(
        expected_status
    )
    if (
        allowed_next_statuses is None
        or next_status not in allowed_next_statuses
    ):
        raise ValueError(
            "illegal operation execution transition: "
            f"{expected_status} -> {next_status}"
        )

    values: dict[str, object] = {"status": next_status}
    if (
        expected_status in {"PARTIALLY_COMPLETED", "FAILED"}
        and next_status == "EXECUTING"
    ):
        values["attempt"] = OperationExecution.attempt + 1
        values["completed_at"] = None
    if completed_at is not None:
        values["completed_at"] = completed_at
    if undone_at is not None:
        values["undone_at"] = undone_at

    statement = (
        update(OperationExecution)
        .where(
            OperationExecution.id == execution_id,
            OperationExecution.status == expected_status,
        )
        .values(**values)
    )
    result = session.execute(
        statement,
        execution_options={"synchronize_session": False},
    )
    return result.rowcount == 1


def compare_and_set_operation_execution_item_status(
    session: Session,
    execution_item_id: int,
    expected_status: OperationExecutionItemStatus,
    *,
    next_status: OperationExecutionItemStatus,
    after_size_bytes: int | None = None,
    after_mtime_ns: int | None = None,
    after_sha256: str | None = None,
    completed_at: datetime | None = None,
    error_code: str | None = None,
    failed_at: datetime | None = None,
    undone_at: datetime | None = None,
) -> bool:
    """按合法状态图原子转换明细，并持久化成功或失败证据。"""

    allowed_next_statuses = _OPERATION_EXECUTION_ITEM_TRANSITIONS.get(
        expected_status
    )
    if (
        allowed_next_statuses is None
        or next_status not in allowed_next_statuses
    ):
        raise ValueError(
            "illegal operation execution item transition: "
            f"{expected_status} -> {next_status}"
        )
    if next_status == "FAILED":
        if (
            error_code is None
            or not error_code.strip()
            or error_code != error_code.strip()
            or len(error_code) > 100
            or failed_at is None
            or failed_at.tzinfo is None
            or failed_at.utcoffset() is None
        ):
            raise ValueError(
                "failed execution item requires error_code and failed_at"
            )
    elif error_code is not None or failed_at is not None:
        raise ValueError(
            "failure evidence is only valid for FAILED execution items"
        )

    values: dict[str, object] = {"status": next_status}
    if expected_status == "FAILED" and next_status == "PENDING":
        values["error_code"] = None
        values["failed_at"] = None
    if after_size_bytes is not None:
        values["after_size_bytes"] = after_size_bytes
    if after_mtime_ns is not None:
        values["after_mtime_ns"] = after_mtime_ns
    if after_sha256 is not None:
        values["after_sha256"] = after_sha256
    if completed_at is not None:
        values["completed_at"] = completed_at
    if error_code is not None:
        values["error_code"] = error_code
    if failed_at is not None:
        values["failed_at"] = failed_at
    if undone_at is not None:
        values["undone_at"] = undone_at

    statement = (
        update(OperationExecutionItem)
        .where(
            OperationExecutionItem.id == execution_item_id,
            OperationExecutionItem.status == expected_status,
        )
        .values(**values)
    )
    result = session.execute(
        statement,
        execution_options={"synchronize_session": False},
    )
    return result.rowcount == 1


def compare_and_set_file_entry_location(
    session: Session,
    workspace_id: int,
    file_entry_id: int,
    expected_relative_path: str,
    *,
    next_relative_path: str,
    size_bytes: int,
    mtime_ns: int,
) -> bool:
    """仅在索引仍指向预期旧路径时更新文件位置和元数据。"""

    path = PurePosixPath(next_relative_path)
    if (
        "\\" in next_relative_path
        or path.is_absolute()
        or path.name in {"", ".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != next_relative_path
    ):
        raise ValueError("next_relative_path must be normalized and relative")
    if size_bytes < 0 or mtime_ns < 0:
        raise ValueError("file metadata must not be negative")

    statement = (
        update(FileEntry)
        .where(
            FileEntry.workspace_id == workspace_id,
            FileEntry.id == file_entry_id,
            FileEntry.relative_path == expected_relative_path,
        )
        .values(
            relative_path=next_relative_path,
            name=path.name,
            extension=path.suffix,
            size_bytes=size_bytes,
            mtime_ns=mtime_ns,
        )
    )
    result = session.execute(
        statement,
        execution_options={"synchronize_session": False},
    )
    return result.rowcount == 1
