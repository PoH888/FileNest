"""安全文件操作执行前的统一门禁。"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from .filesystem_adapter import FileMetadata, FileSystemAdapter
from .models import OperationExecution, OperationExecutionItem
from .operation_plan import OperationPlan
from .repositories import (
    add_operation_execution,
    add_operation_execution_item,
    compare_and_set_file_entry_location,
    compare_and_set_operation_execution_item_status,
    compare_and_set_operation_execution_status,
    find_operation_execution_items,
    get_file_entry_by_id,
    get_operation_execution_by_plan_id,
    get_operation_execution_by_workflow_id,
    get_workspace_by_id,
)
from .safe_file_mover import SafeFileMover
from .services import require_approved_operation_plan, validate_operation_plan


class SafeExecutionRequest(BaseModel):
    """一次安全执行所需的不可变输入。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_id: UUID
    plan: OperationPlan


class SafeExecutionErrorCode(StrEnum):
    """安全执行或撤销失败时供程序稳定判断的错误码。"""

    BATCH_NOT_SUPPORTED = "safe_execution_batch_not_supported"
    HISTORY_EXISTS = "safe_execution_history_exists"
    HISTORY_NOT_FOUND = "safe_execution_history_not_found"
    INVALID_HISTORY_STATE = "safe_execution_history_state_invalid"
    WORKSPACE_NOT_FOUND = "safe_execution_workspace_not_found"
    FILE_ENTRY_NOT_FOUND = "safe_execution_file_entry_not_found"
    STATE_CHANGED = "safe_execution_state_changed"
    FILE_CHANGED = "safe_execution_file_changed"
    UNDO_TARGET_CONFLICT = "safe_execution_undo_target_conflict"
    HISTORY_WRITE_FAILED = "safe_execution_history_write_failed"


class SafeExecutionError(RuntimeError):
    """执行历史或当前磁盘状态不允许继续操作。"""

    def __init__(
        self,
        code: SafeExecutionErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class SafeExecutionResult:
    """一次执行或撤销完成后的稳定业务结果。"""

    execution_id: int
    workflow_id: UUID
    plan_id: UUID
    status: str
    before_relative_path: str
    after_relative_path: str


def validate_safe_execution_request(
    session: Session,
    request: SafeExecutionRequest,
    *,
    now: datetime | None = None,
) -> None:
    """确认人工审批与当前磁盘状态都仍允许执行。"""

    # 审批必须先于磁盘探测，避免未获批准的计划进入执行准备阶段。
    require_approved_operation_plan(
        session,
        request.workflow_id,
        request.plan,
    )
    validate_operation_plan(session, request.plan, now=now)


def execute_safe_operation_plan(
    session: Session,
    request: SafeExecutionRequest,
    *,
    now: datetime | None = None,
) -> SafeExecutionResult:
    """执行一个已批准的单项计划，并持久化 before/after/undo。"""

    current_time = _aware_current_time(now)
    require_approved_operation_plan(
        session,
        request.workflow_id,
        request.plan,
    )
    if (
        get_operation_execution_by_workflow_id(
            session,
            str(request.workflow_id),
        )
        is not None
        or get_operation_execution_by_plan_id(
            session,
            str(request.plan.plan_id),
        )
        is not None
    ):
        raise SafeExecutionError(
            SafeExecutionErrorCode.HISTORY_EXISTS,
            "当前工作流或计划已经存在执行历史",
        )
    if len(request.plan.operations) != 1:
        raise SafeExecutionError(
            SafeExecutionErrorCode.BATCH_NOT_SUPPORTED,
            "部分失败恢复完成前只允许执行单项计划",
        )

    validate_operation_plan(session, request.plan, now=current_time)
    operation = request.plan.operations[0]
    workspace = get_workspace_by_id(session, request.plan.workspace_id)
    if workspace is None:
        raise SafeExecutionError(
            SafeExecutionErrorCode.WORKSPACE_NOT_FOUND,
            "执行计划对应的工作区不存在",
        )
    file_entry = get_file_entry_by_id(
        session,
        request.plan.workspace_id,
        operation.source_file_id,
    )
    if file_entry is None:
        raise SafeExecutionError(
            SafeExecutionErrorCode.FILE_ENTRY_NOT_FOUND,
            "执行计划对应的文件索引不存在",
        )

    execution = OperationExecution(
        workflow_id=str(request.workflow_id),
        plan_id=str(request.plan.plan_id),
        workspace_id=request.plan.workspace_id,
    )
    add_operation_execution(session, execution)
    expected_hash = operation.source_precondition.content_hash
    try:
        session.flush()
        execution_item = OperationExecutionItem(
            execution_id=execution.id,
            sequence_no=1,
            operation_type="move",
            source_file_id=operation.source_file_id,
            before_location="workspace",
            before_relative_path=operation.source_relative_path,
            before_size_bytes=operation.source_precondition.size_bytes,
            before_mtime_ns=operation.source_precondition.mtime_ns,
            before_sha256=(
                expected_hash.digest if expected_hash is not None else None
            ),
            after_location="workspace",
            after_relative_path=operation.target_relative_path,
            undo_source_relative_path=operation.target_relative_path,
            undo_target_relative_path=operation.source_relative_path,
        )
        add_operation_execution_item(session, execution_item)
        session.flush()
        execution_id = execution.id
        execution_item_id = execution_item.id
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise SafeExecutionError(
            SafeExecutionErrorCode.HISTORY_EXISTS,
            "执行历史已被并发创建或违反唯一约束",
        ) from error
    except SQLAlchemyError as error:
        session.rollback()
        raise SafeExecutionError(
            SafeExecutionErrorCode.HISTORY_WRITE_FAILED,
            "无法保存执行前历史",
        ) from error

    adapter = FileSystemAdapter(Path(workspace.root_path))
    moved_path = SafeFileMover(adapter).move(
        Path(operation.source_relative_path),
        Path(operation.target_relative_path),
    )
    after_metadata = _get_required_metadata(
        adapter,
        moved_path,
        "执行后的目标文件不可用",
    )
    after_hash = _verify_optional_hash(
        adapter,
        moved_path,
        expected_hash.digest if expected_hash is not None else None,
    )

    item_updated = compare_and_set_operation_execution_item_status(
        session,
        execution_item_id,
        "PENDING",
        next_status="COMPLETED",
        after_size_bytes=after_metadata.size_bytes,
        after_mtime_ns=after_metadata.mtime_ns,
        after_sha256=after_hash,
        completed_at=current_time,
    )
    index_updated = compare_and_set_file_entry_location(
        session,
        request.plan.workspace_id,
        operation.source_file_id,
        operation.source_relative_path,
        next_relative_path=operation.target_relative_path,
        size_bytes=after_metadata.size_bytes,
        mtime_ns=after_metadata.mtime_ns,
    )
    execution_updated = compare_and_set_operation_execution_status(
        session,
        execution_id,
        "EXECUTING",
        next_status="COMPLETED",
        completed_at=current_time,
    )
    if not (item_updated and index_updated and execution_updated):
        session.rollback()
        raise SafeExecutionError(
            SafeExecutionErrorCode.STATE_CHANGED,
            "磁盘移动完成后执行历史或文件索引已经变化",
        )
    _commit_completion(
        session,
        "磁盘移动完成，但无法提交 after 与索引状态",
    )

    return SafeExecutionResult(
        execution_id=execution_id,
        workflow_id=request.workflow_id,
        plan_id=request.plan.plan_id,
        status="COMPLETED",
        before_relative_path=operation.source_relative_path,
        after_relative_path=operation.target_relative_path,
    )


def undo_safe_operation_execution(
    session: Session,
    workflow_id: UUID,
    *,
    now: datetime | None = None,
) -> SafeExecutionResult:
    """撤销一条已完成且磁盘状态未变化的单项移动记录。"""

    current_time = _aware_current_time(now)
    execution = get_operation_execution_by_workflow_id(
        session,
        str(workflow_id),
    )
    if execution is None:
        raise SafeExecutionError(
            SafeExecutionErrorCode.HISTORY_NOT_FOUND,
            "找不到需要撤销的执行历史",
        )
    if execution.status != "COMPLETED":
        raise SafeExecutionError(
            SafeExecutionErrorCode.INVALID_HISTORY_STATE,
            "只有已完成且尚未撤销的执行记录可以撤销",
        )

    execution_items = find_operation_execution_items(session, execution.id)
    if len(execution_items) != 1:
        raise SafeExecutionError(
            SafeExecutionErrorCode.INVALID_HISTORY_STATE,
            "当前撤销边界只接受一条完整执行明细",
        )
    execution_item = execution_items[0]
    if (
        execution_item.status != "COMPLETED"
        or execution_item.operation_type != "move"
        or execution_item.before_location != "workspace"
        or execution_item.after_location != "workspace"
        or execution_item.after_size_bytes is None
        or execution_item.after_mtime_ns is None
    ):
        raise SafeExecutionError(
            SafeExecutionErrorCode.INVALID_HISTORY_STATE,
            "执行明细缺少可安全撤销的 after 证据",
        )

    workspace = get_workspace_by_id(session, execution.workspace_id)
    if workspace is None:
        raise SafeExecutionError(
            SafeExecutionErrorCode.WORKSPACE_NOT_FOUND,
            "执行历史对应的工作区不存在",
        )
    adapter = FileSystemAdapter(Path(workspace.root_path))
    undo_source = Path(execution_item.undo_source_relative_path)
    undo_target = Path(execution_item.undo_target_relative_path)
    current_metadata = _get_required_metadata(
        adapter,
        undo_source,
        "撤销源文件不存在或不是普通文件",
    )
    if (
        current_metadata.size_bytes != execution_item.after_size_bytes
        or current_metadata.mtime_ns != execution_item.after_mtime_ns
    ):
        raise SafeExecutionError(
            SafeExecutionErrorCode.FILE_CHANGED,
            "执行后的文件已经变化，拒绝撤销",
        )
    _verify_optional_hash(
        adapter,
        undo_source,
        execution_item.after_sha256,
    )
    if adapter.path_exists(undo_target):
        raise SafeExecutionError(
            SafeExecutionErrorCode.UNDO_TARGET_CONFLICT,
            "原路径已经被占用，拒绝覆盖撤销",
        )

    execution_marked = compare_and_set_operation_execution_status(
        session,
        execution.id,
        "COMPLETED",
        next_status="UNDOING",
    )
    item_marked = compare_and_set_operation_execution_item_status(
        session,
        execution_item.id,
        "COMPLETED",
        next_status="UNDOING",
    )
    if not (execution_marked and item_marked):
        session.rollback()
        raise SafeExecutionError(
            SafeExecutionErrorCode.STATE_CHANGED,
            "撤销开始前执行历史已经变化",
        )
    _commit_completion(session, "无法保存撤销开始状态")

    restored_path = SafeFileMover(adapter).move(undo_source, undo_target)
    restored_metadata = _get_required_metadata(
        adapter,
        restored_path,
        "撤销后的原文件不可用",
    )
    if (
        restored_metadata.size_bytes != execution_item.before_size_bytes
        or restored_metadata.mtime_ns != execution_item.before_mtime_ns
    ):
        raise SafeExecutionError(
            SafeExecutionErrorCode.FILE_CHANGED,
            "撤销结果与 before 元数据不一致",
        )
    _verify_optional_hash(
        adapter,
        restored_path,
        execution_item.before_sha256,
    )

    index_updated = compare_and_set_file_entry_location(
        session,
        execution.workspace_id,
        execution_item.source_file_id,
        execution_item.after_relative_path,
        next_relative_path=execution_item.before_relative_path,
        size_bytes=restored_metadata.size_bytes,
        mtime_ns=restored_metadata.mtime_ns,
    )
    item_updated = compare_and_set_operation_execution_item_status(
        session,
        execution_item.id,
        "UNDOING",
        next_status="UNDONE",
        undone_at=current_time,
    )
    execution_updated = compare_and_set_operation_execution_status(
        session,
        execution.id,
        "UNDOING",
        next_status="UNDONE",
        undone_at=current_time,
    )
    if not (index_updated and item_updated and execution_updated):
        session.rollback()
        raise SafeExecutionError(
            SafeExecutionErrorCode.STATE_CHANGED,
            "磁盘撤销完成后执行历史或文件索引已经变化",
        )
    _commit_completion(
        session,
        "磁盘撤销完成，但无法提交 undo 与索引状态",
    )

    return SafeExecutionResult(
        execution_id=execution.id,
        workflow_id=workflow_id,
        plan_id=UUID(execution.plan_id),
        status="UNDONE",
        before_relative_path=execution_item.before_relative_path,
        after_relative_path=execution_item.after_relative_path,
    )


def _aware_current_time(value: datetime | None) -> datetime:
    current_time = value or datetime.now(timezone.utc)
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise ValueError("now must include a timezone")
    return current_time


def _get_required_metadata(
    adapter: FileSystemAdapter,
    path: Path,
    message: str,
) -> FileMetadata:
    try:
        metadata = adapter.get_file_metadata(path)
    except OSError as error:
        raise SafeExecutionError(
            SafeExecutionErrorCode.FILE_CHANGED,
            message,
        ) from error
    if metadata is None:
        raise SafeExecutionError(
            SafeExecutionErrorCode.FILE_CHANGED,
            message,
        )
    return metadata


def _verify_optional_hash(
    adapter: FileSystemAdapter,
    path: Path,
    expected_hash: str | None,
) -> str | None:
    if expected_hash is None:
        return None
    try:
        current_hash = adapter.get_file_sha256(path)
    except OSError as error:
        raise SafeExecutionError(
            SafeExecutionErrorCode.FILE_CHANGED,
            "无法验证文件内容摘要",
        ) from error
    if current_hash != expected_hash:
        raise SafeExecutionError(
            SafeExecutionErrorCode.FILE_CHANGED,
            "文件内容摘要已经变化",
        )
    return current_hash


def _commit_completion(session: Session, message: str) -> None:
    try:
        session.commit()
    except SQLAlchemyError as error:
        session.rollback()
        raise SafeExecutionError(
            SafeExecutionErrorCode.HISTORY_WRITE_FAILED,
            message,
        ) from error
