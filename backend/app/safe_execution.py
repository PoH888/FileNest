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
from .operation_plan import OperationPlan, OperationPlanItem
from .repositories import (
    add_operation_execution,
    add_operation_execution_item,
    compare_and_set_file_entry_location,
    compare_and_set_operation_execution_item_status,
    compare_and_set_operation_execution_status,
    find_operation_execution_items,
    get_operation_execution_by_plan_id,
    get_operation_execution_by_workflow_id,
    get_workspace_by_id,
    OperationExecutionStatus,
)
from .safe_file_mover import (
    SafeFileMoveError,
    SafeFileMoveErrorCode,
    SafeFileMover,
)
from .services import require_approved_operation_plan, validate_operation_plan


class SafeExecutionRequest(BaseModel):
    """一次安全执行所需的不可变输入。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_id: UUID
    plan: OperationPlan


class SafeExecutionErrorCode(StrEnum):
    """安全执行或撤销失败时供程序稳定判断的错误码。"""

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
class SafeExecutionItemResult:
    """一个计划项的持久化执行结果。"""

    sequence_no: int
    source_file_id: int
    status: str
    before_relative_path: str
    after_relative_path: str
    error_code: str | None


@dataclass(frozen=True, slots=True)
class SafeExecutionResult:
    """一次批量执行或单项撤销的稳定业务结果。"""

    execution_id: int
    workflow_id: UUID
    plan_id: UUID
    status: str
    items: tuple[SafeExecutionItemResult, ...]

    @property
    def before_relative_path(self) -> str:
        """保留单项调用兼容性，批量结果必须读取 items。"""

        if len(self.items) != 1:
            raise ValueError("batch execution result has multiple items")
        return self.items[0].before_relative_path

    @property
    def after_relative_path(self) -> str:
        """保留单项调用兼容性，批量结果必须读取 items。"""

        if len(self.items) != 1:
            raise ValueError("batch execution result has multiple items")
        return self.items[0].after_relative_path


_RECOVERABLE_ITEM_FAILURE_CODES = frozenset(
    {
        SafeFileMoveErrorCode.SOURCE_UNAVAILABLE,
        SafeFileMoveErrorCode.TARGET_DIRECTORY_UNAVAILABLE,
        SafeFileMoveErrorCode.TARGET_CONFLICT,
    }
)


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
    """顺序执行已批准计划，并逐项持久化成功或可恢复失败。"""

    current_time = _aware_current_time(now)
    require_approved_operation_plan(
        session,
        request.workflow_id,
        request.plan,
    )
    existing_result = _get_idempotent_execution_result(session, request)
    if existing_result is not None:
        return existing_result

    validate_operation_plan(session, request.plan, now=current_time)
    workspace = get_workspace_by_id(session, request.plan.workspace_id)
    if workspace is None:
        raise SafeExecutionError(
            SafeExecutionErrorCode.WORKSPACE_NOT_FOUND,
            "执行计划对应的工作区不存在",
        )
    execution = OperationExecution(
        workflow_id=str(request.workflow_id),
        plan_id=str(request.plan.plan_id),
        workspace_id=request.plan.workspace_id,
    )
    add_operation_execution(session, execution)
    try:
        session.flush()
        execution_items: list[OperationExecutionItem] = []
        for sequence_no, operation in enumerate(
            request.plan.operations,
            start=1,
        ):
            expected_hash = operation.source_precondition.content_hash
            execution_item = OperationExecutionItem(
                execution_id=execution.id,
                sequence_no=sequence_no,
                operation_type="move",
                source_file_id=operation.source_file_id,
                before_location="workspace",
                before_relative_path=operation.source_relative_path,
                before_size_bytes=operation.source_precondition.size_bytes,
                before_mtime_ns=operation.source_precondition.mtime_ns,
                before_sha256=(
                    expected_hash.digest
                    if expected_hash is not None
                    else None
                ),
                after_location="workspace",
                after_relative_path=operation.target_relative_path,
                undo_source_relative_path=operation.target_relative_path,
                undo_target_relative_path=operation.source_relative_path,
            )
            add_operation_execution_item(session, execution_item)
            execution_items.append(execution_item)
        session.flush()
        execution_id = execution.id
        operation_items: tuple[
            tuple[OperationPlanItem, int], ...
        ] = tuple(
            (operation, execution_item.id)
            for operation, execution_item in zip(
                request.plan.operations,
                execution_items,
                strict=True,
            )
        )
        session.commit()
    except IntegrityError as error:
        session.rollback()
        existing_result = _get_idempotent_execution_result(session, request)
        if existing_result is not None:
            return existing_result
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
    mover = SafeFileMover(adapter)
    completed_count = 0
    failed_count = 0
    for operation, execution_item_id in operation_items:
        expected_hash = operation.source_precondition.content_hash
        try:
            moved_path = mover.move(
                Path(operation.source_relative_path),
                Path(operation.target_relative_path),
            )
        except SafeFileMoveError as error:
            if error.code not in _RECOVERABLE_ITEM_FAILURE_CODES:
                raise
            item_updated = compare_and_set_operation_execution_item_status(
                session,
                execution_item_id,
                "PENDING",
                next_status="FAILED",
                error_code=error.code.value,
                failed_at=current_time,
            )
            if not item_updated:
                session.rollback()
                raise SafeExecutionError(
                    SafeExecutionErrorCode.STATE_CHANGED,
                    "记录单项失败前执行明细已经变化",
                ) from error
            _commit_completion(session, "无法保存单项执行失败证据")
            failed_count += 1
            continue

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
        if not (item_updated and index_updated):
            session.rollback()
            raise SafeExecutionError(
                SafeExecutionErrorCode.STATE_CHANGED,
                "磁盘移动完成后执行明细或文件索引已经变化",
            )
        _commit_completion(
            session,
            "磁盘移动完成，但无法提交单项 after 与索引状态",
        )
        completed_count += 1

    final_status: OperationExecutionStatus
    if completed_count == len(operation_items):
        final_status = "COMPLETED"
    elif failed_count == len(operation_items):
        final_status = "FAILED"
    else:
        final_status = "PARTIALLY_COMPLETED"

    execution_updated = compare_and_set_operation_execution_status(
        session,
        execution_id,
        "EXECUTING",
        next_status=final_status,
        completed_at=current_time,
    )
    if not execution_updated:
        session.rollback()
        raise SafeExecutionError(
            SafeExecutionErrorCode.STATE_CHANGED,
            "汇总批量结果前执行主记录已经变化",
        )
    _commit_completion(session, "无法保存批量执行汇总状态")
    session.refresh(execution)
    persisted_items = find_operation_execution_items(session, execution_id)
    return _build_execution_result(
        execution,
        workflow_id=request.workflow_id,
        plan_id=request.plan.plan_id,
        execution_items=persisted_items,
    )


def retry_failed_operation_execution(
    session: Session,
    workflow_id: UUID,
    *,
    now: datetime | None = None,
) -> SafeExecutionResult:
    """只重试部分失败或全部失败执行中的失败明细。"""

    current_time = _aware_current_time(now)
    execution = get_operation_execution_by_workflow_id(
        session,
        str(workflow_id),
    )
    if execution is None:
        raise SafeExecutionError(
            SafeExecutionErrorCode.HISTORY_NOT_FOUND,
            "找不到需要重试的执行历史",
        )
    if execution.status not in {"PARTIALLY_COMPLETED", "FAILED"}:
        raise SafeExecutionError(
            SafeExecutionErrorCode.INVALID_HISTORY_STATE,
            "只有部分完成或失败的执行记录可以重试",
        )

    execution_items = find_operation_execution_items(session, execution.id)
    failed_items = [
        item for item in execution_items if item.status == "FAILED"
    ]
    if (
        not failed_items
        or any(
            item.status not in {"COMPLETED", "FAILED"}
            for item in execution_items
        )
    ):
        raise SafeExecutionError(
            SafeExecutionErrorCode.INVALID_HISTORY_STATE,
            "执行明细不满足失败项重试条件",
        )

    workspace = get_workspace_by_id(session, execution.workspace_id)
    if workspace is None:
        raise SafeExecutionError(
            SafeExecutionErrorCode.WORKSPACE_NOT_FOUND,
            "执行历史对应的工作区不存在",
        )
    adapter = FileSystemAdapter(Path(workspace.root_path))
    for item in failed_items:
        _validate_failed_item_for_retry(adapter, item)

    previous_status = execution.status
    execution_marked = compare_and_set_operation_execution_status(
        session,
        execution.id,
        previous_status,
        next_status="EXECUTING",
    )
    items_marked = True
    for item in failed_items:
        items_marked = (
            compare_and_set_operation_execution_item_status(
                session,
                item.id,
                "FAILED",
                next_status="PENDING",
            )
            and items_marked
        )
    if not (execution_marked and items_marked):
        session.rollback()
        raise SafeExecutionError(
            SafeExecutionErrorCode.STATE_CHANGED,
            "重试开始前执行历史已经变化",
        )
    _commit_completion(session, "无法保存失败项重试开始状态")

    mover = SafeFileMover(adapter)
    for item in failed_items:
        try:
            moved_path = mover.move(
                Path(item.before_relative_path),
                Path(item.after_relative_path),
            )
        except SafeFileMoveError as error:
            if error.code not in _RECOVERABLE_ITEM_FAILURE_CODES:
                raise
            item_updated = compare_and_set_operation_execution_item_status(
                session,
                item.id,
                "PENDING",
                next_status="FAILED",
                error_code=error.code.value,
                failed_at=current_time,
            )
            if not item_updated:
                session.rollback()
                raise SafeExecutionError(
                    SafeExecutionErrorCode.STATE_CHANGED,
                    "记录重试失败前执行明细已经变化",
                ) from error
            _commit_completion(session, "无法保存失败项重试失败证据")
            continue

        after_metadata = _get_required_metadata(
            adapter,
            moved_path,
            "重试后的目标文件不可用",
        )
        after_hash = _verify_optional_hash(
            adapter,
            moved_path,
            item.before_sha256,
        )
        item_updated = compare_and_set_operation_execution_item_status(
            session,
            item.id,
            "PENDING",
            next_status="COMPLETED",
            after_size_bytes=after_metadata.size_bytes,
            after_mtime_ns=after_metadata.mtime_ns,
            after_sha256=after_hash,
            completed_at=current_time,
        )
        index_updated = compare_and_set_file_entry_location(
            session,
            execution.workspace_id,
            item.source_file_id,
            item.before_relative_path,
            next_relative_path=item.after_relative_path,
            size_bytes=after_metadata.size_bytes,
            mtime_ns=after_metadata.mtime_ns,
        )
        if not (item_updated and index_updated):
            session.rollback()
            raise SafeExecutionError(
                SafeExecutionErrorCode.STATE_CHANGED,
                "重试移动完成后执行明细或文件索引已经变化",
            )
        _commit_completion(
            session,
            "重试移动完成，但无法提交单项 after 与索引状态",
        )

    persisted_items = find_operation_execution_items(session, execution.id)
    item_statuses = {item.status for item in persisted_items}
    final_status: OperationExecutionStatus
    if item_statuses == {"COMPLETED"}:
        final_status = "COMPLETED"
    elif item_statuses == {"FAILED"}:
        final_status = "FAILED"
    elif item_statuses <= {"COMPLETED", "FAILED"}:
        final_status = "PARTIALLY_COMPLETED"
    else:
        raise SafeExecutionError(
            SafeExecutionErrorCode.INVALID_HISTORY_STATE,
            "重试完成后执行明细状态无法汇总",
        )

    execution_updated = compare_and_set_operation_execution_status(
        session,
        execution.id,
        "EXECUTING",
        next_status=final_status,
        completed_at=current_time,
    )
    if not execution_updated:
        session.rollback()
        raise SafeExecutionError(
            SafeExecutionErrorCode.STATE_CHANGED,
            "汇总重试结果前执行主记录已经变化",
        )
    _commit_completion(session, "无法保存失败项重试汇总状态")

    session.refresh(execution)
    return _build_execution_result(
        execution,
        workflow_id=workflow_id,
        plan_id=UUID(execution.plan_id),
        execution_items=find_operation_execution_items(
            session,
            execution.id,
        ),
    )


def compensate_partial_operation_execution(
    session: Session,
    workflow_id: UUID,
    *,
    now: datetime | None = None,
) -> SafeExecutionResult:
    """逆序撤回部分成功执行中已经完成的文件移动。"""

    current_time = _aware_current_time(now)
    execution = get_operation_execution_by_workflow_id(
        session,
        str(workflow_id),
    )
    if execution is None:
        raise SafeExecutionError(
            SafeExecutionErrorCode.HISTORY_NOT_FOUND,
            "找不到需要补偿的执行历史",
        )
    if execution.status != "PARTIALLY_COMPLETED":
        raise SafeExecutionError(
            SafeExecutionErrorCode.INVALID_HISTORY_STATE,
            "只有部分完成的执行记录可以补偿",
        )

    execution_items = find_operation_execution_items(session, execution.id)
    completed_items = [
        item for item in execution_items if item.status == "COMPLETED"
    ]
    if (
        not completed_items
        or any(
            item.status not in {"COMPLETED", "FAILED"}
            for item in execution_items
        )
    ):
        raise SafeExecutionError(
            SafeExecutionErrorCode.INVALID_HISTORY_STATE,
            "执行明细不满足部分成功补偿条件",
        )

    workspace = get_workspace_by_id(session, execution.workspace_id)
    if workspace is None:
        raise SafeExecutionError(
            SafeExecutionErrorCode.WORKSPACE_NOT_FOUND,
            "执行历史对应的工作区不存在",
        )
    adapter = FileSystemAdapter(Path(workspace.root_path))
    for item in completed_items:
        _validate_completed_item_for_compensation(adapter, item)

    execution_marked = compare_and_set_operation_execution_status(
        session,
        execution.id,
        "PARTIALLY_COMPLETED",
        next_status="UNDOING",
    )
    items_marked = True
    for item in completed_items:
        items_marked = (
            compare_and_set_operation_execution_item_status(
                session,
                item.id,
                "COMPLETED",
                next_status="UNDOING",
            )
            and items_marked
        )
    if not (execution_marked and items_marked):
        session.rollback()
        raise SafeExecutionError(
            SafeExecutionErrorCode.STATE_CHANGED,
            "补偿开始前执行历史已经变化",
        )
    _commit_completion(session, "无法保存部分成功补偿开始状态")

    mover = SafeFileMover(adapter)
    for item in reversed(completed_items):
        restored_path = mover.move(
            Path(item.undo_source_relative_path),
            Path(item.undo_target_relative_path),
        )
        restored_metadata = _get_required_metadata(
            adapter,
            restored_path,
            "补偿后的原文件不可用",
        )
        if (
            restored_metadata.size_bytes != item.before_size_bytes
            or restored_metadata.mtime_ns != item.before_mtime_ns
        ):
            raise SafeExecutionError(
                SafeExecutionErrorCode.FILE_CHANGED,
                "补偿结果与 before 元数据不一致",
            )
        _verify_optional_hash(
            adapter,
            restored_path,
            item.before_sha256,
        )

        index_updated = compare_and_set_file_entry_location(
            session,
            execution.workspace_id,
            item.source_file_id,
            item.after_relative_path,
            next_relative_path=item.before_relative_path,
            size_bytes=restored_metadata.size_bytes,
            mtime_ns=restored_metadata.mtime_ns,
        )
        item_updated = compare_and_set_operation_execution_item_status(
            session,
            item.id,
            "UNDOING",
            next_status="UNDONE",
            undone_at=current_time,
        )
        if not (index_updated and item_updated):
            session.rollback()
            raise SafeExecutionError(
                SafeExecutionErrorCode.STATE_CHANGED,
                "补偿移动完成后执行明细或文件索引已经变化",
            )
        _commit_completion(
            session,
            "补偿移动完成，但无法提交 undo 与索引状态",
        )

    execution_updated = compare_and_set_operation_execution_status(
        session,
        execution.id,
        "UNDOING",
        next_status="UNDONE",
        undone_at=current_time,
    )
    if not execution_updated:
        session.rollback()
        raise SafeExecutionError(
            SafeExecutionErrorCode.STATE_CHANGED,
            "汇总补偿结果前执行主记录已经变化",
        )
    _commit_completion(session, "无法保存部分成功补偿汇总状态")

    session.refresh(execution)
    return _build_execution_result(
        execution,
        workflow_id=workflow_id,
        plan_id=UUID(execution.plan_id),
        execution_items=find_operation_execution_items(
            session,
            execution.id,
        ),
    )


def recover_interrupted_operation_execution(
    session: Session,
    workflow_id: UUID,
    *,
    now: datetime | None = None,
) -> SafeExecutionResult:
    """核对持久化证据与磁盘状态后恢复一条中断的执行。"""

    current_time = _aware_current_time(now)
    execution = get_operation_execution_by_workflow_id(
        session,
        str(workflow_id),
    )
    if execution is None:
        raise SafeExecutionError(
            SafeExecutionErrorCode.HISTORY_NOT_FOUND,
            "找不到需要恢复的执行历史",
        )
    if execution.status not in {"EXECUTING", "UNDOING"}:
        raise SafeExecutionError(
            SafeExecutionErrorCode.INVALID_HISTORY_STATE,
            "只有执行中或撤销中的记录可以恢复",
        )

    execution_items = find_operation_execution_items(session, execution.id)
    workspace = get_workspace_by_id(session, execution.workspace_id)
    if workspace is None:
        raise SafeExecutionError(
            SafeExecutionErrorCode.WORKSPACE_NOT_FOUND,
            "执行历史对应的工作区不存在",
        )
    adapter = FileSystemAdapter(Path(workspace.root_path))

    if execution.status == "EXECUTING":
        _recover_interrupted_execution_items(
            session,
            execution,
            execution_items,
            adapter,
            current_time,
        )
        # 生产 Session 不会在 commit 后自动过期，汇总必须重新读取 CAS 结果。
        session.expire_all()
        persisted_items = find_operation_execution_items(
            session,
            execution.id,
        )
        item_statuses = {item.status for item in persisted_items}
        final_status: OperationExecutionStatus
        if item_statuses == {"COMPLETED"}:
            final_status = "COMPLETED"
        elif item_statuses == {"FAILED"}:
            final_status = "FAILED"
        elif item_statuses <= {"COMPLETED", "FAILED"}:
            final_status = "PARTIALLY_COMPLETED"
        else:
            raise SafeExecutionError(
                SafeExecutionErrorCode.INVALID_HISTORY_STATE,
                "恢复执行后明细状态无法汇总",
            )
        execution_updated = compare_and_set_operation_execution_status(
            session,
            execution.id,
            "EXECUTING",
            next_status=final_status,
            completed_at=current_time,
        )
        completion_message = "无法保存中断执行的恢复汇总状态"
    else:
        _recover_interrupted_undo_items(
            session,
            execution,
            execution_items,
            adapter,
            current_time,
        )
        session.expire_all()
        persisted_items = find_operation_execution_items(
            session,
            execution.id,
        )
        if any(
            item.status not in {"UNDONE", "FAILED"}
            for item in persisted_items
        ):
            raise SafeExecutionError(
                SafeExecutionErrorCode.INVALID_HISTORY_STATE,
                "恢复撤销后明细状态无法汇总",
            )
        execution_updated = compare_and_set_operation_execution_status(
            session,
            execution.id,
            "UNDOING",
            next_status="UNDONE",
            undone_at=current_time,
        )
        completion_message = "无法保存中断撤销的恢复汇总状态"

    if not execution_updated:
        session.rollback()
        raise SafeExecutionError(
            SafeExecutionErrorCode.STATE_CHANGED,
            "汇总恢复结果前执行主记录已经变化",
        )
    _commit_completion(session, completion_message)

    session.expire_all()
    session.refresh(execution)
    return _build_execution_result(
        execution,
        workflow_id=workflow_id,
        plan_id=UUID(execution.plan_id),
        execution_items=find_operation_execution_items(
            session,
            execution.id,
        ),
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

    session.refresh(execution)
    return _build_execution_result(
        execution,
        workflow_id=workflow_id,
        plan_id=UUID(execution.plan_id),
        execution_items=find_operation_execution_items(
            session,
            execution.id,
        ),
    )


def _recover_interrupted_execution_items(
    session: Session,
    execution: OperationExecution,
    execution_items: list[OperationExecutionItem],
    adapter: FileSystemAdapter,
    current_time: datetime,
) -> None:
    if not execution_items or any(
        item.status not in {"PENDING", "COMPLETED", "FAILED"}
        for item in execution_items
    ):
        raise SafeExecutionError(
            SafeExecutionErrorCode.INVALID_HISTORY_STATE,
            "执行中记录包含无法恢复的明细状态",
        )

    recovery_actions: list[tuple[OperationExecutionItem, str]] = []
    for item in execution_items:
        if item.status != "PENDING":
            continue
        if (
            item.operation_type != "move"
            or item.before_location != "workspace"
            or item.after_location != "workspace"
        ):
            raise SafeExecutionError(
                SafeExecutionErrorCode.INVALID_HISTORY_STATE,
                "待执行明细缺少可安全恢复的持久化证据",
            )

        source_path = Path(item.before_relative_path)
        target_path = Path(item.after_relative_path)
        action = _interrupted_path_action(
            adapter,
            source_path,
            target_path,
            "执行中断后的源路径和目标路径状态不唯一",
        )
        evidence_path = source_path if action == "move" else target_path
        _validate_recovery_file(
            adapter,
            evidence_path,
            expected_size_bytes=item.before_size_bytes,
            expected_mtime_ns=item.before_mtime_ns,
            expected_sha256=item.before_sha256,
            message="执行中断后的文件与 before 证据不一致",
        )
        recovery_actions.append((item, action))

    mover = SafeFileMover(adapter)
    for item, action in recovery_actions:
        target_path = Path(item.after_relative_path)
        if action == "move":
            try:
                recovered_path = mover.move(
                    Path(item.before_relative_path),
                    target_path,
                )
            except SafeFileMoveError as error:
                if error.code not in _RECOVERABLE_ITEM_FAILURE_CODES:
                    raise
                item_updated = (
                    compare_and_set_operation_execution_item_status(
                        session,
                        item.id,
                        "PENDING",
                        next_status="FAILED",
                        error_code=error.code.value,
                        failed_at=current_time,
                    )
                )
                if not item_updated:
                    session.rollback()
                    raise SafeExecutionError(
                        SafeExecutionErrorCode.STATE_CHANGED,
                        "记录恢复失败前执行明细已经变化",
                    ) from error
                _commit_completion(session, "无法保存中断执行的失败证据")
                continue
        else:
            recovered_path = adapter.authorized_path(target_path)

        after_metadata = _get_required_metadata(
            adapter,
            recovered_path,
            "恢复执行后的目标文件不可用",
        )
        if (
            after_metadata.size_bytes != item.before_size_bytes
            or after_metadata.mtime_ns != item.before_mtime_ns
        ):
            raise SafeExecutionError(
                SafeExecutionErrorCode.FILE_CHANGED,
                "恢复执行后的文件与 before 证据不一致",
            )
        after_hash = _verify_optional_hash(
            adapter,
            recovered_path,
            item.before_sha256,
        )
        item_updated = compare_and_set_operation_execution_item_status(
            session,
            item.id,
            "PENDING",
            next_status="COMPLETED",
            after_size_bytes=after_metadata.size_bytes,
            after_mtime_ns=after_metadata.mtime_ns,
            after_sha256=after_hash,
            completed_at=current_time,
        )
        index_updated = compare_and_set_file_entry_location(
            session,
            execution.workspace_id,
            item.source_file_id,
            item.before_relative_path,
            next_relative_path=item.after_relative_path,
            size_bytes=after_metadata.size_bytes,
            mtime_ns=after_metadata.mtime_ns,
        )
        if not (item_updated and index_updated):
            session.rollback()
            raise SafeExecutionError(
                SafeExecutionErrorCode.STATE_CHANGED,
                "恢复执行后明细或文件索引已经变化",
            )
        _commit_completion(
            session,
            "无法提交中断执行的明细与索引状态",
        )


def _recover_interrupted_undo_items(
    session: Session,
    execution: OperationExecution,
    execution_items: list[OperationExecutionItem],
    adapter: FileSystemAdapter,
    current_time: datetime,
) -> None:
    if not execution_items or any(
        item.status not in {"UNDOING", "UNDONE", "FAILED"}
        for item in execution_items
    ):
        raise SafeExecutionError(
            SafeExecutionErrorCode.INVALID_HISTORY_STATE,
            "撤销中记录包含无法恢复的明细状态",
        )

    recovery_actions: list[tuple[OperationExecutionItem, str]] = []
    for item in execution_items:
        if item.status != "UNDOING":
            continue
        if (
            item.operation_type != "move"
            or item.before_location != "workspace"
            or item.after_location != "workspace"
            or item.after_size_bytes is None
            or item.after_mtime_ns is None
        ):
            raise SafeExecutionError(
                SafeExecutionErrorCode.INVALID_HISTORY_STATE,
                "撤销中明细缺少可安全恢复的持久化证据",
            )

        source_path = Path(item.undo_source_relative_path)
        target_path = Path(item.undo_target_relative_path)
        action = _interrupted_path_action(
            adapter,
            source_path,
            target_path,
            "撤销中断后的源路径和目标路径状态不唯一",
        )
        if action == "move":
            _validate_recovery_file(
                adapter,
                source_path,
                expected_size_bytes=item.after_size_bytes,
                expected_mtime_ns=item.after_mtime_ns,
                expected_sha256=item.after_sha256,
                message="撤销中断后的文件与 after 证据不一致",
            )
        else:
            _validate_recovery_file(
                adapter,
                target_path,
                expected_size_bytes=item.before_size_bytes,
                expected_mtime_ns=item.before_mtime_ns,
                expected_sha256=item.before_sha256,
                message="撤销中断后的文件与 before 证据不一致",
            )
        recovery_actions.append((item, action))

    mover = SafeFileMover(adapter)
    for item, action in recovery_actions:
        target_path = Path(item.undo_target_relative_path)
        if action == "move":
            restored_path = mover.move(
                Path(item.undo_source_relative_path),
                target_path,
            )
        else:
            restored_path = adapter.authorized_path(target_path)

        restored_metadata = _get_required_metadata(
            adapter,
            restored_path,
            "恢复撤销后的原文件不可用",
        )
        if (
            restored_metadata.size_bytes != item.before_size_bytes
            or restored_metadata.mtime_ns != item.before_mtime_ns
        ):
            raise SafeExecutionError(
                SafeExecutionErrorCode.FILE_CHANGED,
                "恢复撤销后的文件与 before 证据不一致",
            )
        _verify_optional_hash(
            adapter,
            restored_path,
            item.before_sha256,
        )
        index_updated = compare_and_set_file_entry_location(
            session,
            execution.workspace_id,
            item.source_file_id,
            item.after_relative_path,
            next_relative_path=item.before_relative_path,
            size_bytes=restored_metadata.size_bytes,
            mtime_ns=restored_metadata.mtime_ns,
        )
        item_updated = compare_and_set_operation_execution_item_status(
            session,
            item.id,
            "UNDOING",
            next_status="UNDONE",
            undone_at=current_time,
        )
        if not (index_updated and item_updated):
            session.rollback()
            raise SafeExecutionError(
                SafeExecutionErrorCode.STATE_CHANGED,
                "恢复撤销后明细或文件索引已经变化",
            )
        _commit_completion(
            session,
            "无法提交中断撤销的明细与索引状态",
        )


def _interrupted_path_action(
    adapter: FileSystemAdapter,
    source_path: Path,
    target_path: Path,
    message: str,
) -> str:
    source_exists = adapter.path_exists(source_path)
    target_exists = adapter.path_exists(target_path)
    if source_exists == target_exists:
        raise SafeExecutionError(
            SafeExecutionErrorCode.FILE_CHANGED,
            message,
        )
    return "move" if source_exists else "reconcile"


def _validate_recovery_file(
    adapter: FileSystemAdapter,
    path: Path,
    *,
    expected_size_bytes: int,
    expected_mtime_ns: int,
    expected_sha256: str | None,
    message: str,
) -> None:
    metadata = _get_required_metadata(adapter, path, message)
    if (
        metadata.size_bytes != expected_size_bytes
        or metadata.mtime_ns != expected_mtime_ns
    ):
        raise SafeExecutionError(
            SafeExecutionErrorCode.FILE_CHANGED,
            message,
        )
    _verify_optional_hash(adapter, path, expected_sha256)


def _validate_failed_item_for_retry(
    adapter: FileSystemAdapter,
    execution_item: OperationExecutionItem,
) -> None:
    if (
        execution_item.status != "FAILED"
        or execution_item.operation_type != "move"
        or execution_item.before_location != "workspace"
        or execution_item.after_location != "workspace"
        or execution_item.error_code is None
        or execution_item.failed_at is None
    ):
        raise SafeExecutionError(
            SafeExecutionErrorCode.INVALID_HISTORY_STATE,
            "失败明细缺少可安全重试的持久化证据",
        )

    try:
        source_metadata = adapter.get_file_metadata(
            Path(execution_item.before_relative_path)
        )
    except OSError:
        # 源文件仍不可用时交给 mover 生成新的可恢复失败证据。
        return
    if source_metadata is None:
        return
    if (
        source_metadata.size_bytes != execution_item.before_size_bytes
        or source_metadata.mtime_ns != execution_item.before_mtime_ns
    ):
        raise SafeExecutionError(
            SafeExecutionErrorCode.FILE_CHANGED,
            "失败项源文件已经变化，拒绝按旧证据重试",
        )
    _verify_optional_hash(
        adapter,
        Path(execution_item.before_relative_path),
        execution_item.before_sha256,
    )


def _validate_completed_item_for_compensation(
    adapter: FileSystemAdapter,
    execution_item: OperationExecutionItem,
) -> None:
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
            "完成明细缺少可安全补偿的 after 证据",
        )

    compensation_source = Path(execution_item.undo_source_relative_path)
    compensation_target = Path(execution_item.undo_target_relative_path)
    current_metadata = _get_required_metadata(
        adapter,
        compensation_source,
        "补偿源文件不存在或不是普通文件",
    )
    if (
        current_metadata.size_bytes != execution_item.after_size_bytes
        or current_metadata.mtime_ns != execution_item.after_mtime_ns
    ):
        raise SafeExecutionError(
            SafeExecutionErrorCode.FILE_CHANGED,
            "执行后的文件已经变化，拒绝补偿",
        )
    _verify_optional_hash(
        adapter,
        compensation_source,
        execution_item.after_sha256,
    )
    if adapter.path_exists(compensation_target):
        raise SafeExecutionError(
            SafeExecutionErrorCode.UNDO_TARGET_CONFLICT,
            "原路径已经被占用，拒绝覆盖补偿",
        )


def _get_idempotent_execution_result(
    session: Session,
    request: SafeExecutionRequest,
) -> SafeExecutionResult | None:
    workflow_key = str(request.workflow_id)
    plan_key = str(request.plan.plan_id)
    execution_by_workflow = get_operation_execution_by_workflow_id(
        session,
        workflow_key,
    )
    execution_by_plan = get_operation_execution_by_plan_id(
        session,
        plan_key,
    )
    if execution_by_workflow is None and execution_by_plan is None:
        return None

    if (
        execution_by_workflow is not None
        and execution_by_plan is not None
        and execution_by_workflow.id != execution_by_plan.id
    ):
        raise SafeExecutionError(
            SafeExecutionErrorCode.HISTORY_EXISTS,
            "当前工作流和计划分别绑定了不同执行历史",
        )

    execution = execution_by_workflow or execution_by_plan
    if (
        execution is None
        or execution.workflow_id != workflow_key
        or execution.plan_id != plan_key
    ):
        raise SafeExecutionError(
            SafeExecutionErrorCode.HISTORY_EXISTS,
            "当前工作流或计划已经绑定其他执行历史",
        )

    execution_items = find_operation_execution_items(session, execution.id)
    if len(execution_items) != len(request.plan.operations):
        raise SafeExecutionError(
            SafeExecutionErrorCode.INVALID_HISTORY_STATE,
            "执行历史明细数量与确定计划不一致",
        )
    for sequence_no, (execution_item, operation) in enumerate(
        zip(execution_items, request.plan.operations, strict=True),
        start=1,
    ):
        if (
            execution_item.sequence_no != sequence_no
            or execution_item.source_file_id != operation.source_file_id
            or execution_item.before_relative_path
            != operation.source_relative_path
            or execution_item.after_relative_path
            != operation.target_relative_path
        ):
            raise SafeExecutionError(
                SafeExecutionErrorCode.INVALID_HISTORY_STATE,
                "执行历史明细与确定计划不一致",
            )

    return _build_execution_result(
        execution,
        workflow_id=request.workflow_id,
        plan_id=request.plan.plan_id,
        execution_items=execution_items,
    )


def _build_execution_result(
    execution: OperationExecution,
    *,
    workflow_id: UUID,
    plan_id: UUID,
    execution_items: list[OperationExecutionItem],
) -> SafeExecutionResult:
    return SafeExecutionResult(
        execution_id=execution.id,
        workflow_id=workflow_id,
        plan_id=plan_id,
        status=execution.status,
        items=tuple(
            SafeExecutionItemResult(
                sequence_no=item.sequence_no,
                source_file_id=item.source_file_id,
                status=item.status,
                before_relative_path=item.before_relative_path,
                after_relative_path=item.after_relative_path,
                error_code=item.error_code,
            )
            for item in execution_items
        ),
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
