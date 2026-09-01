"""最小界面使用的整理计划 HTTP 边界。"""

from collections.abc import Iterator
from datetime import datetime
import sqlite3
from time import sleep
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy.orm import Session

from .approval_recovery import (
    ApprovalRecoveryErrorCode,
    scan_waiting_approval_tasks,
)
from .database import get_session
from .events import build_workflow_event_stream
from .models import OperationPlanRecord
from .operation_plan import OperationPlan
from .operation_plan import OperationPlanItem
from .operation_projection import OperationProjection
from .organization_decisions import (
    OrganizationDecisionError,
    OrganizationDecisionErrorCode,
    apply_organization_decision,
    apply_organization_plan_edit,
)
from .organization_planning import (
    CreateApprovalWorkflowRequest,
    EditOrganizationPlanRequest,
    OrganizationTargetSelection,
    create_waiting_approval_workflow,
)
from .path_policy import PathPolicyError
from .repositories import (
    count_operation_plans,
    count_pending_approval_requests,
    find_approval_audit_events,
    find_operation_execution_items,
    find_operation_plans,
    find_pending_approval_requests,
    get_approval_request_by_workflow_id,
    get_operation_execution_by_id,
    get_operation_plan_by_id,
    get_operation_execution_by_workflow_id,
    get_operation_projection_by_workflow_id,
    get_workspace_by_id,
)
from .safe_execution import (
    SafeExecutionError,
    SafeExecutionErrorCode,
    SafeExecutionRequest,
    SafeExecutionResult,
    execute_safe_operation_plan,
    undo_safe_operation_execution,
)
from .services import (
    ApprovalTransitionError,
    ApprovalTransitionErrorCode,
    FileEntryNotFoundError,
    OperationPlanApprovalError,
    OperationPlanApprovalErrorCode,
    OperationPlanExpiredError,
    OperationPlanSourceChangedError,
    OperationPlanSourceMismatchError,
    OperationPlanTargetConflictError,
    OperationPlanTargetUnavailableError,
    OperationPreviewPathUnavailableError,
    OperationPlanPersistenceError,
    WorkspacePolicyError,
    WorkspaceNotFoundError,
    get_operation_plan,
    validate_operation_plan,
)
from .operation_status import (
    OperationStatus,
    map_approval_status_to_operation_status,
    map_workflow_status_to_operation_status,
)
from .workflow import WorkflowState, WorkflowStatus, WorkflowTransitionError
from .workflow_graph import (
    WorkflowCheckpointError,
    workflow_checkpoint_config,
)
from .workflow_runtime import get_workflow_graph


router = APIRouter(prefix="/api/v1")
_WORKFLOW_EVENT_POLL_SECONDS = 0.1
_POST_APPROVAL_OPERATION_STATUSES = frozenset(
    {
        OperationStatus.EXECUTING,
        OperationStatus.PARTIAL_FAILED,
        OperationStatus.COMPLETED,
        OperationStatus.UNDOING,
        OperationStatus.UNDONE,
        OperationStatus.COMPENSATED,
        OperationStatus.FAILED,
    }
)
ApprovalStatus = Literal[
    "WAITING_APPROVAL",
    "APPROVED",
    "REJECTED",
    "CANCELLED",
]
OperationType = Literal["move", "quarantine", "rename"]
PlanValidationStatus = Literal["valid", "blocked"]
RecoveryStatus = Literal["available", "blocked", "not_applicable"]
OperationPlanStatus = Literal[
    "WAITING_APPROVAL",
    "APPROVED",
    "REJECTED",
    "CANCELLED",
    "SUPERSEDED",
]


class OrganizationWorkflowResponse(BaseModel):
    """供页面展示的工作流、不可变计划和审批状态。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_id: UUID
    status: WorkflowStatus
    revision: int
    wait_reason_code: str | None
    operation_plan: OperationPlan
    approval_status: ApprovalStatus
    operation: OperationProjection


class PendingApprovalSourceSummary(BaseModel):
    """待审批列表中的只读源文件和目标摘要。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_file_id: int = Field(ge=1)
    source_relative_path: str
    target_relative_path: str


class PendingApprovalItemResponse(BaseModel):
    """一条来自业务数据库的待审批事实快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: int = Field(ge=1)
    workspace_id: int = Field(ge=1)
    workflow_id: UUID
    plan_id: UUID
    operation_type: OperationType
    source_summary: tuple[PendingApprovalSourceSummary, ...] = Field(
        min_length=1,
    )
    targets: tuple[str, ...] = Field(min_length=1)
    created_at: datetime
    current_revision: int = Field(ge=0)
    approval_status: ApprovalStatus
    recovery_status: RecoveryStatus
    recovery_error_code: str | None = None


class PendingApprovalListResponse(BaseModel):
    """分页返回待审批事实，不包含任何写盘动作。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[PendingApprovalItemResponse, ...]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total: int = Field(ge=0)
    has_more: bool


class OperationPlanDetailResponse(BaseModel):
    """重新从业务数据库加载并校验后的计划详情。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: UUID
    workspace_id: int = Field(ge=1)
    workflow_id: UUID
    operation_type: OperationType
    status: str
    created_at: datetime
    current_revision: int = Field(ge=0)
    approval_status: ApprovalStatus | None
    validation_status: PlanValidationStatus
    validation_error_code: str | None = None
    recovery_status: RecoveryStatus
    recovery_error_code: str | None = None
    operations: tuple[OperationPlanItem, ...] = Field(min_length=1)


class OperationPlanListResponse(BaseModel):
    """按工作区和计划状态分页返回计划事实。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[OperationPlanDetailResponse, ...]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total: int = Field(ge=0)
    has_more: bool


class OrganizationDecisionRequest(BaseModel):
    """页面针对当前所见计划提交的最小人工决定。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["approve", "edit", "reject", "cancel"]
    expected_plan_id: UUID
    expected_revision: int | None = Field(default=None, ge=0)
    changes: tuple[OrganizationTargetSelection, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    @model_validator(mode="after")
    def validate_action_payload(self) -> "OrganizationDecisionRequest":
        if self.action == "edit" and self.changes is None:
            raise ValueError("edit action requires changes")
        if self.action != "edit" and self.changes is not None:
            raise ValueError("changes are only allowed for edit action")
        return self


class SafeExecutionItemResponse(BaseModel):
    """供界面展示的一条安全执行或撤销结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence_no: int
    source_file_id: int
    status: str
    before_relative_path: str
    after_relative_path: str
    error_code: str | None


class SafeExecutionResponse(BaseModel):
    """安全执行边界对界面公开的最小结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_id: int
    workflow_id: UUID
    plan_id: UUID
    status: str
    items: tuple[SafeExecutionItemResponse, ...]

    @classmethod
    def from_result(cls, result: SafeExecutionResult) -> "SafeExecutionResponse":
        return cls(
            execution_id=result.execution_id,
            workflow_id=result.workflow_id,
            plan_id=result.plan_id,
            status=result.status,
            items=tuple(
                SafeExecutionItemResponse(
                    sequence_no=item.sequence_no,
                    source_file_id=item.source_file_id,
                    status=item.status,
                    before_relative_path=item.before_relative_path,
                    after_relative_path=item.after_relative_path,
                    error_code=item.error_code,
                )
                for item in result.items
            ),
        )


def _plan_validation_error_code(error: Exception) -> str:
    """把详情查询中的重新校验失败收敛为稳定、无路径细节的程序码。"""

    if isinstance(error, OperationPlanExpiredError):
        return "operation_plan_expired"
    if isinstance(error, WorkspaceNotFoundError):
        return "workspace_not_found"
    if isinstance(error, FileEntryNotFoundError):
        return "file_not_found"
    if isinstance(error, OperationPlanSourceMismatchError):
        return "operation_plan_source_mismatch"
    if isinstance(error, OperationPlanSourceChangedError):
        return "operation_plan_source_changed"
    if isinstance(error, OperationPlanTargetConflictError):
        return "operation_plan_target_conflict"
    if isinstance(error, OperationPlanTargetUnavailableError):
        return "operation_plan_target_unavailable"
    if isinstance(error, PathPolicyError):
        return "operation_plan_policy_denied"
    if isinstance(error, WorkspacePolicyError):
        return error.code.value
    return "operation_plan_validation_failed"


def _validate_plan_for_query(
    session: Session,
    plan: OperationPlan,
) -> tuple[PlanValidationStatus, str | None]:
    """读取详情时重新走执行前校验，但只返回结果，不产生文件副作用。"""

    try:
        validate_operation_plan(session, plan)
    except (
        FileEntryNotFoundError,
        OperationPlanExpiredError,
        OperationPlanSourceChangedError,
        OperationPlanSourceMismatchError,
        OperationPlanTargetConflictError,
        OperationPlanTargetUnavailableError,
        PathPolicyError,
        WorkspacePolicyError,
        WorkspaceNotFoundError,
    ) as error:
        return "blocked", _plan_validation_error_code(error)
    return "valid", None


def _pending_recovery_states(
    session: Session,
    graph: CompiledStateGraph,
    approval_ids: tuple[int, ...],
) -> dict[int, tuple[RecoveryStatus, str | None]]:
    """查询本次响应涉及的 checkpoint 状态，失败时统一显示为安全阻断。"""

    states: dict[int, tuple[RecoveryStatus, str | None]] = {
        approval_id: (
            "blocked",
            ApprovalRecoveryErrorCode.RECOVERY_UNAVAILABLE.value,
        )
        for approval_id in approval_ids
    }
    if not approval_ids:
        return states

    try:
        scan = scan_waiting_approval_tasks(session, graph)
    except (OSError, ValueError, RuntimeError, sqlite3.Error):
        return states

    for task in scan.recovered_tasks:
        if task.approval_id in states:
            states[task.approval_id] = ("available", None)
    for issue in scan.issues:
        if issue.approval_id in states:
            states[issue.approval_id] = ("blocked", issue.code)
    return states


def _check_expected_operation_snapshot(
    session: Session,
    workflow_id: UUID,
    *,
    expected_plan_id: UUID | None = None,
    expected_revision: int | None = None,
) -> None:
    """在写操作前检查页面所见的 plan/revision 快照。"""

    if expected_plan_id is None and expected_revision is None:
        return
    operation = get_operation_projection_by_workflow_id(
        session,
        str(workflow_id),
    )
    if operation is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "operation_state_conflict",
                "message": "Operation 当前状态不可用。",
            },
        )
    if expected_plan_id is not None and operation.plan_id != expected_plan_id:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "organization_workflow_plan_mismatch",
                "message": "审批决定与当前工作流状态冲突。",
            },
        )
    if expected_revision is not None and operation.revision != expected_revision:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "organization_workflow_revision_conflict",
                "message": "页面所见 revision 已经过期。",
            },
        )


def _load_plan_for_query(
    session: Session,
    plan_id: UUID,
    *,
    workspace_id: int | None = None,
 ) -> tuple[OperationPlan, OperationPlanRecord]:
    """按持久化 plan_id 读取计划，并先执行工作区隔离检查。"""

    record = get_operation_plan_by_id(session, str(plan_id))
    if record is None or (
        workspace_id is not None and record.workspace_id != workspace_id
    ):
        raise HTTPException(
            status_code=404,
            detail={
                "code": "operation_plan_not_found",
                "message": "操作计划不存在。",
            },
        )
    if get_workspace_by_id(session, record.workspace_id) is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workspace_not_found",
                "message": "工作区不存在。",
            },
        )

    try:
        workflow_id = UUID(record.workflow_id)
        plan = get_operation_plan(
            session,
            plan_id,
            workflow_id=workflow_id,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "operation_plan_state_invalid",
                "message": "操作计划当前不可用。",
            },
        ) from error
    except OperationPlanPersistenceError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "operation_plan_state_invalid",
                "message": "操作计划当前不可用。",
            },
        ) from error
    if plan is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "operation_plan_state_conflict",
                "message": "操作计划与工作流记录不一致。",
            },
        )
    _validate_plan_associations(session, plan, record)
    return plan, record


def _validate_plan_associations(
    session: Session,
    plan: OperationPlan,
    record: OperationPlanRecord,
) -> None:
    """确认 plan、workflow、approval、Operation 和 execution 仍是同一事实链。"""

    try:
        workflow_id = UUID(record.workflow_id)
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "operation_plan_state_invalid",
                "message": "操作计划工作流关联数据损坏。",
            },
        ) from error

    approval = get_approval_request_by_workflow_id(session, record.workflow_id)
    operation = get_operation_projection_by_workflow_id(
        session,
        record.workflow_id,
    )
    if approval is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "operation_plan_approval_missing",
                "message": "操作计划缺少审批关联，不能安全使用。",
            },
        )
    if operation is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "operation_plan_operation_missing",
                "message": "操作计划缺少 Operation 关联，不能安全使用。",
            },
        )
    if operation.workflow_id != workflow_id:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "operation_plan_workflow_mismatch",
                "message": "操作计划与工作流关联不一致。",
            },
        )

    if record.status == "SUPERSEDED":
        history = find_approval_audit_events(session, approval.id)
        if not any(
            record.plan_id in {event.previous_plan_id, event.next_plan_id}
            for event in history
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "operation_plan_history_mismatch",
                    "message": "操作计划不在审批历史关联中。",
                },
            )
    else:
        if approval.plan_id != record.plan_id:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "operation_plan_approval_mismatch",
                    "message": "操作计划与当前审批记录不一致。",
                },
            )
        if operation.plan_id != plan.plan_id or operation.approval_id != approval.id:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "operation_plan_operation_mismatch",
                    "message": "操作计划与当前 Operation 投影不一致。",
                },
            )

        allowed_operation_statuses = {
            "WAITING_APPROVAL": {"WAITING_APPROVAL"},
            "APPROVED": {
                "APPROVED",
                "EXECUTING",
                "PARTIAL_FAILED",
                "COMPLETED",
                "UNDOING",
                "UNDONE",
                "COMPENSATED",
                "FAILED",
            },
            "REJECTED": {"REJECTED"},
            "CANCELLED": {"CANCELLED"},
        }
        if operation.overall_status.value not in allowed_operation_statuses.get(
            record.status,
            set(),
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "operation_plan_status_mismatch",
                    "message": "操作计划与 Operation 状态不一致。",
                },
            )

    if operation.execution_id is not None:
        execution = get_operation_execution_by_id(
            session,
            operation.execution_id,
        )
        if (
            execution is None
            or execution.workflow_id != record.workflow_id
            or execution.plan_id != record.plan_id
            or execution.workspace_id != record.workspace_id
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "operation_plan_execution_mismatch",
                    "message": "操作计划与执行历史关联不一致。",
                },
            )


def _operation_plan_detail_response(
    session: Session,
    graph: CompiledStateGraph,
    plan: OperationPlan,
    record: OperationPlanRecord,
) -> OperationPlanDetailResponse:
    """组合业务库、Operation 投影和 checkpoint 的只读详情。"""

    # 调用方只传入 OperationPlanRecord；用属性访问避免把 ORM 类型扩散到契约层。
    workflow_id = UUID(record.workflow_id)
    approval = get_approval_request_by_workflow_id(session, str(workflow_id))
    operation = get_operation_projection_by_workflow_id(
        session,
        str(workflow_id),
    )
    current_revision = 0
    if operation is not None and operation.plan_id == plan.plan_id:
        current_revision = operation.revision

    validation_status, validation_error_code = _validate_plan_for_query(
        session,
        plan,
    )
    recovery_status: RecoveryStatus = "not_applicable"
    recovery_error_code: str | None = None
    approval_status: ApprovalStatus | None = None
    if approval is not None and approval.plan_id == str(plan.plan_id):
        approval_status = approval.status
        states = _pending_recovery_states(session, graph, (approval.id,))
        recovery_status, recovery_error_code = states[approval.id]

    return OperationPlanDetailResponse(
        plan_id=plan.plan_id,
        workspace_id=plan.workspace_id,
        workflow_id=workflow_id,
        operation_type=plan.operations[0].operation_type,
        status=record.status,
        created_at=plan.created_at,
        current_revision=current_revision,
        approval_status=approval_status,
        validation_status=validation_status,
        validation_error_code=validation_error_code,
        recovery_status=recovery_status,
        recovery_error_code=recovery_error_code,
        operations=plan.operations,
    )


def _workflow_response(
    session: Session,
    graph: CompiledStateGraph,
    workflow_id: UUID,
) -> OrganizationWorkflowResponse:
    saved_values = graph.get_state(
        workflow_checkpoint_config(workflow_id)
    ).values
    approval = get_approval_request_by_workflow_id(session, str(workflow_id))

    if "workflow" not in saved_values and approval is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workflow_not_found",
                "message": "工作流不存在。",
            },
        )
    if "workflow" not in saved_values or approval is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "workflow_state_conflict",
                "message": "工作流状态当前不一致。",
            },
        )

    try:
        workflow = WorkflowState.model_validate(saved_values["workflow"])
    except ValidationError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "workflow_state_invalid",
                "message": "工作流状态当前不可用。",
            },
        ) from error

    try:
        operation_plan = get_operation_plan(
            session,
            approval.plan_id,
            workflow_id=workflow_id,
        )
    except OperationPlanPersistenceError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "workflow_state_invalid",
                "message": "业务操作计划当前不可用。",
            },
        ) from error

    operation = get_operation_projection_by_workflow_id(
        session,
        str(workflow_id),
    )
    try:
        expected_approval_status = map_approval_status_to_operation_status(
            approval.status,
        )
        expected_workflow_status = map_workflow_status_to_operation_status(
            workflow.status,
            error_code=workflow.error_code,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "workflow_state_conflict",
                "message": "工作流状态当前不一致。",
            },
        ) from error

    operation_state_is_consistent = (
        operation is not None
        and (
            (
                operation.overall_status == expected_approval_status
                and operation.overall_status == expected_workflow_status
            )
            or (
                operation.overall_status in _POST_APPROVAL_OPERATION_STATUSES
                and expected_approval_status == OperationStatus.APPROVED
                and expected_workflow_status == OperationStatus.APPROVED
            )
        )
    )
    if (
        workflow.workflow_id != workflow_id
        or str(workflow.operation_plan.plan_id) != approval.plan_id
        or operation_plan is None
        or operation_plan.plan_id != workflow.operation_plan.plan_id
        or approval.status not in {
            "WAITING_APPROVAL",
            "APPROVED",
            "REJECTED",
            "CANCELLED",
        }
        or operation is None
        or operation.workflow_id != workflow_id
        or operation.plan_id != workflow.operation_plan.plan_id
        or operation.approval_id != approval.id
        or not operation_state_is_consistent
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "workflow_state_conflict",
                "message": "工作流状态当前不一致。",
            },
        )

    return OrganizationWorkflowResponse(
        workflow_id=workflow.workflow_id,
        status=workflow.status,
        revision=workflow.revision,
        wait_reason_code=workflow.wait_reason_code,
        operation_plan=operation_plan,
        approval_status=approval.status,
        operation=operation,
    )


def _ready_workflow_response(
    session: Session,
    graph: CompiledStateGraph,
    workflow_id: UUID,
) -> OrganizationWorkflowResponse:
    """只让已批准且进入 ready 的工作流进入文件副作用边界。"""

    workflow = _workflow_response(session, graph, workflow_id)
    if workflow.status != "ready" or workflow.approval_status != "APPROVED":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "organization_workflow_not_ready",
                "message": "工作流尚未获得批准，不能执行文件操作。",
            },
        )
    return workflow


def _safe_execution_http_error(
    error: SafeExecutionError | OperationPlanApprovalError,
) -> HTTPException:
    """把安全执行失败收敛为稳定的 HTTP 错误，不泄露底层异常细节。"""

    not_found_codes = {
        SafeExecutionErrorCode.HISTORY_NOT_FOUND,
        SafeExecutionErrorCode.WORKSPACE_NOT_FOUND,
        SafeExecutionErrorCode.FILE_ENTRY_NOT_FOUND,
        OperationPlanApprovalErrorCode.NOT_FOUND,
    }
    status_code = 404 if error.code in not_found_codes else 409
    if error.code == SafeExecutionErrorCode.HISTORY_NOT_FOUND:
        message = "找不到可撤销的执行历史。"
    elif isinstance(error, OperationPlanApprovalError):
        message = "操作计划尚未获得人工批准。"
    else:
        message = "安全文件操作当前不可用。"
    return HTTPException(
        status_code=status_code,
        detail={"code": str(error.code), "message": message},
    )


def _operation_plan_http_error(error: Exception) -> HTTPException:
    """把执行前的工作区、计划和路径校验失败映射到 API 边界。"""

    if isinstance(error, WorkspaceNotFoundError):
        status_code = 404
        detail = {
            "code": "workspace_not_found",
            "message": "工作区不存在。",
        }
    elif isinstance(error, FileEntryNotFoundError):
        status_code = 404
        detail = {
            "code": "file_not_found",
            "message": "文件索引不存在。",
        }
    else:
        status_code = 409
        detail = {
            "code": "organization_plan_unavailable",
            "message": "当前文件状态无法安全执行。",
        }
    return HTTPException(status_code=status_code, detail=detail)


def _stream_workflow_events(
    session: Session,
    workflow_id: UUID,
    *,
    after_event_id: int = 0,
) -> Iterator[str]:
    """轮询已提交的审批、执行和撤销事实，保持 SSE 可恢复。"""

    emitted_event_count = after_event_id
    workflow_key = str(workflow_id)
    while True:
        # 每轮结束只读事务，保证长连接能看到其他请求的新提交。
        session.rollback()
        approval = get_approval_request_by_workflow_id(session, workflow_key)
        if approval is None:
            return
        audits = find_approval_audit_events(session, approval.id)
        execution = get_operation_execution_by_workflow_id(
            session,
            workflow_key,
        )
        execution_items = (
            find_operation_execution_items(session, execution.id)
            if execution is not None
            else ()
        )
        events = build_workflow_event_stream(
            approval,
            audits,
            execution,
            execution_items,
        )
        is_terminal = approval.status in {"REJECTED", "CANCELLED"} or (
            execution is not None
            and execution.status
            in {"PARTIALLY_COMPLETED", "FAILED", "UNDONE"}
        )
        session.rollback()

        for event in events[emitted_event_count:]:
            emitted_event_count = event.event_id
            yield event.encode()

        if is_terminal:
            return
        sleep(_WORKFLOW_EVENT_POLL_SECONDS)


@router.post(
    "/workflows",
    response_model=OrganizationWorkflowResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_organization_workflow(
    request: CreateApprovalWorkflowRequest,
    session: Session = Depends(get_session),
    graph: CompiledStateGraph = Depends(get_workflow_graph),
) -> OrganizationWorkflowResponse:
    """通过既有 Service 创建待人工审批的确定计划。"""

    try:
        created = create_waiting_approval_workflow(session, graph, request)
    except WorkspaceNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workspace_not_found",
                "message": "工作区不存在。",
            },
        ) from error
    except FileEntryNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "file_not_found",
                "message": "文件索引不存在。",
            },
        ) from error
    except (
        OperationPreviewPathUnavailableError,
        OperationPlanExpiredError,
        OperationPlanSourceChangedError,
        OperationPlanSourceMismatchError,
        OperationPlanTargetConflictError,
        OperationPlanTargetUnavailableError,
        PathPolicyError,
        WorkspacePolicyError,
    ) as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": (
                    error.code.value
                    if isinstance(error, WorkspacePolicyError)
                    else "organization_plan_unavailable"
                ),
                "message": (
                    str(error)
                    if isinstance(error, WorkspacePolicyError)
                    else "当前文件状态无法生成安全计划。"
                ),
            },
        ) from error
    except WorkflowCheckpointError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": str(error.code),
                "message": "工作流 checkpoint 冲突。",
            },
        ) from error

    return _workflow_response(
        session,
        graph,
        created.workflow.workflow_id,
    )


@router.get(
    "/approvals/pending",
    response_model=PendingApprovalListResponse,
)
def list_pending_approvals(
    workspace_id: int = Query(..., ge=1),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_session),
    graph: CompiledStateGraph = Depends(get_workflow_graph),
) -> PendingApprovalListResponse:
    """从业务数据库分页读取待审批计划，并显示恢复阻断原因。"""

    if get_workspace_by_id(session, workspace_id) is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workspace_not_found",
                "message": "工作区不存在。",
            },
        )

    offset = (page - 1) * page_size
    rows = find_pending_approval_requests(
        session,
        workspace_id,
        offset=offset,
        limit=page_size,
    )
    total = count_pending_approval_requests(session, workspace_id)
    recovery_states = _pending_recovery_states(
        session,
        graph,
        tuple(approval.id for approval, _ in rows),
    )
    items: list[PendingApprovalItemResponse] = []
    for approval, _record in rows:
        try:
            workflow_id = UUID(approval.workflow_id)
            plan = get_operation_plan(
                session,
                approval.plan_id,
                workflow_id=workflow_id,
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "approval_business_state_invalid",
                    "message": "待审批业务记录当前不可用。",
                },
            ) from error
        except OperationPlanPersistenceError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "operation_plan_state_invalid",
                    "message": "待审批操作计划当前不可用。",
                },
            ) from error
        if plan is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "approval_plan_mismatch",
                    "message": "待审批记录与操作计划不一致。",
                },
            )
        operation = get_operation_projection_by_workflow_id(
            session,
            approval.workflow_id,
        )
        recovery_status, recovery_error_code = recovery_states[approval.id]
        items.append(
            PendingApprovalItemResponse(
                approval_id=approval.id,
                workspace_id=plan.workspace_id,
                workflow_id=workflow_id,
                plan_id=plan.plan_id,
                operation_type=plan.operations[0].operation_type,
                source_summary=tuple(
                    PendingApprovalSourceSummary(
                        source_file_id=operation_item.source_file_id,
                        source_relative_path=operation_item.source_relative_path,
                        target_relative_path=operation_item.target_relative_path,
                    )
                    for operation_item in plan.operations
                ),
                targets=tuple(
                    operation_item.target_relative_path
                    for operation_item in plan.operations
                ),
                created_at=plan.created_at,
                current_revision=(
                    operation.revision
                    if operation is not None
                    and operation.plan_id == plan.plan_id
                    else 0
                ),
                approval_status=approval.status,
                recovery_status=recovery_status,
                recovery_error_code=recovery_error_code,
            )
        )

    return PendingApprovalListResponse(
        items=tuple(items),
        page=page,
        page_size=page_size,
        total=total,
        has_more=offset + len(items) < total,
    )


@router.get(
    "/operation-plans",
    response_model=OperationPlanListResponse,
)
def list_operation_plan_details(
    workspace_id: int = Query(..., ge=1),
    plan_status: OperationPlanStatus | None = Query(
        default=None,
        alias="status",
    ),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_session),
    graph: CompiledStateGraph = Depends(get_workflow_graph),
) -> OperationPlanListResponse:
    """从业务数据库分页读取完整计划，并逐项校验关联事实。"""

    if get_workspace_by_id(session, workspace_id) is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workspace_not_found",
                "message": "工作区不存在。",
            },
        )

    offset = (page - 1) * page_size
    records = find_operation_plans(
        session,
        workspace_id,
        plan_status=plan_status,
        offset=offset,
        limit=page_size,
    )
    total = count_operation_plans(
        session,
        workspace_id,
        plan_status=plan_status,
    )
    items: list[OperationPlanDetailResponse] = []
    for record in records:
        try:
            plan_id = UUID(record.plan_id)
        except ValueError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "operation_plan_state_invalid",
                    "message": "操作计划标识数据损坏。",
                },
            ) from error
        plan, loaded_record = _load_plan_for_query(
            session,
            plan_id,
            workspace_id=workspace_id,
        )
        items.append(
            _operation_plan_detail_response(
                session,
                graph,
                plan,
                loaded_record,
            )
        )

    return OperationPlanListResponse(
        items=tuple(items),
        page=page,
        page_size=page_size,
        total=total,
        has_more=offset + len(items) < total,
    )


@router.get(
    "/operation-plans/{plan_id}",
    response_model=OperationPlanDetailResponse,
)
def get_operation_plan_detail(
    plan_id: UUID,
    workspace_id: int | None = Query(default=None, ge=1),
    session: Session = Depends(get_session),
    graph: CompiledStateGraph = Depends(get_workflow_graph),
) -> OperationPlanDetailResponse:
    """返回重新加载、校验并标注恢复状态的服务器端计划事实。"""

    plan, record = _load_plan_for_query(
        session,
        plan_id,
        workspace_id=workspace_id,
    )
    return _operation_plan_detail_response(session, graph, plan, record)


@router.get(
    "/workflows/{workflow_id}",
    response_model=OrganizationWorkflowResponse,
)
def get_organization_workflow(
    workflow_id: UUID,
    workspace_id: int | None = Query(default=None, ge=1),
    session: Session = Depends(get_session),
    graph: CompiledStateGraph = Depends(get_workflow_graph),
) -> OrganizationWorkflowResponse:
    """读取 checkpoint 与审批记录一致的计划快照。"""

    if workspace_id is not None:
        approval = get_approval_request_by_workflow_id(session, str(workflow_id))
        record = (
            get_operation_plan_by_id(session, approval.plan_id)
            if approval is not None
            else None
        )
        if record is None or record.workspace_id != workspace_id:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "workflow_not_found",
                    "message": "工作流不存在。",
                },
            )

    return _workflow_response(session, graph, workflow_id)


@router.get("/workflows/{workflow_id}/events")
def stream_organization_workflow_events(
    workflow_id: UUID,
    last_event_id: int | None = Header(
        default=None,
        alias="Last-Event-ID",
        ge=1,
    ),
    session: Session = Depends(get_session),
) -> StreamingResponse:
    """以 SSE 只读传递工作流的审批、执行和撤销事件。"""

    if get_approval_request_by_workflow_id(session, str(workflow_id)) is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workflow_not_found",
                "message": "工作流不存在。",
            },
        )

    return StreamingResponse(
        _stream_workflow_events(
            session,
            workflow_id,
            after_event_id=last_event_id or 0,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/workflows/{workflow_id}/execute",
    response_model=SafeExecutionResponse,
)
def execute_organization_workflow(
    workflow_id: UUID,
    expected_plan_id: UUID | None = Query(default=None),
    expected_revision: int | None = Query(default=None, ge=0),
    session: Session = Depends(get_session),
    graph: CompiledStateGraph = Depends(get_workflow_graph),
) -> SafeExecutionResponse:
    """执行当前已批准工作流中的服务器端计划。"""

    _check_expected_operation_snapshot(
        session,
        workflow_id,
        expected_plan_id=expected_plan_id,
        expected_revision=expected_revision,
    )
    workflow = _ready_workflow_response(session, graph, workflow_id)
    request = SafeExecutionRequest(
        workflow_id=workflow_id,
        plan=workflow.operation_plan,
    )
    try:
        result = execute_safe_operation_plan(session, request)
    except OperationPlanApprovalError as error:
        raise _safe_execution_http_error(error) from error
    except SafeExecutionError as error:
        raise _safe_execution_http_error(error) from error
    except (
        FileEntryNotFoundError,
        OperationPlanExpiredError,
        OperationPlanSourceChangedError,
        OperationPlanSourceMismatchError,
        OperationPlanTargetConflictError,
        OperationPlanTargetUnavailableError,
        PathPolicyError,
        WorkspaceNotFoundError,
    ) as error:
        raise _operation_plan_http_error(error) from error

    return SafeExecutionResponse.from_result(result)


@router.post(
    "/workflows/{workflow_id}/undo",
    response_model=SafeExecutionResponse,
)
def undo_organization_workflow(
    workflow_id: UUID,
    expected_plan_id: UUID | None = Query(default=None),
    expected_revision: int | None = Query(default=None, ge=0),
    session: Session = Depends(get_session),
    graph: CompiledStateGraph = Depends(get_workflow_graph),
) -> SafeExecutionResponse:
    """撤销当前已批准工作流对应的安全执行历史。"""

    _check_expected_operation_snapshot(
        session,
        workflow_id,
        expected_plan_id=expected_plan_id,
        expected_revision=expected_revision,
    )
    _ready_workflow_response(session, graph, workflow_id)
    try:
        result = undo_safe_operation_execution(session, workflow_id)
    except SafeExecutionError as error:
        raise _safe_execution_http_error(error) from error

    return SafeExecutionResponse.from_result(result)


@router.post(
    "/workflows/{workflow_id}/decisions",
    response_model=OrganizationWorkflowResponse,
)
def decide_organization_workflow(
    workflow_id: UUID,
    request: OrganizationDecisionRequest,
    session: Session = Depends(get_session),
    graph: CompiledStateGraph = Depends(get_workflow_graph),
) -> OrganizationWorkflowResponse:
    """通过既有协调器批准、编辑、拒绝或取消页面当前所见计划。"""

    _check_expected_operation_snapshot(
        session,
        workflow_id,
        expected_revision=request.expected_revision,
    )
    try:
        if request.action == "edit":
            if request.changes is None:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "invalid_edit_request",
                        "message": "编辑计划必须提供 changes。",
                    },
                )
            apply_organization_plan_edit(
                session,
                graph,
                workflow_id,
                request.expected_plan_id,
                EditOrganizationPlanRequest(changes=request.changes),
            )
        else:
            apply_organization_decision(
                session,
                graph,
                workflow_id,
                request.expected_plan_id,
                request.action,
            )
    except OrganizationDecisionError as error:
        not_found = error.code == OrganizationDecisionErrorCode.NOT_FOUND
        raise HTTPException(
            status_code=404 if not_found else 409,
            detail={
                "code": str(error.code),
                "message": (
                    "工作流不存在。"
                    if not_found
                    else "审批决定与当前工作流状态冲突。"
                ),
            },
        ) from error
    except ApprovalTransitionError as error:
        not_found = error.code == ApprovalTransitionErrorCode.NOT_FOUND
        raise HTTPException(
            status_code=404 if not_found else 409,
            detail={
                "code": str(error.code),
                "message": (
                    "审批任务不存在。"
                    if not_found
                    else "审批任务状态已经变化。"
                ),
            },
        ) from error
    except (WorkflowCheckpointError, WorkflowTransitionError) as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": str(error.code),
                "message": "工作流状态已经变化。",
            },
        ) from error
    except (
        OperationPreviewPathUnavailableError,
        OperationPlanExpiredError,
        OperationPlanSourceChangedError,
        OperationPlanSourceMismatchError,
        OperationPlanTargetConflictError,
        OperationPlanTargetUnavailableError,
        PathPolicyError,
        WorkspacePolicyError,
    ) as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": (
                    error.code.value
                    if isinstance(error, WorkspacePolicyError)
                    else "organization_plan_unavailable"
                ),
                "message": (
                    str(error)
                    if isinstance(error, WorkspacePolicyError)
                    else "当前文件状态无法生成安全计划。"
                ),
            },
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "organization_workflow_plan_mismatch",
                "message": "编辑请求与当前工作流状态冲突。",
            },
        ) from error

    return _workflow_response(session, graph, workflow_id)
