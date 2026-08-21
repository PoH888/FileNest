"""最小界面使用的整理计划 HTTP 边界。"""

from collections.abc import Iterator
from time import sleep
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy.orm import Session

from .database import get_session
from .events import build_workflow_event_stream
from .operation_plan import OperationPlan
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
    find_approval_audit_events,
    find_operation_execution_items,
    get_approval_request_by_workflow_id,
    get_operation_execution_by_workflow_id,
    get_operation_projection_by_workflow_id,
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
    WorkspaceNotFoundError,
    get_operation_plan,
)
from .operation_status import (
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
ApprovalStatus = Literal[
    "WAITING_APPROVAL",
    "APPROVED",
    "REJECTED",
    "CANCELLED",
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


class OrganizationDecisionRequest(BaseModel):
    """页面针对当前所见计划提交的最小人工决定。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["approve", "edit", "reject", "cancel"]
    expected_plan_id: UUID
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
        or operation.overall_status != expected_approval_status
        or operation.overall_status != expected_workflow_status
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
    ) as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "organization_plan_unavailable",
                "message": "当前文件状态无法生成安全计划。",
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
    "/workflows/{workflow_id}",
    response_model=OrganizationWorkflowResponse,
)
def get_organization_workflow(
    workflow_id: UUID,
    session: Session = Depends(get_session),
    graph: CompiledStateGraph = Depends(get_workflow_graph),
) -> OrganizationWorkflowResponse:
    """读取 checkpoint 与审批记录一致的计划快照。"""

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
    session: Session = Depends(get_session),
    graph: CompiledStateGraph = Depends(get_workflow_graph),
) -> SafeExecutionResponse:
    """执行当前已批准工作流中的服务器端计划。"""

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
    session: Session = Depends(get_session),
    graph: CompiledStateGraph = Depends(get_workflow_graph),
) -> SafeExecutionResponse:
    """撤销当前已批准工作流对应的安全执行历史。"""

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
    ) as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "organization_plan_unavailable",
                "message": "当前文件状态无法生成安全计划。",
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
