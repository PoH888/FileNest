"""最小界面使用的整理计划 HTTP 边界。"""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy.orm import Session

from .database import get_session
from .operation_plan import OperationPlan
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
from .repositories import get_approval_request_by_workflow_id
from .services import (
    ApprovalTransitionError,
    ApprovalTransitionErrorCode,
    FileEntryNotFoundError,
    OperationPlanExpiredError,
    OperationPlanSourceChangedError,
    OperationPlanSourceMismatchError,
    OperationPlanTargetConflictError,
    OperationPlanTargetUnavailableError,
    OperationPreviewPathUnavailableError,
    WorkspaceNotFoundError,
)
from .workflow import WorkflowState, WorkflowStatus, WorkflowTransitionError
from .workflow_graph import (
    WorkflowCheckpointError,
    workflow_checkpoint_config,
)
from .workflow_runtime import get_workflow_graph


router = APIRouter(prefix="/api/v1")
ApprovalStatus = Literal["WAITING_APPROVAL", "APPROVED", "REJECTED"]


class OrganizationWorkflowResponse(BaseModel):
    """供页面展示的工作流、不可变计划和审批状态。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_id: UUID
    status: WorkflowStatus
    revision: int
    wait_reason_code: str | None
    operation_plan: OperationPlan
    approval_status: ApprovalStatus


class OrganizationDecisionRequest(BaseModel):
    """页面针对当前所见计划提交的最小人工决定。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["approve", "edit", "reject"]
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

    if (
        workflow.workflow_id != workflow_id
        or str(workflow.operation_plan.plan_id) != approval.plan_id
        or approval.status not in {
            "WAITING_APPROVAL",
            "APPROVED",
            "REJECTED",
        }
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
        operation_plan=workflow.operation_plan,
        approval_status=approval.status,
    )


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
    """通过既有协调器批准、编辑或拒绝页面当前所见计划。"""

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
