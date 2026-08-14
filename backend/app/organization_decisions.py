"""协调审批业务状态与 workflow checkpoint 的人工决定。"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Callable, Literal
from uuid import UUID, uuid4

from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError
from sqlalchemy.orm import Session

from .operation_plan import OperationPlan
from .repositories import get_approval_request_by_workflow_id
from .organization_planning import (
    EditOrganizationPlanRequest,
    build_organization_plan,
    merge_edit_request,
)
from .services import (
    approve_operation_plan,
    edit_operation_plan,
    reject_operation_plan,
)
from .workflow import WorkflowEvent, WorkflowState
from .workflow_graph import (
    run_checkpointed_workflow_event,
    workflow_checkpoint_config,
)


OrganizationDecisionAction = Literal["approve", "reject"]


class OrganizationDecisionErrorCode(StrEnum):
    """人工决定无法安全应用时供 API 稳定映射的程序码。"""

    NOT_FOUND = "organization_workflow_not_found"
    INVALID_CHECKPOINT = "organization_workflow_checkpoint_invalid"
    STATE_CONFLICT = "organization_workflow_state_conflict"
    PLAN_MISMATCH = "organization_workflow_plan_mismatch"


class OrganizationDecisionError(RuntimeError):
    """拒绝缺失、损坏、过期或状态不一致的人工决定。"""

    def __init__(
        self,
        code: OrganizationDecisionErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class AppliedOrganizationDecision:
    """审批记录和 checkpoint 均已接受同一人工决定。"""

    approval_status: Literal["APPROVED", "REJECTED"]
    workflow: WorkflowState


@dataclass(frozen=True, slots=True)
class AppliedOrganizationPlanEdit:
    """替代计划与审批记录均处于等待人工确认状态。"""

    approval_status: Literal["WAITING_APPROVAL"]
    workflow: WorkflowState


def apply_organization_decision(
    session: Session,
    graph: CompiledStateGraph,
    workflow_id: UUID,
    expected_plan_id: UUID,
    action: OrganizationDecisionAction,
) -> AppliedOrganizationDecision:
    """安全应用批准或拒绝，并允许 checkpoint 失败后重试。"""

    workflow = _load_waiting_or_applied_workflow(
        graph,
        workflow_id,
        expected_plan_id,
        action,
    )
    approval = get_approval_request_by_workflow_id(session, str(workflow_id))
    if approval is None:
        raise OrganizationDecisionError(
            OrganizationDecisionErrorCode.NOT_FOUND,
            "审批任务不存在",
        )
    if approval.plan_id != str(expected_plan_id):
        raise OrganizationDecisionError(
            OrganizationDecisionErrorCode.PLAN_MISMATCH,
            "页面所见计划已经变化",
        )

    expected_status: Literal["APPROVED", "REJECTED"] = (
        "APPROVED" if action == "approve" else "REJECTED"
    )
    if approval.status == "WAITING_APPROVAL":
        transition = (
            approve_operation_plan
            if action == "approve"
            else reject_operation_plan
        )
        approval = transition(session, workflow_id, expected_plan_id)
    elif approval.status != expected_status:
        raise OrganizationDecisionError(
            OrganizationDecisionErrorCode.STATE_CONFLICT,
            "审批任务已经接受其他决定",
        )

    if _decision_already_checkpointed(workflow, action):
        return AppliedOrganizationDecision(
            approval_status=expected_status,
            workflow=workflow,
        )

    event = WorkflowEvent(
        workflow_id=workflow_id,
        sequence_no=workflow.revision + 1,
        kind=(
            "resume_requested"
            if action == "approve"
            else "workflow_failed"
        ),
        error_code="human_rejected" if action == "reject" else None,
    )
    updated_workflow = run_checkpointed_workflow_event(graph, event)
    if not _decision_already_checkpointed(updated_workflow, action):
        raise OrganizationDecisionError(
            OrganizationDecisionErrorCode.STATE_CONFLICT,
            "workflow checkpoint 未进入预期状态",
        )

    return AppliedOrganizationDecision(
        approval_status=expected_status,
        workflow=updated_workflow,
    )


def apply_organization_plan_edit(
    session: Session,
    graph: CompiledStateGraph,
    workflow_id: UUID,
    expected_plan_id: UUID,
    request: EditOrganizationPlanRequest,
    *,
    now: datetime | None = None,
    plan_id_factory: Callable[[], UUID] = uuid4,
) -> AppliedOrganizationPlanEdit:
    """重建替代计划，并让 checkpoint 与审批记录可安全重试地对齐。"""

    workflow = _load_workflow_checkpoint(graph, workflow_id)
    approval = get_approval_request_by_workflow_id(session, str(workflow_id))
    if approval is None:
        raise OrganizationDecisionError(
            OrganizationDecisionErrorCode.NOT_FOUND,
            "审批任务不存在",
        )
    if approval.status != "WAITING_APPROVAL":
        raise OrganizationDecisionError(
            OrganizationDecisionErrorCode.STATE_CONFLICT,
            "审批任务当前不接受编辑",
        )
    if (
        workflow.status != "waiting"
        or workflow.wait_reason_code != "human_approval_required"
    ):
        raise OrganizationDecisionError(
            OrganizationDecisionErrorCode.STATE_CONFLICT,
            "workflow 当前不接受编辑",
        )

    if workflow.operation_plan.plan_id == expected_plan_id:
        if approval.plan_id != str(expected_plan_id):
            raise OrganizationDecisionError(
                OrganizationDecisionErrorCode.PLAN_MISMATCH,
                "页面所见计划已经变化",
            )
        replacement_request = merge_edit_request(
            workflow.operation_plan,
            request,
        )
        replacement_plan = build_organization_plan(
            session,
            replacement_request,
            now=now,
            plan_id_factory=plan_id_factory,
        )
        _checkpoint_replacement(
            graph,
            workflow,
            replacement_plan,
        )
        replacement_plan_id = replacement_plan.plan_id
    else:
        # graph 已经写入替代计划但审批提交失败时，沿用 checkpoint 中的完整计划。
        if approval.plan_id != str(expected_plan_id):
            raise OrganizationDecisionError(
                OrganizationDecisionErrorCode.PLAN_MISMATCH,
                "页面所见计划已经变化",
            )
        _ensure_staged_replacement_matches_request(
            workflow,
            request,
        )
        replacement_plan_id = workflow.operation_plan.plan_id

    edit_operation_plan(
        session,
        workflow_id,
        expected_plan_id,
        replacement_plan_id,
    )

    return AppliedOrganizationPlanEdit(
        approval_status="WAITING_APPROVAL",
        workflow=(
            workflow
            if workflow.operation_plan.plan_id != expected_plan_id
            else _load_workflow_checkpoint(graph, workflow_id)
        ),
    )


def _checkpoint_replacement(
    graph: CompiledStateGraph,
    workflow: WorkflowState,
    replacement_plan: OperationPlan,
) -> None:
    updated_workflow = run_checkpointed_workflow_event(
        graph,
        WorkflowEvent(
            workflow_id=workflow.workflow_id,
            sequence_no=workflow.revision + 1,
            kind="plan_replaced",
            replacement_plan=replacement_plan,
        ),
    )
    if updated_workflow.operation_plan.plan_id != replacement_plan.plan_id:
        raise OrganizationDecisionError(
            OrganizationDecisionErrorCode.STATE_CONFLICT,
            "workflow checkpoint 未保存替代计划",
        )


def _ensure_staged_replacement_matches_request(
    workflow: WorkflowState,
    request: EditOrganizationPlanRequest,
) -> None:
    merged_request = merge_edit_request(workflow.operation_plan, request)
    requested_targets = {
        selection.source_file_id: selection.target_directory
        for selection in merged_request.selections
    }
    for operation in workflow.operation_plan.operations:
        current_target = operation.target_relative_path.rsplit(
            "/",
            1,
        )[0]
        if requested_targets[operation.source_file_id] != current_target:
            raise OrganizationDecisionError(
                OrganizationDecisionErrorCode.PLAN_MISMATCH,
                "checkpoint 中的替代计划与当前编辑请求不一致",
            )


def _load_waiting_or_applied_workflow(
    graph: CompiledStateGraph,
    workflow_id: UUID,
    expected_plan_id: UUID,
    action: OrganizationDecisionAction,
) -> WorkflowState:
    workflow = _load_workflow_checkpoint(graph, workflow_id)

    if workflow.workflow_id != workflow_id:
        raise OrganizationDecisionError(
            OrganizationDecisionErrorCode.STATE_CONFLICT,
            "workflow_id 与 checkpoint 不一致",
        )
    if workflow.operation_plan.plan_id != expected_plan_id:
        raise OrganizationDecisionError(
            OrganizationDecisionErrorCode.PLAN_MISMATCH,
            "页面所见计划已经变化",
        )
    if _decision_already_checkpointed(workflow, action):
        return workflow
    if (
        workflow.status != "waiting"
        or workflow.wait_reason_code != "human_approval_required"
    ):
        raise OrganizationDecisionError(
            OrganizationDecisionErrorCode.STATE_CONFLICT,
            "workflow 当前不接受人工决定",
        )
    return workflow


def _load_workflow_checkpoint(
    graph: CompiledStateGraph,
    workflow_id: UUID,
) -> WorkflowState:
    raw_workflow = graph.get_state(
        workflow_checkpoint_config(workflow_id)
    ).values.get("workflow")
    if raw_workflow is None:
        raise OrganizationDecisionError(
            OrganizationDecisionErrorCode.NOT_FOUND,
            "workflow checkpoint 不存在",
        )
    try:
        workflow = WorkflowState.model_validate(raw_workflow)
    except ValidationError as error:
        raise OrganizationDecisionError(
            OrganizationDecisionErrorCode.INVALID_CHECKPOINT,
            "workflow checkpoint 无法通过业务状态校验",
        ) from error
    if workflow.workflow_id != workflow_id:
        raise OrganizationDecisionError(
            OrganizationDecisionErrorCode.STATE_CONFLICT,
            "workflow_id 与 checkpoint 不一致",
        )
    return workflow


def _decision_already_checkpointed(
    workflow: WorkflowState,
    action: OrganizationDecisionAction,
) -> bool:
    if action == "approve":
        return workflow.status == "ready" and workflow.wait_reason_code is None
    return workflow.status == "failed" and workflow.error_code == "human_rejected"
