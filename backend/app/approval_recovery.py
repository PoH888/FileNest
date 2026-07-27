"""重启后重新绑定审批业务状态与 LangGraph checkpoint。"""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError
from sqlalchemy.orm import Session

from .repositories import find_waiting_approval_requests
from .workflow import WorkflowState
from .workflow_graph import workflow_checkpoint_config


class ApprovalRecoveryErrorCode(StrEnum):
    """待审批任务无法安全恢复时供程序稳定判断的错误码。"""

    INVALID_BUSINESS_STATE = "approval_business_state_invalid"
    CHECKPOINT_NOT_FOUND = "approval_checkpoint_not_found"
    INVALID_CHECKPOINT = "approval_checkpoint_invalid"
    WORKFLOW_MISMATCH = "approval_workflow_mismatch"
    PLAN_MISMATCH = "approval_checkpoint_plan_mismatch"
    NOT_WAITING_FOR_APPROVAL = "checkpoint_not_waiting_for_approval"


class ApprovalRecoveryError(RuntimeError):
    """拒绝恢复缺失、损坏或与审批业务状态不一致的 checkpoint。"""

    def __init__(
        self,
        code: ApprovalRecoveryErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class RecoveredApprovalTask:
    """离开数据库和 checkpoint 边界后的可恢复审批快照。"""

    approval_id: int
    workflow_id: UUID
    plan_id: UUID
    workflow: WorkflowState


def recover_waiting_approval_tasks(
    session: Session,
    graph: CompiledStateGraph,
) -> list[RecoveredApprovalTask]:
    """恢复所有业务上仍等待审批且 checkpoint 完全匹配的任务。"""

    recovered_tasks: list[RecoveredApprovalTask] = []
    for approval in find_waiting_approval_requests(session):
        try:
            workflow_id = UUID(approval.workflow_id)
            plan_id = UUID(approval.plan_id)
        except ValueError as error:
            raise ApprovalRecoveryError(
                ApprovalRecoveryErrorCode.INVALID_BUSINESS_STATE,
                "审批业务记录包含无效标识",
            ) from error

        saved_values = graph.get_state(
            workflow_checkpoint_config(workflow_id)
        ).values
        raw_workflow = saved_values.get("workflow")
        if raw_workflow is None:
            raise ApprovalRecoveryError(
                ApprovalRecoveryErrorCode.CHECKPOINT_NOT_FOUND,
                "待审批任务缺少 workflow checkpoint",
            )

        try:
            workflow = WorkflowState.model_validate(raw_workflow)
        except ValidationError as error:
            raise ApprovalRecoveryError(
                ApprovalRecoveryErrorCode.INVALID_CHECKPOINT,
                "workflow checkpoint 无法通过业务状态校验",
            ) from error

        if workflow.workflow_id != workflow_id:
            raise ApprovalRecoveryError(
                ApprovalRecoveryErrorCode.WORKFLOW_MISMATCH,
                "审批业务记录与 checkpoint 的 workflow_id 不一致",
            )
        if workflow.operation_plan.plan_id != plan_id:
            raise ApprovalRecoveryError(
                ApprovalRecoveryErrorCode.PLAN_MISMATCH,
                "审批业务记录与 checkpoint 的 plan_id 不一致",
            )
        if (
            workflow.status != "waiting"
            or workflow.wait_reason_code != "human_approval_required"
        ):
            raise ApprovalRecoveryError(
                ApprovalRecoveryErrorCode.NOT_WAITING_FOR_APPROVAL,
                "checkpoint 当前并非等待人工审批状态",
            )

        recovered_tasks.append(
            RecoveredApprovalTask(
                approval_id=approval.id,
                workflow_id=workflow_id,
                plan_id=plan_id,
                workflow=workflow,
            )
        )

    return recovered_tasks
