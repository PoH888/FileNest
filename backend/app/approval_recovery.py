"""重启后重新绑定审批业务状态与 LangGraph checkpoint。"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError
from sqlalchemy.orm import Session

from .repositories import find_waiting_approval_requests
from .filesystem_adapter import FileSystemAdapter
from .models import ApprovalRequest, Workspace
from .path_policy import PathPolicyError, validate_workspace_root
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
    WORKSPACE_NOT_FOUND = "approval_workspace_not_found"
    POLICY_NOT_ALLOWED = "approval_workspace_policy_denied"
    RECOVERY_UNAVAILABLE = "approval_recovery_unavailable"


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


@dataclass(frozen=True, slots=True)
class ApprovalRecoveryIssue:
    """单个待审批任务的稳定恢复阻断记录。"""

    approval_id: int | None
    workflow_id: str | None
    plan_id: str | None
    workspace_id: int | None
    code: str


@dataclass(frozen=True, slots=True)
class ApprovalRecoveryScan:
    """启动扫描得到的有效快照和阻断项集合。"""

    recovered_tasks: tuple[RecoveredApprovalTask, ...]
    issues: tuple[ApprovalRecoveryIssue, ...]


def recover_waiting_approval_tasks(
    session: Session,
    graph: CompiledStateGraph,
) -> list[RecoveredApprovalTask]:
    """恢复所有业务上仍等待审批且 checkpoint 完全匹配的任务。"""

    return [
        _recover_approval(approval, graph)
        for approval in find_waiting_approval_requests(session)
    ]


def scan_waiting_approval_tasks(
    session: Session,
    graph: CompiledStateGraph,
) -> ApprovalRecoveryScan:
    """逐项扫描待审批任务，保留有效快照并记录所有安全阻断。"""

    recovered_tasks: list[RecoveredApprovalTask] = []
    issues: list[ApprovalRecoveryIssue] = []
    for approval in find_waiting_approval_requests(session):
        try:
            task = _recover_approval(approval, graph)
            _validate_workspace_policy(session, task)
        except ApprovalRecoveryError as error:
            issues.append(
                ApprovalRecoveryIssue(
                    approval_id=approval.id,
                    workflow_id=approval.workflow_id,
                    plan_id=approval.plan_id,
                    workspace_id=_workflow_workspace_id(approval, graph),
                    code=error.code.value,
                )
            )
        else:
            recovered_tasks.append(task)

    return ApprovalRecoveryScan(
        recovered_tasks=tuple(recovered_tasks),
        issues=tuple(issues),
    )


def _recover_approval(
    approval: ApprovalRequest,
    graph: CompiledStateGraph,
) -> RecoveredApprovalTask:
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

    return RecoveredApprovalTask(
        approval_id=approval.id,
        workflow_id=workflow_id,
        plan_id=plan_id,
        workflow=workflow,
    )


def _validate_workspace_policy(
    session: Session,
    task: RecoveredApprovalTask,
) -> None:
    workspace_id = task.workflow.operation_plan.workspace_id
    workspace = session.get(Workspace, workspace_id)
    if workspace is None:
        raise ApprovalRecoveryError(
            ApprovalRecoveryErrorCode.WORKSPACE_NOT_FOUND,
            "待审批任务所属工作区不存在",
        )
    try:
        workspace_root = validate_workspace_root(workspace.root_path)
        adapter = FileSystemAdapter(workspace_root)
        for operation in task.workflow.operation_plan.operations:
            adapter.authorized_path(Path(operation.source_relative_path))
    except (OSError, PathPolicyError) as error:
        raise ApprovalRecoveryError(
            ApprovalRecoveryErrorCode.POLICY_NOT_ALLOWED,
            "待审批任务来源路径不再通过 Workspace Policy",
        ) from error


def _workflow_workspace_id(
    approval: ApprovalRequest,
    graph: CompiledStateGraph,
) -> int | None:
    """错误记录中尽量保留已校验 checkpoint 的 workspace。"""

    try:
        workflow_id = UUID(approval.workflow_id)
        raw_workflow = graph.get_state(
            workflow_checkpoint_config(workflow_id)
        ).values.get("workflow")
        if not isinstance(raw_workflow, dict):
            return None
        workflow = WorkflowState.model_validate(raw_workflow)
    except (TypeError, ValueError, ValidationError):
        return None
    return workflow.operation_plan.workspace_id
