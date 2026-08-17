"""暂停与恢复工作流的框架无关状态契约。"""

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .operation_plan import OperationPlan


WorkflowStatus = Literal["ready", "waiting", "completed", "failed", "cancelled"]
WorkflowEventKind = Literal[
    "pause_requested",
    "resume_requested",
    "plan_replaced",
    "workflow_completed",
    "workflow_failed",
    "workflow_cancelled",
]


class WorkflowTransitionErrorCode(StrEnum):
    """工作流转换失败时供程序稳定判断的错误码。"""

    WORKFLOW_MISMATCH = "workflow_mismatch"
    EVENT_SEQUENCE_MISMATCH = "event_sequence_mismatch"
    INVALID_TRANSITION = "invalid_transition"


class WorkflowTransitionError(ValueError):
    """拒绝不属于当前工作流或不符合状态机规则的事件。"""

    def __init__(
        self,
        code: WorkflowTransitionErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


class WorkflowState(BaseModel):
    """可进入 checkpoint 的完整状态，不包含 Session 或文件系统对象。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    workflow_id: UUID
    operation_plan: OperationPlan
    status: WorkflowStatus = "ready"
    revision: int = Field(default=0, ge=0)
    wait_reason_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )
    error_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )
    @model_validator(mode="after")
    def validate_status_details(self) -> "WorkflowState":
        if self.status == "waiting" and self.wait_reason_code is None:
            raise ValueError("waiting workflow must contain a wait reason code")
        if self.status != "waiting" and self.wait_reason_code is not None:
            raise ValueError("non-waiting workflow must not contain a wait reason code")
        if self.status == "failed" and self.error_code is None:
            raise ValueError("failed workflow must contain an error code")
        if self.status != "failed" and self.error_code is not None:
            raise ValueError("non-failed workflow must not contain an error code")
        return self


class WorkflowEvent(BaseModel):
    """驱动一次纯状态转换的事件；权限判断由上层业务边界负责。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    workflow_id: UUID
    sequence_no: int = Field(ge=1)
    kind: WorkflowEventKind
    reason_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )
    error_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )
    replacement_plan: OperationPlan | None = None

    @model_validator(mode="after")
    def validate_event_details(self) -> "WorkflowEvent":
        if self.kind == "pause_requested" and self.reason_code is None:
            raise ValueError("pause event must contain a reason code")
        if self.kind != "pause_requested" and self.reason_code is not None:
            raise ValueError("non-pause event must not contain a reason code")
        if self.kind == "workflow_failed" and self.error_code is None:
            raise ValueError("failed event must contain an error code")
        if self.kind != "workflow_failed" and self.error_code is not None:
            raise ValueError("non-failed event must not contain an error code")
        if self.kind == "plan_replaced" and self.replacement_plan is None:
            raise ValueError("plan replacement event requires a replacement plan")
        if self.kind != "plan_replaced" and self.replacement_plan is not None:
            raise ValueError("non-replacement event must not contain a plan")
        return self


def transition_workflow(
    state: WorkflowState,
    event: WorkflowEvent,
) -> WorkflowState:
    """应用一个有序事件并返回新状态，不执行持久化或文件操作。"""

    if event.workflow_id != state.workflow_id:
        raise WorkflowTransitionError(
            WorkflowTransitionErrorCode.WORKFLOW_MISMATCH,
            "工作流事件不属于当前工作流",
        )
    if event.sequence_no != state.revision + 1:
        raise WorkflowTransitionError(
            WorkflowTransitionErrorCode.EVENT_SEQUENCE_MISMATCH,
            "工作流事件序号不连续",
        )

    next_status, wait_reason_code, error_code = _next_status(state, event)
    operation_plan = state.operation_plan
    if event.kind == "plan_replaced":
        replacement_plan = event.replacement_plan
        if replacement_plan is None:
            raise WorkflowTransitionError(
                WorkflowTransitionErrorCode.INVALID_TRANSITION,
                "计划替换事件缺少替代计划",
            )
        _validate_replacement_plan(state.operation_plan, replacement_plan)
        operation_plan = replacement_plan
    return WorkflowState(
        workflow_id=state.workflow_id,
        operation_plan=operation_plan,
        status=next_status,
        revision=event.sequence_no,
        wait_reason_code=wait_reason_code,
        error_code=error_code,
    )


def _next_status(
    state: WorkflowState,
    event: WorkflowEvent,
) -> tuple[WorkflowStatus, str | None, str | None]:
    if state.status == "ready" and event.kind == "pause_requested":
        return "waiting", event.reason_code, None
    if state.status == "waiting" and event.kind == "resume_requested":
        return "ready", None, None
    if state.status == "waiting" and event.kind == "plan_replaced":
        return "waiting", state.wait_reason_code, None
    if state.status == "ready" and event.kind == "workflow_completed":
        return "completed", None, None
    if state.status in {"ready", "waiting"} and event.kind == "workflow_failed":
        return "failed", None, event.error_code
    if state.status == "waiting" and event.kind == "workflow_cancelled":
        return "cancelled", None, None

    raise WorkflowTransitionError(
        WorkflowTransitionErrorCode.INVALID_TRANSITION,
        f"状态 {state.status} 不接受事件 {event.kind}",
    )


def _validate_replacement_plan(
    current_plan: OperationPlan,
    replacement_plan: OperationPlan,
) -> None:
    current_sources = {
        (operation.source_file_id, operation.source_relative_path)
        for operation in current_plan.operations
    }
    replacement_sources = {
        (operation.source_file_id, operation.source_relative_path)
        for operation in replacement_plan.operations
    }
    if (
        replacement_plan.plan_id == current_plan.plan_id
        or replacement_plan.workspace_id != current_plan.workspace_id
        or replacement_sources != current_sources
    ):
        raise WorkflowTransitionError(
            WorkflowTransitionErrorCode.INVALID_TRANSITION,
            "替代计划必须保留工作区和源文件，并使用新的 plan_id",
        )
