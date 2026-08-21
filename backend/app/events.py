"""供实时界面消费的业务事件与 SSE 文本协议。"""

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Literal, TypeAlias
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .agent_loop import AGENT_RUN_ACTIVE_STATUSES, AgentRunLifecycleStatus
from .agent_observability import RecordedToolStatus
from .models import (
    AgentRun,
    AgentToolCall,
    ApprovalAuditEvent,
    ApprovalRequest,
    OperationExecution,
    OperationExecutionItem,
)
from .repositories import ApprovalAction, ApprovalStatus
from .workflow import WorkflowStatus


AgentRunEventStatus: TypeAlias = AgentRunLifecycleStatus
AgentToolCallEventStatus: TypeAlias = Literal["requested"] | RecordedToolStatus


class _BusinessEvent(BaseModel):
    """所有业务事件共享的稳定、只读传输字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def normalize_occurred_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("business event timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)


class AgentRunStatusChangedEvent(_BusinessEvent):
    """来自现有 AgentRun 生命周期记录的状态事件。"""

    kind: Literal["agent_run.status_changed"] = "agent_run.status_changed"
    run_id: int = Field(ge=1)
    status: AgentRunEventStatus
    model_turns: int = Field(default=0, ge=0)
    error_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )


class AgentToolCallStatusChangedEvent(_BusinessEvent):
    """来自现有 AgentToolCall 记录的运行时间线事件。"""

    kind: Literal["agent_tool_call.status_changed"] = (
        "agent_tool_call.status_changed"
    )
    run_id: int = Field(ge=1)
    tool_call_id: int = Field(ge=1)
    sequence_no: int = Field(ge=1)
    tool_name: str = Field(min_length=1, max_length=100)
    status: AgentToolCallEventStatus
    error_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )


class WorkflowStatusChangedEvent(_BusinessEvent):
    """来自现有 WorkflowState 的状态和版本事件。"""

    kind: Literal["workflow.status_changed"] = "workflow.status_changed"
    workflow_id: UUID
    status: WorkflowStatus
    revision: int = Field(ge=0)


class ApprovalStatusChangedEvent(_BusinessEvent):
    """来自现有审批审计记录的状态转换事件。"""

    kind: Literal["approval.status_changed"] = "approval.status_changed"
    audit_event_id: int = Field(ge=1)
    approval_request_id: int = Field(ge=1)
    workflow_id: UUID
    action: ApprovalAction
    previous_status: ApprovalStatus
    next_status: ApprovalStatus


class AgentStartedEvent(_BusinessEvent):
    """Agent Run 已进入可执行生命周期。"""

    kind: Literal["agent.started"] = "agent.started"
    run_id: int = Field(ge=1)


class AgentResumedEvent(_BusinessEvent):
    """Agent Run 从可恢复状态重新进入执行。"""

    kind: Literal["agent.resumed"] = "agent.resumed"
    run_id: int = Field(ge=1)


class AgentStepStartedEvent(_BusinessEvent):
    """Agent 的一个已记录工具步骤开始执行。"""

    kind: Literal["agent.step.started"] = "agent.step.started"
    run_id: int = Field(ge=1)
    step_id: int = Field(ge=1)
    step_index: int = Field(ge=1)
    step_type: str = Field(min_length=1, max_length=100)


class AgentStepCompletedEvent(_BusinessEvent):
    """Agent 的一个已记录工具步骤完成。"""

    kind: Literal["agent.step.completed"] = "agent.step.completed"
    run_id: int = Field(ge=1)
    step_id: int = Field(ge=1)
    step_index: int = Field(ge=1)
    step_type: str = Field(min_length=1, max_length=100)
    status: AgentToolCallEventStatus
    error_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )


class AgentCancelledEvent(_BusinessEvent):
    """Agent Run 被取消。"""

    kind: Literal["agent.cancelled"] = "agent.cancelled"
    run_id: int = Field(ge=1)


class AgentErrorEvent(_BusinessEvent):
    """Agent Run 以可公开的错误结束。"""

    kind: Literal["agent.error"] = "agent.error"
    run_id: int = Field(ge=1)
    error_code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")


class ApprovalWaitingEvent(_BusinessEvent):
    """一个审批计划进入等待人工决定状态。"""

    kind: Literal["approval.waiting"] = "approval.waiting"
    workflow_id: UUID
    approval_request_id: int = Field(ge=1)
    plan_id: UUID


class ApprovalApprovedEvent(_BusinessEvent):
    """审批计划已批准。"""

    kind: Literal["approval.approved"] = "approval.approved"
    workflow_id: UUID
    approval_request_id: int = Field(ge=1)
    plan_id: UUID
    action: Literal["approve"] = "approve"


class ApprovalRejectedEvent(_BusinessEvent):
    """审批计划被拒绝或取消。"""

    kind: Literal["approval.rejected"] = "approval.rejected"
    workflow_id: UUID
    approval_request_id: int = Field(ge=1)
    plan_id: UUID
    action: Literal["reject", "cancel"]


ExecutionStatus: TypeAlias = Literal[
    "EXECUTING",
    "PARTIALLY_COMPLETED",
    "COMPLETED",
    "FAILED",
]


class ExecutionStartedEvent(_BusinessEvent):
    """安全执行主记录已建立。"""

    kind: Literal["execution.started"] = "execution.started"
    workflow_id: UUID
    execution_id: int = Field(ge=1)
    plan_id: UUID


class ExecutionItemCompletedEvent(_BusinessEvent):
    """一个安全执行明细完成。"""

    kind: Literal["execution.item.completed"] = "execution.item.completed"
    workflow_id: UUID
    execution_id: int = Field(ge=1)
    sequence_no: int = Field(ge=1)
    source_file_id: int = Field(ge=1)


class ExecutionItemFailedEvent(_BusinessEvent):
    """一个安全执行明细失败。"""

    kind: Literal["execution.item.failed"] = "execution.item.failed"
    workflow_id: UUID
    execution_id: int = Field(ge=1)
    sequence_no: int = Field(ge=1)
    source_file_id: int = Field(ge=1)
    error_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,99}$",
    )


class ExecutionCompletedEvent(_BusinessEvent):
    """安全执行进入汇总终态。"""

    kind: Literal["execution.completed"] = "execution.completed"
    workflow_id: UUID
    execution_id: int = Field(ge=1)
    plan_id: UUID
    status: ExecutionStatus


class UndoStartedEvent(_BusinessEvent):
    """安全执行历史进入撤销阶段。"""

    kind: Literal["undo.started"] = "undo.started"
    workflow_id: UUID
    execution_id: int = Field(ge=1)
    plan_id: UUID


class UndoItemCompletedEvent(_BusinessEvent):
    """一个执行明细完成撤销。"""

    kind: Literal["undo.item.completed"] = "undo.item.completed"
    workflow_id: UUID
    execution_id: int = Field(ge=1)
    sequence_no: int = Field(ge=1)
    source_file_id: int = Field(ge=1)


class UndoCompletedEvent(_BusinessEvent):
    """整条安全执行历史完成撤销。"""

    kind: Literal["undo.completed"] = "undo.completed"
    workflow_id: UUID
    execution_id: int = Field(ge=1)
    plan_id: UUID


BusinessEvent: TypeAlias = (
    AgentRunStatusChangedEvent
    | AgentToolCallStatusChangedEvent
    | WorkflowStatusChangedEvent
    | ApprovalStatusChangedEvent
    | AgentStartedEvent
    | AgentResumedEvent
    | AgentStepStartedEvent
    | AgentStepCompletedEvent
    | AgentCancelledEvent
    | AgentErrorEvent
    | ApprovalWaitingEvent
    | ApprovalApprovedEvent
    | ApprovalRejectedEvent
    | ExecutionStartedEvent
    | ExecutionItemCompletedEvent
    | ExecutionItemFailedEvent
    | ExecutionCompletedEvent
    | UndoStartedEvent
    | UndoItemCompletedEvent
    | UndoCompletedEvent
)


class SseEvent(BaseModel):
    """一个带有服务端序号的 SSE 事件。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: int = Field(ge=1)
    data: BusinessEvent

    def encode(self) -> str:
        return encode_sse_event(self.data, event_id=self.event_id)


def build_agent_run_event_stream(
    agent_run: AgentRun,
    tool_calls: Sequence[AgentToolCall],
    *,
    resumed: bool = False,
) -> tuple[SseEvent, ...]:
    """把已持久化的 Agent Run 事实按时间线投影为带序号的 SSE 事件。"""

    events: list[SseEvent] = []

    def append(data: BusinessEvent) -> None:
        events.append(SseEvent(event_id=len(events) + 1, data=data))

    initial_status = (
        agent_run.status
        if agent_run.status in AGENT_RUN_ACTIVE_STATUSES
        else "running"
    )
    append(
        AgentRunStatusChangedEvent(
            occurred_at=_storage_timestamp(agent_run.started_at),
            run_id=agent_run.id,
            status=initial_status,
        )
    )
    append(
        AgentResumedEvent(
            occurred_at=_storage_timestamp(agent_run.started_at),
            run_id=agent_run.id,
        )
        if resumed
        else AgentStartedEvent(
            occurred_at=_storage_timestamp(agent_run.started_at),
            run_id=agent_run.id,
        )
    )
    for tool_call in sorted(tool_calls, key=lambda item: item.sequence_no):
        append(
            AgentToolCallStatusChangedEvent(
                occurred_at=_storage_timestamp(tool_call.started_at),
                run_id=agent_run.id,
                tool_call_id=tool_call.id,
                sequence_no=tool_call.sequence_no,
                tool_name=tool_call.tool_name,
                status="requested",
            )
        )
        append(
            AgentStepStartedEvent(
                occurred_at=_storage_timestamp(tool_call.started_at),
                run_id=agent_run.id,
                step_id=tool_call.id,
                step_index=tool_call.sequence_no,
                step_type=tool_call.tool_name,
            )
        )
        if tool_call.finished_at is not None:
            append(
                AgentToolCallStatusChangedEvent(
                    occurred_at=_storage_timestamp(tool_call.finished_at),
                    run_id=agent_run.id,
                    tool_call_id=tool_call.id,
                    sequence_no=tool_call.sequence_no,
                    tool_name=tool_call.tool_name,
                    status=tool_call.status,
                    error_code=tool_call.error_code,
                )
            )
            append(
                AgentStepCompletedEvent(
                    occurred_at=_storage_timestamp(tool_call.finished_at),
                    run_id=agent_run.id,
                    step_id=tool_call.id,
                    step_index=tool_call.sequence_no,
                    step_type=tool_call.tool_name,
                    status=tool_call.status,
                    error_code=tool_call.error_code,
                )
            )

    if agent_run.status not in AGENT_RUN_ACTIVE_STATUSES:
        if agent_run.status == "cancelled":
            append(
                AgentCancelledEvent(
                    occurred_at=_storage_timestamp(
                        agent_run.finished_at or agent_run.started_at
                    ),
                    run_id=agent_run.id,
                )
            )
        elif agent_run.status in {"failed", "timed_out"}:
            append(
                AgentErrorEvent(
                    occurred_at=_storage_timestamp(
                        agent_run.finished_at or agent_run.started_at
                    ),
                    run_id=agent_run.id,
                    error_code=agent_run.error_code
                    or (
                        "agent_timed_out"
                        if agent_run.status == "timed_out"
                        else "agent_failed"
                    ),
                )
            )
        append(
            AgentRunStatusChangedEvent(
                occurred_at=_storage_timestamp(
                    agent_run.finished_at or agent_run.started_at
                ),
                run_id=agent_run.id,
                status=agent_run.status,
                model_turns=agent_run.model_turns,
                error_code=agent_run.error_code,
            )
        )

    return tuple(events)


def build_workflow_event_stream(
    approval: ApprovalRequest,
    approval_audits: Sequence[ApprovalAuditEvent],
    execution: OperationExecution | None,
    execution_items: Sequence[OperationExecutionItem],
) -> tuple[SseEvent, ...]:
    """把审批、执行和撤销的持久化事实投影为可恢复 SSE 事件。"""

    events: list[SseEvent] = []
    workflow_id = UUID(approval.workflow_id)

    def append(data: BusinessEvent) -> None:
        events.append(SseEvent(event_id=len(events) + 1, data=data))

    ordered_audits = sorted(approval_audits, key=lambda item: item.id)
    initial_plan_id = UUID(
        ordered_audits[0].previous_plan_id
        if ordered_audits
        else approval.plan_id
    )
    append(
        ApprovalWaitingEvent(
            occurred_at=_storage_timestamp(approval.created_at),
            workflow_id=workflow_id,
            approval_request_id=approval.id,
            plan_id=initial_plan_id,
        )
    )
    for audit in ordered_audits:
        occurred_at = _storage_timestamp(audit.recorded_at)
        plan_id = UUID(audit.next_plan_id)
        if audit.action == "edit":
            append(
                ApprovalWaitingEvent(
                    occurred_at=occurred_at,
                    workflow_id=workflow_id,
                    approval_request_id=approval.id,
                    plan_id=plan_id,
                )
            )
        elif audit.action == "approve":
            append(
                ApprovalApprovedEvent(
                    occurred_at=occurred_at,
                    workflow_id=workflow_id,
                    approval_request_id=approval.id,
                    plan_id=plan_id,
                )
            )
        elif audit.action in {"reject", "cancel"}:
            append(
                ApprovalRejectedEvent(
                    occurred_at=occurred_at,
                    workflow_id=workflow_id,
                    approval_request_id=approval.id,
                    plan_id=plan_id,
                    action=audit.action,
                )
            )

    if execution is None:
        return tuple(events)

    execution_id = execution.id
    execution_plan_id = UUID(execution.plan_id)
    append(
        ExecutionStartedEvent(
            occurred_at=_storage_timestamp(execution.started_at),
            workflow_id=workflow_id,
            execution_id=execution_id,
            plan_id=execution_plan_id,
        )
    )
    for item in sorted(execution_items, key=lambda value: value.sequence_no):
        if item.status in {"COMPLETED", "UNDOING", "UNDONE"}:
            append(
                ExecutionItemCompletedEvent(
                    occurred_at=_storage_timestamp(
                        item.completed_at
                        or item.undone_at
                        or item.recorded_at
                    ),
                    workflow_id=workflow_id,
                    execution_id=execution_id,
                    sequence_no=item.sequence_no,
                    source_file_id=item.source_file_id,
                )
            )
        elif item.status == "FAILED":
            append(
                ExecutionItemFailedEvent(
                    occurred_at=_storage_timestamp(
                        item.failed_at or item.recorded_at
                    ),
                    workflow_id=workflow_id,
                    execution_id=execution_id,
                    sequence_no=item.sequence_no,
                    source_file_id=item.source_file_id,
                    error_code=item.error_code,
                )
            )

    if execution.status in {
        "COMPLETED",
        "PARTIALLY_COMPLETED",
        "FAILED",
        "UNDOING",
        "UNDONE",
    }:
        execution_status: ExecutionStatus = (
            "COMPLETED"
            if execution.status in {"UNDOING", "UNDONE"}
            else execution.status
        )
        append(
            ExecutionCompletedEvent(
                occurred_at=_storage_timestamp(
                    execution.completed_at or execution.started_at
                ),
                workflow_id=workflow_id,
                execution_id=execution_id,
                plan_id=execution_plan_id,
                status=execution_status,
            )
        )

    if execution.status in {"UNDOING", "UNDONE"}:
        undo_occurred_at = _storage_timestamp(
            execution.undone_at
            or execution.completed_at
            or execution.started_at
        )
        append(
            UndoStartedEvent(
                occurred_at=undo_occurred_at,
                workflow_id=workflow_id,
                execution_id=execution_id,
                plan_id=execution_plan_id,
            )
        )
        for item in sorted(execution_items, key=lambda value: value.sequence_no):
            if item.status == "UNDONE":
                append(
                    UndoItemCompletedEvent(
                        occurred_at=_storage_timestamp(
                            item.undone_at
                            or execution.undone_at
                            or execution.started_at
                        ),
                        workflow_id=workflow_id,
                        execution_id=execution_id,
                        sequence_no=item.sequence_no,
                        source_file_id=item.source_file_id,
                    )
                )
        if execution.status == "UNDONE":
            append(
                UndoCompletedEvent(
                    occurred_at=_storage_timestamp(
                        execution.undone_at or execution.started_at
                    ),
                    workflow_id=workflow_id,
                    execution_id=execution_id,
                    plan_id=execution_plan_id,
                )
            )

    return tuple(events)


def encode_sse_event(
    event: BusinessEvent,
    *,
    event_id: int | None = None,
) -> str:
    """把一个已验证业务事件编码为 SSE 消息。"""

    if event_id is not None and event_id < 1:
        raise ValueError("SSE event id must be positive")

    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.extend(
        [
            f"event: {event.kind}",
            f"data: {event.model_dump_json()}",
        ]
    )
    return "\n".join(lines) + "\n\n"


def _storage_timestamp(value: datetime) -> datetime:
    """SQLite 读取 timezone 列时可能丢失偏移，但记录器保存的是 UTC。"""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
