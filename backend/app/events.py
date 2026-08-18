"""供实时界面消费的业务事件与 SSE 文本协议。"""

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Literal, TypeAlias
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .agent_loop import AGENT_RUN_ACTIVE_STATUSES, AgentRunLifecycleStatus
from .agent_observability import RecordedToolStatus
from .models import AgentRun, AgentToolCall
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


BusinessEvent: TypeAlias = (
    AgentRunStatusChangedEvent
    | AgentToolCallStatusChangedEvent
    | WorkflowStatusChangedEvent
    | ApprovalStatusChangedEvent
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

    if agent_run.status not in AGENT_RUN_ACTIVE_STATUSES:
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
