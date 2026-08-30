"""Agent 运行轨迹与可恢复上下文的最小持久化边界。"""

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
from typing import Literal, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from sqlalchemy import func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .model_client import ModelCallMetrics, ModelMessage, ModelToolCall
from .models import (
    AgentMessage,
    AgentMetric,
    AgentModelRun,
    AgentRun,
    AgentSession,
    AgentStep,
    AgentToolCall,
    agent_run_sessions,
)
from .repositories import (
    add_agent_run,
    add_agent_tool_call,
    get_agent_run_by_id,
    get_agent_tool_call_by_id,
)
from .tool_contracts import ToolResult


RecordedRunStatus = Literal[
    "completed",
    "max_steps_reached",
    "timed_out",
    "cancelled",
    "failed",
]
RecordedToolStatus = Literal["succeeded", "rejected", "failed"]
RecordedStepStatus = Literal[
    "completed",
    "max_steps_reached",
    "timed_out",
    "cancelled",
    "failed",
]
UtcClock = Callable[[], datetime]
_SAFE_RUN_ERROR_CODES = frozenset(
    {
        "model_timeout",
        "model_connection_error",
        "model_rate_limited",
        "model_server_error",
        "model_request_rejected",
        "model_provider_error",
        "agent_result_persistence_error",
        "worker_interrupted",
    }
)
_SAFE_TOOL_ERROR_CODES = frozenset(
    {
        "file_not_found",
        "invalid_arguments",
        "invalid_tool_result",
        "tool_execution_failed",
        "unknown_tool",
        "workspace_not_found",
    }
)


class AgentRunMetrics(BaseModel):
    """一次 Run 已知的安全模型指标汇总。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_provider: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)

    @field_validator("model_provider", "model_name", "prompt_version")
    @classmethod
    def reject_blank_identity(cls, value: str | None) -> str | None:
        if value is not None and (not value or value != value.strip()):
            raise ValueError("metric identity must be non-empty without whitespace")
        return value

    @model_validator(mode="after")
    def validate_token_usage_pair(self) -> "AgentRunMetrics":
        if (self.input_tokens is None) != (self.output_tokens is None):
            raise ValueError("run token usage must contain both token counts")
        return self


class AgentObservabilityError(RuntimeError):
    """记录失败时向上层公开的稳定且不含数据库细节的错误。"""


class _StepInputSummary(BaseModel):
    """只记录回合输入的形状，不把 prompt 或工具参数写入生命周期表。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message_count: int = Field(ge=0)
    roles: tuple[Literal["system", "user", "assistant", "tool"], ...]


class _TextMessageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["text"]
    content_length: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)


class _ToolCallMessageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["tool_call"]
    call_id_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_name: str = Field(min_length=1, max_length=200)
    argument_keys: tuple[str, ...]


class _ToolResultMessageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["tool_result"]
    ok: bool
    error_code: str | None = None
    item_count: int | None = Field(default=None, ge=0)


class _StepOutputSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message_count: int = Field(ge=0)
    last_role: Literal["user", "assistant", "tool"] | None = None
    error_code: str | None = None


def _validated_messages(messages: Sequence[ModelMessage]) -> tuple[ModelMessage, ...]:
    try:
        return TypeAdapter(tuple[ModelMessage, ...]).validate_python(messages)
    except (TypeError, ValueError, ValidationError) as error:
        raise AgentObservabilityError("Agent 消息上下文不可持久化") from error


def _step_input_summary(messages: Sequence[ModelMessage]) -> str:
    validated_messages = _validated_messages(messages)
    summary = _StepInputSummary(
        message_count=len(validated_messages),
        roles=tuple(message.role for message in validated_messages),
    )
    return summary.model_dump_json()


def _message_summary(
    message: ModelMessage,
) -> tuple[str, Mapping[str, object]]:
    validated_message = _validated_messages((message,))[0]
    if validated_message.role == "system":
        raise AgentObservabilityError("系统消息不允许写入 Agent 生命周期")
    if validated_message.role in {"user", "assistant"}:
        return validated_message.role, _TextMessageSummary(
            kind="text",
            content_length=len(validated_message.content or ""),
            tool_call_count=len(validated_message.tool_calls),
        ).model_dump()

    try:
        tool_result = ToolResult.model_validate_json(validated_message.content or "")
    except (TypeError, ValueError, ValidationError):
        return "tool_result", _ToolResultMessageSummary(
            kind="tool_result",
            ok=False,
            error_code="invalid_tool_result",
        ).model_dump()

    error_code = (
        tool_result.error.code
        if tool_result.error is not None
        and tool_result.error.code in _SAFE_TOOL_ERROR_CODES
        else ("invalid_tool_result" if tool_result.error is not None else None)
    )
    item_count: int | None = None
    if isinstance(tool_result.data, dict):
        items = tool_result.data.get("items")
        if isinstance(items, list):
            item_count = len(items)
    return "tool_result", _ToolResultMessageSummary(
        kind="tool_result",
        ok=tool_result.ok,
        error_code=error_code,
        item_count=item_count,
    ).model_dump()


def _tool_call_summary(tool_call: ModelToolCall) -> Mapping[str, object]:
    try:
        validated_tool_call = ModelToolCall.model_validate(tool_call)
    except (TypeError, ValueError, ValidationError) as error:
        raise AgentObservabilityError("Agent 工具调用不可持久化") from error
    return _ToolCallMessageSummary(
        kind="tool_call",
        call_id_sha256=_hashed_call_id(validated_tool_call.id),
        tool_name=validated_tool_call.name,
        argument_keys=tuple(sorted(validated_tool_call.arguments)),
    ).model_dump()


@runtime_checkable
class AgentRunRecorder(Protocol):
    """Agent Loop 可调用的生命周期与已验证上下文记录契约。"""

    def start_run(self) -> int:
        """创建运行记录并返回程序侧主键。"""

        ...

    def start_step(
        self,
        *,
        agent_run_id: int,
        step_index: int,
        step_type: str,
        messages: Sequence[ModelMessage],
    ) -> int:
        """创建或恢复一个模型回合步骤。"""

        ...

    def next_message_sequence(self, *, agent_step_id: int) -> int:
        """返回步骤内下一个不可重复的消息序号。"""

        ...

    def record_model_message(
        self,
        *,
        agent_run_id: int,
        agent_step_id: int,
        sequence_no: int,
        message: ModelMessage,
    ) -> None:
        """保存经过脱敏的 user/assistant/tool 消息摘要。"""

        ...

    def record_tool_call_message(
        self,
        *,
        agent_run_id: int,
        agent_step_id: int,
        sequence_no: int,
        tool_call: ModelToolCall,
    ) -> None:
        """保存不含工具参数值的工具请求摘要。"""

        ...

    def record_model_run(
        self,
        *,
        agent_run_id: int,
        agent_step_id: int,
        model: str | None,
        model_provider: str | None,
        prompt_version: str | None,
        metrics: ModelCallMetrics | None,
    ) -> int:
        """保存一次模型响应及其安全指标。"""

        ...

    def start_tool_call(
        self,
        *,
        agent_run_id: int,
        sequence_no: int,
        model_call_id: str,
        tool_name: str,
        agent_step_id: int | None = None,
    ) -> int:
        """在工具执行前创建不含参数的调用记录。"""

        ...

    def finish_tool_call(
        self,
        *,
        agent_run_id: int,
        tool_call_record_id: int,
        status: RecordedToolStatus,
        error_code: str | None,
    ) -> None:
        """只记录工具终态与安全错误码。"""

        ...

    def finish_run(
        self,
        *,
        agent_run_id: int,
        status: RecordedRunStatus,
        model_turns: int,
        error_code: str | None,
        metrics: AgentRunMetrics | None = None,
    ) -> None:
        """记录 Agent Run 终态与已知模型指标。"""

        ...

    def finish_step(
        self,
        *,
        agent_run_id: int,
        agent_step_id: int,
        status: RecordedStepStatus,
        messages: Sequence[ModelMessage],
        error_code: str | None,
    ) -> None:
        """以条件更新关闭步骤并保存脱敏输出摘要。"""

        ...

    def record_result(
        self,
        *,
        agent_run_id: int,
        final_answer: str | None,
        sources_json: str | None,
    ) -> None:
        """保存已经过响应模型校验的用户可见结果。"""

        ...

    def checkpoint_run(
        self,
        *,
        agent_run_id: int,
        messages: Sequence[ModelMessage],
        model_turns: int,
    ) -> None:
        """保存可供后续 Resume 使用的已验证消息上下文。"""

        ...


class SqlAlchemyAgentRunRecorder:
    """使用独立提交保存生命周期和可恢复的 Agent 消息上下文。"""

    def __init__(
        self,
        session: Session,
        *,
        clock: UtcClock | None = None,
        workspace_id: int | None = None,
    ) -> None:
        self._session = session
        self._clock = clock or _utc_now
        self._workspace_id = workspace_id
        self._active_session_id: int | None = None

    def start_run(self) -> int:
        agent_run = AgentRun(
            started_at=self._now(),
            workspace_id=self._workspace_id,
        )
        add_agent_run(self._session, agent_run)
        self._session.flush()
        self._ensure_run_session(agent_run)
        self._commit()
        return agent_run.id

    def start_pending_run(
        self,
        *,
        workspace_id: int | None = None,
        request_text: str | None = None,
        messages: Sequence[ModelMessage] | None = None,
    ) -> int:
        """持久化排队状态，供 HTTP 请求返回前取得稳定的 run_id。"""

        agent_run = AgentRun(
            status="pending",
            started_at=self._now(),
            workspace_id=workspace_id,
            request_text=request_text,
            context_json=(
                _serialize_messages(messages) if messages is not None else None
            ),
        )
        add_agent_run(self._session, agent_run)
        self._session.flush()
        self._ensure_run_session(agent_run)
        self._commit()
        return agent_run.id

    def queue_resume(
        self,
        agent_run_id: int,
        *,
        allowed_statuses: Sequence[str],
    ) -> bool:
        """以一次条件更新把可恢复运行重新放入排队状态。"""

        if not allowed_statuses:
            raise ValueError("allowed_statuses must not be empty")
        result = self._session.execute(
            update(AgentRun)
            .where(
                AgentRun.id == agent_run_id,
                AgentRun.status.in_(allowed_statuses),
            )
            .values(
                status="pending",
                finished_at=None,
                error_code=None,
                final_answer=None,
                sources_json=None,
            )
        )
        if result.rowcount != 1:
            self._session.rollback()
            return False
        self._commit()
        return True

    def start_existing_run(self, agent_run_id: int) -> int:
        """把已排队运行原子地推进到执行中，并允许重复领取。"""

        agent_run = get_agent_run_by_id(self._session, agent_run_id)
        if agent_run is None:
            raise AgentObservabilityError("Agent 运行记录不存在")
        if agent_run.status not in {"pending", "running"}:
            raise AgentObservabilityError("Agent 运行记录已结束")
        self._ensure_run_session(agent_run)
        if agent_run.status == "pending":
            agent_run.status = "running"
            self._commit()
        else:
            self._commit()
        return agent_run.id

    def start_step(
        self,
        *,
        agent_run_id: int,
        step_index: int,
        step_type: str,
        messages: Sequence[ModelMessage],
    ) -> int:
        """创建或恢复步骤，并在模型调用前将其推进到 running。"""

        if step_index < 0:
            raise ValueError("step_index must be non-negative")
        if not step_type or step_type != step_type.strip():
            raise ValueError("step_type must be non-empty without whitespace")
        validated_messages = _validated_messages(messages)
        run = get_agent_run_by_id(self._session, agent_run_id)
        if run is None:
            raise AgentObservabilityError("Agent 运行记录不存在")
        session_id = self._ensure_run_session(run)
        existing = self._session.scalar(
            select(AgentStep).where(
                AgentStep.agent_session_id == session_id,
                AgentStep.step_index == step_index,
            )
        )
        if existing is not None:
            if existing.status == "pending":
                self._promote_step_to_running(existing.id)
            elif existing.status != "running":
                raise AgentObservabilityError("Agent 步骤身份已完成，不能重复写入")
            return existing.id

        agent_step = AgentStep(
            agent_session_id=session_id,
            step_index=step_index,
            step_type=step_type,
            input=_step_input_summary(validated_messages),
            status="pending",
            started_at=self._now(),
        )
        self._session.add(agent_step)
        self._commit()
        self._promote_step_to_running(agent_step.id)
        return agent_step.id

    def next_message_sequence(self, *, agent_step_id: int) -> int:
        step = self._get_active_step(agent_step_id)
        latest_sequence = self._session.scalar(
            select(func.max(AgentMessage.sequence_no)).where(
                AgentMessage.agent_step_id == step.id,
            )
        )
        return (latest_sequence if latest_sequence is not None else -1) + 1

    def record_model_message(
        self,
        *,
        agent_run_id: int,
        agent_step_id: int,
        sequence_no: int,
        message: ModelMessage,
    ) -> None:
        """保存不含原文、工具参数和工具结果正文的消息摘要。"""

        if sequence_no < 0:
            raise ValueError("sequence_no must be non-negative")
        self._get_active_step(agent_step_id, agent_run_id=agent_run_id)
        message_type, payload = _message_summary(message)
        self._add_message(
            agent_step_id=agent_step_id,
            sequence_no=sequence_no,
            message_type=message_type,
            payload=payload,
        )

    def record_tool_call_message(
        self,
        *,
        agent_run_id: int,
        agent_step_id: int,
        sequence_no: int,
        tool_call: ModelToolCall,
    ) -> None:
        """保存不含工具参数值的工具请求摘要。"""

        if sequence_no < 0:
            raise ValueError("sequence_no must be non-negative")
        self._get_active_step(agent_step_id, agent_run_id=agent_run_id)
        self._add_message(
            agent_step_id=agent_step_id,
            sequence_no=sequence_no,
            message_type="tool_call",
            payload=_tool_call_summary(tool_call),
        )

    def record_model_run(
        self,
        *,
        agent_run_id: int,
        agent_step_id: int,
        model: str | None,
        model_provider: str | None,
        prompt_version: str | None,
        metrics: ModelCallMetrics | None,
    ) -> int:
        """保存一次模型响应，并将可用指标写入 AgentMetric。"""

        step = self._get_active_step(agent_step_id, agent_run_id=agent_run_id)
        model_name = model or model_provider or "unknown"
        if not model_name or model_name != model_name.strip() or len(model_name) > 200:
            raise AgentObservabilityError("Agent 模型身份不可持久化")
        if model_provider is not None and (
            not model_provider
            or model_provider != model_provider.strip()
            or len(model_provider) > 100
        ):
            raise AgentObservabilityError("Agent 模型供应商不可持久化")
        if prompt_version is not None and (
            not prompt_version
            or prompt_version != prompt_version.strip()
            or len(prompt_version) > 128
        ):
            raise AgentObservabilityError("Agent prompt 版本不可持久化")

        validated_metrics: ModelCallMetrics | None = None
        if metrics is not None:
            try:
                validated_metrics = ModelCallMetrics.model_validate(metrics)
            except (TypeError, ValueError, ValidationError) as error:
                raise AgentObservabilityError("Agent 模型指标不可持久化") from error
        usage = validated_metrics.token_usage if validated_metrics is not None else None
        model_run = AgentModelRun(
            agent_step_id=step.id,
            model=model_name,
            prompt_version=prompt_version,
            input_tokens=usage.input_tokens if usage is not None else None,
            output_tokens=usage.output_tokens if usage is not None else None,
            total_tokens=usage.total_tokens if usage is not None else None,
            latency_ms=(
                validated_metrics.latency_ms
                if validated_metrics is not None
                else 0.0
            ),
            created_at=self._now(),
        )
        self._session.add(model_run)
        self._session.flush()
        self._session.add(
            AgentMetric(
                agent_session_id=step.agent_session_id,
                agent_step_id=step.id,
                agent_model_run_id=model_run.id,
                metric_name="model_turn",
                value_json="1",
                unit="count",
                created_at=self._now(),
            )
        )
        if validated_metrics is not None:
            self._add_model_metrics(
                step=step,
                model_run_id=model_run.id,
                metrics=validated_metrics,
            )
        self._commit()
        return model_run.id

    def start_tool_call(
        self,
        *,
        agent_run_id: int,
        sequence_no: int,
        model_call_id: str,
        tool_name: str,
        agent_step_id: int | None = None,
    ) -> int:
        if agent_step_id is not None:
            self._get_active_step(agent_step_id, agent_run_id=agent_run_id)
        tool_call = AgentToolCall(
            agent_run_id=agent_run_id,
            agent_step_id=agent_step_id,
            sequence_no=sequence_no,
            model_call_id=_hashed_call_id(model_call_id),
            tool_name=tool_name,
            status="requested",
            started_at=self._now(),
        )
        add_agent_tool_call(self._session, tool_call)
        self._commit()
        return tool_call.id

    def finish_tool_call(
        self,
        *,
        agent_run_id: int,
        tool_call_record_id: int,
        status: RecordedToolStatus,
        error_code: str | None,
    ) -> None:
        _validate_terminal_error(
            failed=status != "succeeded",
            error_code=error_code,
            allowed_error_codes=_SAFE_TOOL_ERROR_CODES,
        )
        tool_call = get_agent_tool_call_by_id(
            self._session,
            tool_call_record_id,
        )
        if tool_call is None or tool_call.agent_run_id != agent_run_id:
            raise AgentObservabilityError("工具调用记录不存在")

        tool_call.status = status
        tool_call.finished_at = self._now()
        tool_call.error_code = error_code
        self._commit()

    def finish_run(
        self,
        *,
        agent_run_id: int,
        status: RecordedRunStatus,
        model_turns: int,
        error_code: str | None,
        metrics: AgentRunMetrics | None = None,
    ) -> None:
        _validate_terminal_error(
            failed=status == "failed",
            error_code=error_code,
            allowed_error_codes=_SAFE_RUN_ERROR_CODES,
        )
        agent_run = get_agent_run_by_id(self._session, agent_run_id)
        if agent_run is None:
            raise AgentObservabilityError("Agent 运行记录不存在")

        agent_run.status = status
        agent_run.finished_at = self._now()
        agent_run.model_turns = model_turns
        agent_run.error_code = error_code
        if metrics is not None:
            validated_metrics = AgentRunMetrics.model_validate(metrics)
            agent_run.model_provider = validated_metrics.model_provider
            agent_run.model_name = validated_metrics.model_name
            agent_run.prompt_version = validated_metrics.prompt_version
            agent_run.latency_ms = validated_metrics.latency_ms
            agent_run.input_tokens = validated_metrics.input_tokens
            agent_run.output_tokens = validated_metrics.output_tokens
            agent_run.estimated_cost_usd = (
                validated_metrics.estimated_cost_usd
            )
        self._commit()

    def finish_step(
        self,
        *,
        agent_run_id: int,
        agent_step_id: int,
        status: RecordedStepStatus,
        messages: Sequence[ModelMessage],
        error_code: str | None,
    ) -> None:
        """以状态条件更新关闭步骤，防止重试覆盖已完成证据。"""

        if status == "failed":
            _validate_terminal_error(
                failed=True,
                error_code=error_code,
                allowed_error_codes=_SAFE_RUN_ERROR_CODES,
            )
        elif error_code is not None:
            raise ValueError("non-failed step must not contain an error code")
        validated_messages = _validated_messages(messages)
        step = self._get_active_step(agent_step_id, agent_run_id=agent_run_id)
        last_role = next(
            (
                message.role
                for message in reversed(validated_messages)
                if message.role != "system"
            ),
            None,
        )
        output_summary = _StepOutputSummary(
            message_count=len(validated_messages),
            last_role=last_role,
            error_code=error_code,
        ).model_dump_json()
        result = self._session.execute(
            update(AgentStep)
            .where(
                AgentStep.id == step.id,
                AgentStep.agent_session_id == step.agent_session_id,
                AgentStep.status.in_(("pending", "running")),
            )
            .values(
                status=status,
                output_summary=output_summary,
                completed_at=self._now(),
            )
        )
        if result.rowcount != 1:
            self._session.rollback()
            raise AgentObservabilityError("Agent 步骤状态更新失败")
        self._commit()

    def record_result(
        self,
        *,
        agent_run_id: int,
        final_answer: str | None,
        sources_json: str | None,
    ) -> None:
        """独立提交用户可见结果，不保存隐藏推理或原始工具载荷。"""

        if sources_json is not None:
            try:
                TypeAdapter(list[dict[str, object]]).validate_json(
                    sources_json
                )
            except (TypeError, ValueError, ValidationError) as error:
                raise AgentObservabilityError(
                    "Agent 运行的引用结果不可持久化"
                ) from error

        agent_run = get_agent_run_by_id(self._session, agent_run_id)
        if agent_run is None:
            raise AgentObservabilityError("Agent 运行记录不存在")

        agent_run.final_answer = final_answer
        agent_run.sources_json = sources_json
        self._commit()

    def checkpoint_run(
        self,
        *,
        agent_run_id: int,
        messages: Sequence[ModelMessage],
        model_turns: int,
    ) -> None:
        """独立提交已完成消息，保证中断后能从最近上下文继续。"""

        if model_turns < 0:
            raise ValueError("model_turns must be non-negative")
        agent_run = get_agent_run_by_id(self._session, agent_run_id)
        if agent_run is None:
            raise AgentObservabilityError("Agent 运行记录不存在")
        agent_run.context_json = _serialize_messages(messages)
        agent_run.model_turns = model_turns
        self._commit()

    def load_context(
        self,
        agent_run_id: int,
    ) -> tuple[tuple[ModelMessage, ...], int]:
        """读取并重新校验持久化上下文，拒绝损坏或空上下文。"""

        agent_run = get_agent_run_by_id(self._session, agent_run_id)
        if agent_run is None:
            raise AgentObservabilityError("Agent 运行记录不存在")
        if not agent_run.context_json:
            raise AgentObservabilityError("Agent 运行缺少可恢复的持久状态")
        try:
            messages = TypeAdapter(
                tuple[ModelMessage, ...]
            ).validate_json(agent_run.context_json)
        except (TypeError, ValueError, ValidationError) as error:
            raise AgentObservabilityError(
                "Agent 运行的持久状态不可恢复"
            ) from error
        if not messages:
            raise AgentObservabilityError("Agent 运行缺少可恢复的消息上下文")
        return messages, agent_run.model_turns

    def _ensure_run_session(self, agent_run: AgentRun) -> int:
        if self._workspace_id is not None and (
            agent_run.workspace_id != self._workspace_id
        ):
            raise AgentObservabilityError("Agent 运行不属于当前工作区")

        linked_sessions = sorted(agent_run.sessions, key=lambda item: item.id)
        if not linked_sessions:
            agent_session = AgentSession(workspace_id=agent_run.workspace_id)
            self._session.add(agent_session)
            agent_run.sessions.append(agent_session)
            self._session.flush()
            linked_sessions = [agent_session]

        for agent_session in linked_sessions:
            if agent_session.workspace_id != agent_run.workspace_id:
                raise AgentObservabilityError("Agent 会话与工作区不匹配")
        self._active_session_id = linked_sessions[0].id
        return self._active_session_id

    def _get_active_step(
        self,
        agent_step_id: int,
        *,
        agent_run_id: int | None = None,
    ) -> AgentStep:
        step = self._session.get(AgentStep, agent_step_id)
        if step is None:
            raise AgentObservabilityError("Agent 步骤记录不存在")
        if self._active_session_id is None and agent_run_id is not None:
            run = get_agent_run_by_id(self._session, agent_run_id)
            if run is None:
                raise AgentObservabilityError("Agent 运行记录不存在")
            self._ensure_run_session(run)
        if step.agent_session_id != self._active_session_id:
            raise AgentObservabilityError("Agent 步骤不属于当前会话")
        return step

    def _promote_step_to_running(self, agent_step_id: int) -> None:
        result = self._session.execute(
            update(AgentStep)
            .where(
                AgentStep.id == agent_step_id,
                AgentStep.status == "pending",
            )
            .values(status="running")
        )
        if result.rowcount == 1:
            self._commit()
            return
        self._session.rollback()
        step = self._session.get(AgentStep, agent_step_id)
        if step is None or step.status != "running":
            raise AgentObservabilityError("Agent 步骤状态更新失败")

    def _add_message(
        self,
        *,
        agent_step_id: int,
        sequence_no: int,
        message_type: str,
        payload: Mapping[str, object],
    ) -> None:
        try:
            if message_type in {"user", "assistant"}:
                validated_payload = _TextMessageSummary.model_validate(payload)
            elif message_type == "tool_call":
                validated_payload = _ToolCallMessageSummary.model_validate(payload)
            elif message_type == "tool_result":
                validated_payload = _ToolResultMessageSummary.model_validate(payload)
            else:
                raise ValueError("unsupported message type")
            payload_json = json.dumps(
                validated_payload.model_dump(mode="json", exclude_none=True),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            self._session.add(
                AgentMessage(
                    agent_step_id=agent_step_id,
                    sequence_no=sequence_no,
                    message_type=message_type,
                    payload_json=payload_json,
                    created_at=self._now(),
                )
            )
        except (TypeError, ValueError, ValidationError) as error:
            self._session.rollback()
            raise AgentObservabilityError("Agent 消息载荷不可持久化") from error
        self._commit()

    def _add_model_metrics(
        self,
        *,
        step: AgentStep,
        model_run_id: int,
        metrics: ModelCallMetrics,
    ) -> None:
        metric_values: list[tuple[str, str, str]] = [
            ("latency_ms", str(metrics.latency_ms), "ms"),
        ]
        if metrics.token_usage is not None:
            metric_values.extend(
                [
                    (
                        "input_tokens",
                        str(metrics.token_usage.input_tokens),
                        "tokens",
                    ),
                    (
                        "output_tokens",
                        str(metrics.token_usage.output_tokens),
                        "tokens",
                    ),
                    (
                        "total_tokens",
                        str(metrics.token_usage.total_tokens),
                        "tokens",
                    ),
                ]
            )
        if metrics.estimated_cost_usd is not None:
            metric_values.append(
                ("estimated_cost_usd", str(metrics.estimated_cost_usd), "usd")
            )
        for metric_name, value_json, unit in metric_values:
            self._session.add(
                AgentMetric(
                    agent_session_id=step.agent_session_id,
                    agent_step_id=step.id,
                    agent_model_run_id=model_run_id,
                    metric_name=metric_name,
                    value_json=value_json,
                    unit=unit,
                    created_at=self._now(),
                )
            )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observability clock must return an aware datetime")
        return value.astimezone(timezone.utc)

    def _commit(self) -> None:
        try:
            self._session.commit()
        except SQLAlchemyError:
            self._session.rollback()
            raise AgentObservabilityError("Agent 可观察记录写入失败") from None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hashed_call_id(model_call_id: str) -> str:
    if not model_call_id:
        raise ValueError("model_call_id must not be empty")
    return sha256(model_call_id.encode("utf-8")).hexdigest()


def _validate_terminal_error(
    *,
    failed: bool,
    error_code: str | None,
    allowed_error_codes: frozenset[str],
) -> None:
    if failed and not error_code:
        raise ValueError("failed record must contain an error code")
    if not failed and error_code is not None:
        raise ValueError("successful record must not contain an error code")
    if error_code is not None and error_code not in allowed_error_codes:
        raise ValueError("record contains an unsupported error code")


def _serialize_messages(messages: Sequence[ModelMessage]) -> str:
    return json.dumps(
        [message.model_dump(mode="json") for message in messages],
        ensure_ascii=False,
        separators=(",", ":"),
    )
