"""Agent 运行轨迹与可恢复上下文的最小持久化边界。"""

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Literal, Protocol, runtime_checkable

from pydantic import TypeAdapter, ValidationError
from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .model_client import ModelMessage
from .models import AgentRun, AgentToolCall
from .repositories import (
    add_agent_run,
    add_agent_tool_call,
    get_agent_run_by_id,
    get_agent_tool_call_by_id,
)


RecordedRunStatus = Literal[
    "completed",
    "max_steps_reached",
    "timed_out",
    "cancelled",
    "failed",
]
RecordedToolStatus = Literal["succeeded", "rejected", "failed"]
UtcClock = Callable[[], datetime]
_SAFE_RUN_ERROR_CODES = frozenset(
    {
        "model_timeout",
        "model_connection_error",
        "model_rate_limited",
        "model_server_error",
        "model_request_rejected",
        "model_provider_error",
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


class AgentObservabilityError(RuntimeError):
    """记录失败时向上层公开的稳定且不含数据库细节的错误。"""


@runtime_checkable
class AgentRunRecorder(Protocol):
    """Agent Loop 可调用的生命周期与已验证上下文记录契约。"""

    def start_run(self) -> int:
        """创建运行记录并返回程序侧主键。"""

        ...

    def start_tool_call(
        self,
        *,
        agent_run_id: int,
        sequence_no: int,
        model_call_id: str,
        tool_name: str,
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
    ) -> None:
        """记录 Agent Run 终态。"""

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
    ) -> None:
        self._session = session
        self._clock = clock or _utc_now

    def start_run(self) -> int:
        agent_run = AgentRun(started_at=self._now())
        add_agent_run(self._session, agent_run)
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
        if agent_run.status == "pending":
            agent_run.status = "running"
            self._commit()
        elif agent_run.status != "running":
            raise AgentObservabilityError("Agent 运行记录已结束")
        return agent_run.id

    def start_tool_call(
        self,
        *,
        agent_run_id: int,
        sequence_no: int,
        model_call_id: str,
        tool_name: str,
    ) -> int:
        tool_call = AgentToolCall(
            agent_run_id=agent_run_id,
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
