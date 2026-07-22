"""正式 Agent Loop 的模型回合编排。"""

from collections.abc import Sequence
from math import isfinite
from threading import Event
from time import monotonic, sleep
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent_observability import (
    AgentRunRecorder,
    RecordedToolStatus,
)
from .model_client import (
    ModelClient,
    ModelClientRequestError,
    ModelFinishReason,
    ModelMessage,
    ModelRequestErrorCode,
    ModelToolCall,
)
from .tool_registry import ToolRegistry


class AgentModelTurn(BaseModel):
    """模型完成一次响应后，Agent Loop 可继续处理的不可变状态。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: tuple[ModelMessage, ...]
    finish_reason: ModelFinishReason
    tool_calls: tuple[ModelToolCall, ...]


AgentRunStatus = Literal[
    "completed",
    "max_steps_reached",
    "timed_out",
    "cancelled",
    "failed",
]
AgentBoundaryStatus = Literal["timed_out", "cancelled"]
MAX_MODEL_RETRIES = 5


class AgentRunError(BaseModel):
    """一次 Agent 运行可安全向上层公开的模型请求错误。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: ModelRequestErrorCode
    retryable: bool
    attempts: int = Field(ge=1)


class AgentRunResult(BaseModel):
    """一次 Agent Loop 运行的最终状态与完整消息证据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: AgentRunStatus
    messages: tuple[ModelMessage, ...]
    model_turns: int = Field(ge=0)
    final_answer: str | None = None
    error: AgentRunError | None = None

    @model_validator(mode="after")
    def validate_final_state(self) -> "AgentRunResult":
        """保证完成状态与最终回答存在性一致。"""

        if self.status == "completed" and not self.final_answer:
            raise ValueError("completed run must contain a final answer")
        if self.status == "completed" and self.model_turns < 1:
            raise ValueError("completed run must contain at least one model turn")
        if self.status != "completed" and self.final_answer is not None:
            raise ValueError("stopped run must not contain a final answer")
        if self.status == "failed" and self.error is None:
            raise ValueError("failed run must contain a safe error")
        if self.status != "failed" and self.error is not None:
            raise ValueError("non-failed run must not contain an error")
        return self


class AgentLoop:
    """连接供应商无关模型客户端与程序侧工具注册表。"""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        tool_registry: ToolRegistry,
        recorder: AgentRunRecorder | None = None,
    ) -> None:
        self._model_client = model_client
        self._tool_registry = tool_registry
        self._recorder = recorder

    def request_model_turn(
        self,
        messages: Sequence[ModelMessage],
    ) -> AgentModelTurn:
        """请求下一条助手消息，并提取其中尚未执行的工具调用。"""

        current_messages = tuple(messages)
        response = self._model_client.complete(
            messages=current_messages,
            tools=self._tool_registry.definitions(),
        )

        return AgentModelTurn(
            messages=(*current_messages, response.message),
            finish_reason=response.finish_reason,
            tool_calls=response.message.tool_calls,
        )

    def run(
        self,
        messages: Sequence[ModelMessage],
        *,
        max_steps: int = 8,
        timeout_seconds: float = 60.0,
        cancel_event: Event | None = None,
        max_model_retries: int = 2,
        retry_base_delay_seconds: float = 0.25,
    ) -> AgentRunResult:
        """推进模型与工具回合，直到完成或触发程序侧运行边界。"""

        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        if not 0 <= max_model_retries <= MAX_MODEL_RETRIES:
            raise ValueError(
                f"max_model_retries must be between 0 and {MAX_MODEL_RETRIES}"
            )
        if (
            not isfinite(retry_base_delay_seconds)
            or retry_base_delay_seconds < 0
        ):
            raise ValueError(
                "retry_base_delay_seconds must be finite and non-negative"
            )

        deadline = monotonic() + timeout_seconds
        agent_run_id = (
            self._recorder.start_run()
            if self._recorder is not None
            else None
        )
        current_messages = tuple(messages)
        tool_sequence_no = 0
        for model_turns in range(1, max_steps + 1):
            turn_or_result = self._request_model_turn_with_retries(
                current_messages,
                completed_model_turns=model_turns - 1,
                deadline=deadline,
                cancel_event=cancel_event,
                max_model_retries=max_model_retries,
                retry_base_delay_seconds=retry_base_delay_seconds,
            )
            if isinstance(turn_or_result, AgentRunResult):
                return self._finish_recorded_run(
                    turn_or_result,
                    agent_run_id=agent_run_id,
                )

            turn = turn_or_result
            current_messages = turn.messages

            boundary_status = self._boundary_status(
                deadline=deadline,
                cancel_event=cancel_event,
            )
            if boundary_status is not None:
                return self._finish_recorded_run(
                    AgentRunResult(
                        status=boundary_status,
                        messages=current_messages,
                        model_turns=model_turns,
                    ),
                    agent_run_id=agent_run_id,
                )

            if turn.finish_reason == "stop":
                return self._finish_recorded_run(
                    AgentRunResult(
                        status="completed",
                        messages=current_messages,
                        model_turns=model_turns,
                        final_answer=current_messages[-1].content,
                    ),
                    agent_run_id=agent_run_id,
                )

            if model_turns == max_steps:
                return self._finish_recorded_run(
                    AgentRunResult(
                        status="max_steps_reached",
                        messages=current_messages,
                        model_turns=model_turns,
                    ),
                    agent_run_id=agent_run_id,
                )

            for tool_call in turn.tool_calls:
                boundary_status = self._boundary_status(
                    deadline=deadline,
                    cancel_event=cancel_event,
                )
                if boundary_status is not None:
                    return self._finish_recorded_run(
                        AgentRunResult(
                            status=boundary_status,
                            messages=current_messages,
                            model_turns=model_turns,
                        ),
                        agent_run_id=agent_run_id,
                    )

                tool_sequence_no += 1
                tool_record_id = (
                    self._recorder.start_tool_call(
                        agent_run_id=agent_run_id,
                        sequence_no=tool_sequence_no,
                        model_call_id=tool_call.id,
                        tool_name=(
                            tool_call.name
                            if tool_call.name in self._tool_registry.names
                            else "unknown_tool"
                        ),
                    )
                    if self._recorder is not None
                    and agent_run_id is not None
                    else None
                )
                (
                    current_messages,
                    tool_status,
                    tool_error_code,
                ) = self._execute_tool_call_with_observation(
                    current_messages,
                    tool_call,
                )
                if (
                    self._recorder is not None
                    and agent_run_id is not None
                    and tool_record_id is not None
                ):
                    self._recorder.finish_tool_call(
                        agent_run_id=agent_run_id,
                        tool_call_record_id=tool_record_id,
                        status=tool_status,
                        error_code=tool_error_code,
                    )

                boundary_status = self._boundary_status(
                    deadline=deadline,
                    cancel_event=cancel_event,
                )
                if boundary_status is not None:
                    return self._finish_recorded_run(
                        AgentRunResult(
                            status=boundary_status,
                            messages=current_messages,
                            model_turns=model_turns,
                        ),
                        agent_run_id=agent_run_id,
                    )

        raise RuntimeError("Agent Loop reached an unreachable state")

    def _finish_recorded_run(
        self,
        result: AgentRunResult,
        *,
        agent_run_id: int | None,
    ) -> AgentRunResult:
        """先持久化终态，再把同一运行结果交还调用方。"""

        if self._recorder is not None and agent_run_id is not None:
            self._recorder.finish_run(
                agent_run_id=agent_run_id,
                status=result.status,
                model_turns=result.model_turns,
                error_code=(
                    result.error.code
                    if result.error is not None
                    else None
                ),
            )
        return result

    def _request_model_turn_with_retries(
        self,
        messages: Sequence[ModelMessage],
        *,
        completed_model_turns: int,
        deadline: float,
        cancel_event: Event | None,
        max_model_retries: int,
        retry_base_delay_seconds: float,
    ) -> AgentModelTurn | AgentRunResult:
        """只重试尚未产生模型响应的可重试请求失败。"""

        attempts = 0
        while True:
            boundary_status = self._boundary_status(
                deadline=deadline,
                cancel_event=cancel_event,
            )
            if boundary_status is not None:
                return AgentRunResult(
                    status=boundary_status,
                    messages=tuple(messages),
                    model_turns=completed_model_turns,
                )

            attempts += 1
            try:
                return self.request_model_turn(messages)
            except ModelClientRequestError as error:
                boundary_status = self._boundary_status(
                    deadline=deadline,
                    cancel_event=cancel_event,
                )
                if boundary_status is not None:
                    return AgentRunResult(
                        status=boundary_status,
                        messages=tuple(messages),
                        model_turns=completed_model_turns,
                    )

                if not error.retryable or attempts > max_model_retries:
                    return AgentRunResult(
                        status="failed",
                        messages=tuple(messages),
                        model_turns=completed_model_turns,
                        error=AgentRunError(
                            code=error.code,
                            retryable=error.retryable,
                            attempts=attempts,
                        ),
                    )

                self._wait_before_retry(
                    failed_attempts=attempts,
                    base_delay_seconds=retry_base_delay_seconds,
                    deadline=deadline,
                    cancel_event=cancel_event,
                )

    @staticmethod
    def _wait_before_retry(
        *,
        failed_attempts: int,
        base_delay_seconds: float,
        deadline: float,
        cancel_event: Event | None,
    ) -> None:
        """指数等待不越过总时限，并允许调用方在等待期间取消。"""

        remaining_seconds = max(0.0, deadline - monotonic())
        delay_seconds = min(
            base_delay_seconds * (2 ** (failed_attempts - 1)),
            remaining_seconds,
        )
        if delay_seconds <= 0:
            return

        if cancel_event is not None:
            cancel_event.wait(delay_seconds)
        else:
            sleep(delay_seconds)

    @staticmethod
    def _boundary_status(
        *,
        deadline: float,
        cancel_event: Event | None,
    ) -> AgentBoundaryStatus | None:
        """在可安全停止的边界优先响应调用方取消，再判断总时限。"""

        if cancel_event is not None and cancel_event.is_set():
            return "cancelled"
        if monotonic() >= deadline:
            return "timed_out"
        return None

    def execute_tool_calls(
        self,
        turn: AgentModelTurn,
    ) -> tuple[ModelMessage, ...]:
        """执行通过程序校验的工具，并把安全结果追加到消息历史。"""

        messages = turn.messages
        for tool_call in turn.tool_calls:
            messages, _, _ = self._execute_tool_call_with_observation(
                messages,
                tool_call,
            )

        return messages

    def _execute_tool_call_with_observation(
        self,
        messages: Sequence[ModelMessage],
        tool_call: ModelToolCall,
    ) -> tuple[
        tuple[ModelMessage, ...],
        RecordedToolStatus,
        str | None,
    ]:
        """执行单个工具并生成不含参数与结果的安全终态。"""

        validation = self._tool_registry.validate(
            tool_call.name,
            tool_call.arguments,
        )
        if validation.ok:
            result = self._tool_registry.invoke(
                tool_call.name,
                tool_call.arguments,
            )
            tool_status: RecordedToolStatus = (
                "succeeded" if result.ok else "failed"
            )
        else:
            result = validation
            tool_status = "rejected"

        error_code = (
            result.error.code
            if result.error is not None
            else None
        )
        tool_message = ModelMessage(
            role="tool",
            content=result.model_dump_json(),
            tool_call_id=tool_call.id,
        )

        return (*messages, tool_message), tool_status, error_code
