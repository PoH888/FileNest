"""正式 Agent Loop 的模型回合编排。"""

from collections.abc import Sequence
from decimal import Decimal
from math import isfinite
from threading import Event
from time import monotonic, sleep
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent_observability import (
    AgentRunMetrics,
    AgentRunRecorder,
    RecordedToolStatus,
)
from .model_client import (
    ModelClient,
    ModelClientRequestError,
    ModelCallMetrics,
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
    model_provider: str | None = None
    model_name: str | None = None
    metrics: ModelCallMetrics | None = None


AgentRunStatus = Literal[
    "completed",
    "max_steps_reached",
    "timed_out",
    "cancelled",
    "failed",
]
AgentRunLifecycleStatus = Literal[
    "pending",
    "running",
    "waiting_approval",
] | AgentRunStatus
AGENT_RUN_ACTIVE_STATUSES = frozenset(
    {"pending", "running", "waiting_approval"}
)
AgentBoundaryStatus = Literal["timed_out", "cancelled"]
MAX_MODEL_RETRIES = 5


class _RunMetricsAccumulator:
    """只累计 provider 实际返回的指标，不对缺失 usage 做估算。"""

    def __init__(self, prompt_version: str | None) -> None:
        self._prompt_version = prompt_version
        self._model_providers: set[str] = set()
        self._model_names: set[str] = set()
        self._latency_ms = 0.0
        self._has_latency = False
        self._input_tokens = 0
        self._output_tokens = 0
        self._has_complete_usage = True
        self._has_model_response = False
        self._estimated_cost_usd = Decimal("0")
        self._has_complete_cost = True

    def add(self, turn: AgentModelTurn) -> None:
        """记录一次已返回的模型响应及其可用指标。"""

        self._has_model_response = True
        if turn.model_provider is not None:
            self._model_providers.add(turn.model_provider)
        if turn.model_name is not None:
            self._model_names.add(turn.model_name)

        if turn.metrics is None:
            self._has_complete_usage = False
            self._has_complete_cost = False
            return

        self._latency_ms += turn.metrics.latency_ms
        self._has_latency = True
        if turn.metrics.token_usage is None:
            self._has_complete_usage = False
        elif self._has_complete_usage:
            self._input_tokens += turn.metrics.token_usage.input_tokens
            self._output_tokens += turn.metrics.token_usage.output_tokens

        if turn.metrics.estimated_cost_usd is None:
            self._has_complete_cost = False
        elif self._has_complete_cost:
            self._estimated_cost_usd += turn.metrics.estimated_cost_usd

    def snapshot(self) -> AgentRunMetrics:
        """生成可写入 AgentRun 的安全汇总。"""

        return AgentRunMetrics(
            model_provider=_single_value(self._model_providers),
            model_name=_single_value(self._model_names),
            prompt_version=self._prompt_version,
            latency_ms=self._latency_ms if self._has_latency else None,
            input_tokens=(
                self._input_tokens
                if self._has_model_response and self._has_complete_usage
                else None
            ),
            output_tokens=(
                self._output_tokens
                if self._has_model_response and self._has_complete_usage
                else None
            ),
            estimated_cost_usd=(
                self._estimated_cost_usd
                if self._has_model_response and self._has_complete_cost
                else None
            ),
        )


def _single_value(values: set[str]) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


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
        prompt_version: str | None = None,
    ) -> None:
        if prompt_version is not None and (
            not prompt_version or prompt_version != prompt_version.strip()
        ):
            raise ValueError(
                "prompt_version must be non-empty without surrounding whitespace"
            )
        self._model_client = model_client
        self._tool_registry = tool_registry
        self._recorder = recorder
        self._prompt_version = prompt_version

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
            model_provider=response.model_provider,
            model_name=response.model_name,
            metrics=response.metrics,
        )

    def run(
        self,
        messages: Sequence[ModelMessage],
        *,
        initial_model_turns: int = 0,
        initial_tool_sequence_no: int = 0,
        max_steps: int = 8,
        timeout_seconds: float = 60.0,
        cancel_event: Event | None = None,
        max_model_retries: int = 2,
        retry_base_delay_seconds: float = 0.25,
    ) -> AgentRunResult:
        """推进模型与工具回合，直到完成或触发程序侧运行边界。"""

        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if initial_model_turns < 0:
            raise ValueError("initial_model_turns must be non-negative")
        if initial_tool_sequence_no < 0:
            raise ValueError("initial_tool_sequence_no must be non-negative")
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
        run_metrics = _RunMetricsAccumulator(self._prompt_version)
        current_messages = tuple(messages)
        self._checkpoint_run(
            agent_run_id=agent_run_id,
            messages=current_messages,
            model_turns=initial_model_turns,
        )
        tool_sequence_no = initial_tool_sequence_no
        for additional_turn in range(1, max_steps + 1):
            model_turns = initial_model_turns + additional_turn
            agent_step_id = self._start_agent_step(
                agent_run_id=agent_run_id,
                step_index=model_turns - 1,
                messages=current_messages,
            )
            message_sequence_no = self._record_step_input(
                agent_run_id=agent_run_id,
                agent_step_id=agent_step_id,
                messages=current_messages,
            )
            turn_or_result = self._request_model_turn_with_retries(
                current_messages,
                completed_model_turns=model_turns - 1,
                deadline=deadline,
                cancel_event=cancel_event,
                max_model_retries=max_model_retries,
                retry_base_delay_seconds=retry_base_delay_seconds,
            )
            if isinstance(turn_or_result, AgentRunResult):
                self._finish_agent_step(
                    agent_run_id=agent_run_id,
                    agent_step_id=agent_step_id,
                    status=turn_or_result.status,
                    messages=turn_or_result.messages,
                    error_code=(
                        turn_or_result.error.code
                        if turn_or_result.error is not None
                        else None
                    ),
                )
                return self._finish_recorded_run(
                    turn_or_result,
                    agent_run_id=agent_run_id,
                    metrics=run_metrics.snapshot(),
                )

            turn = turn_or_result
            run_metrics.add(turn)
            message_sequence_no = self._record_model_response(
                agent_run_id=agent_run_id,
                agent_step_id=agent_step_id,
                sequence_no=message_sequence_no,
                turn=turn,
            )
            current_messages = turn.messages
            self._checkpoint_run(
                agent_run_id=agent_run_id,
                messages=current_messages,
                model_turns=model_turns,
            )

            boundary_status = self._boundary_status(
                deadline=deadline,
                cancel_event=cancel_event,
            )
            if boundary_status is not None:
                self._finish_agent_step(
                    agent_run_id=agent_run_id,
                    agent_step_id=agent_step_id,
                    status=boundary_status,
                    messages=current_messages,
                    error_code=None,
                )
                return self._finish_recorded_run(
                    AgentRunResult(
                        status=boundary_status,
                        messages=current_messages,
                        model_turns=model_turns,
                    ),
                    agent_run_id=agent_run_id,
                    metrics=run_metrics.snapshot(),
                )

            if turn.finish_reason == "stop":
                self._finish_agent_step(
                    agent_run_id=agent_run_id,
                    agent_step_id=agent_step_id,
                    status="completed",
                    messages=current_messages,
                    error_code=None,
                )
                return self._finish_recorded_run(
                    AgentRunResult(
                        status="completed",
                        messages=current_messages,
                        model_turns=model_turns,
                        final_answer=current_messages[-1].content,
                    ),
                    agent_run_id=agent_run_id,
                    metrics=run_metrics.snapshot(),
                )

            if additional_turn == max_steps:
                self._finish_agent_step(
                    agent_run_id=agent_run_id,
                    agent_step_id=agent_step_id,
                    status="max_steps_reached",
                    messages=current_messages,
                    error_code=None,
                )
                return self._finish_recorded_run(
                    AgentRunResult(
                        status="max_steps_reached",
                        messages=current_messages,
                        model_turns=model_turns,
                    ),
                    agent_run_id=agent_run_id,
                    metrics=run_metrics.snapshot(),
                )

            for tool_call in turn.tool_calls:
                boundary_status = self._boundary_status(
                    deadline=deadline,
                    cancel_event=cancel_event,
                )
                if boundary_status is not None:
                    self._finish_agent_step(
                        agent_run_id=agent_run_id,
                        agent_step_id=agent_step_id,
                        status=boundary_status,
                        messages=current_messages,
                        error_code=None,
                    )
                    return self._finish_recorded_run(
                        AgentRunResult(
                            status=boundary_status,
                            messages=current_messages,
                            model_turns=model_turns,
                        ),
                        agent_run_id=agent_run_id,
                        metrics=run_metrics.snapshot(),
                    )

                message_sequence_no = self._record_tool_call_message(
                    agent_run_id=agent_run_id,
                    agent_step_id=agent_step_id,
                    sequence_no=message_sequence_no,
                    tool_call=tool_call,
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
                        agent_step_id=agent_step_id,
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
                message_sequence_no = self._record_step_message(
                    agent_run_id=agent_run_id,
                    agent_step_id=agent_step_id,
                    sequence_no=message_sequence_no,
                    message=current_messages[-1],
                )
                self._checkpoint_run(
                    agent_run_id=agent_run_id,
                    messages=current_messages,
                    model_turns=model_turns,
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
                    self._finish_agent_step(
                        agent_run_id=agent_run_id,
                        agent_step_id=agent_step_id,
                        status=boundary_status,
                        messages=current_messages,
                        error_code=None,
                    )
                    return self._finish_recorded_run(
                        AgentRunResult(
                            status=boundary_status,
                            messages=current_messages,
                            model_turns=model_turns,
                        ),
                        agent_run_id=agent_run_id,
                        metrics=run_metrics.snapshot(),
                    )

            self._finish_agent_step(
                agent_run_id=agent_run_id,
                agent_step_id=agent_step_id,
                status="completed",
                messages=current_messages,
                error_code=None,
            )

        raise RuntimeError("Agent Loop reached an unreachable state")

    def _start_agent_step(
        self,
        *,
        agent_run_id: int | None,
        step_index: int,
        messages: Sequence[ModelMessage],
    ) -> int | None:
        if self._recorder is None or agent_run_id is None:
            return None
        start_step = getattr(self._recorder, "start_step", None)
        if not callable(start_step):
            return None
        return start_step(
            agent_run_id=agent_run_id,
            step_index=step_index,
            step_type="model_turn",
            messages=messages,
        )

    def _record_step_input(
        self,
        *,
        agent_run_id: int | None,
        agent_step_id: int | None,
        messages: Sequence[ModelMessage],
    ) -> int:
        if self._recorder is None or agent_run_id is None or agent_step_id is None:
            return 0
        message = next(
            (candidate for candidate in reversed(messages) if candidate.role != "system"),
            None,
        )
        if message is None:
            return 0
        next_sequence = getattr(
            self._recorder,
            "next_message_sequence",
            None,
        )
        sequence_no = (
            next_sequence(agent_step_id=agent_step_id)
            if callable(next_sequence)
            else 0
        )
        return self._record_step_message(
            agent_run_id=agent_run_id,
            agent_step_id=agent_step_id,
            sequence_no=sequence_no,
            message=message,
        )

    def _record_model_response(
        self,
        *,
        agent_run_id: int | None,
        agent_step_id: int | None,
        sequence_no: int,
        turn: AgentModelTurn,
    ) -> int:
        if self._recorder is None or agent_run_id is None or agent_step_id is None:
            return sequence_no
        record_model_run = getattr(self._recorder, "record_model_run", None)
        if callable(record_model_run):
            record_model_run(
                agent_run_id=agent_run_id,
                agent_step_id=agent_step_id,
                model=turn.model_name,
                model_provider=turn.model_provider,
                prompt_version=self._prompt_version,
                metrics=turn.metrics,
            )
        return self._record_step_message(
            agent_run_id=agent_run_id,
            agent_step_id=agent_step_id,
            sequence_no=sequence_no,
            message=turn.messages[-1],
        )

    def _record_tool_call_message(
        self,
        *,
        agent_run_id: int | None,
        agent_step_id: int | None,
        sequence_no: int,
        tool_call: ModelToolCall,
    ) -> int:
        if self._recorder is None or agent_run_id is None or agent_step_id is None:
            return sequence_no
        record_tool_call_message = getattr(
            self._recorder,
            "record_tool_call_message",
            None,
        )
        if callable(record_tool_call_message):
            record_tool_call_message(
                agent_run_id=agent_run_id,
                agent_step_id=agent_step_id,
                sequence_no=sequence_no,
                tool_call=tool_call,
            )
            return sequence_no + 1
        return sequence_no

    def _record_step_message(
        self,
        *,
        agent_run_id: int | None,
        agent_step_id: int | None,
        sequence_no: int,
        message: ModelMessage,
    ) -> int:
        if self._recorder is None or agent_run_id is None or agent_step_id is None:
            return sequence_no
        record_model_message = getattr(
            self._recorder,
            "record_model_message",
            None,
        )
        if callable(record_model_message):
            record_model_message(
                agent_run_id=agent_run_id,
                agent_step_id=agent_step_id,
                sequence_no=sequence_no,
                message=message,
            )
            return sequence_no + 1
        return sequence_no

    def _finish_agent_step(
        self,
        *,
        agent_run_id: int | None,
        agent_step_id: int | None,
        status: str,
        messages: Sequence[ModelMessage],
        error_code: str | None,
    ) -> None:
        if self._recorder is None or agent_run_id is None or agent_step_id is None:
            return
        finish_step = getattr(self._recorder, "finish_step", None)
        if callable(finish_step):
            finish_step(
                agent_run_id=agent_run_id,
                agent_step_id=agent_step_id,
                status=status,
                messages=messages,
                error_code=error_code,
            )

    def _checkpoint_run(
        self,
        *,
        agent_run_id: int | None,
        messages: Sequence[ModelMessage],
        model_turns: int,
    ) -> None:
        """只对支持恢复的记录器保存消息，兼容旧的最小记录器。"""

        if self._recorder is None or agent_run_id is None:
            return
        checkpoint = getattr(self._recorder, "checkpoint_run", None)
        if checkpoint is not None:
            checkpoint(
                agent_run_id=agent_run_id,
                messages=messages,
                model_turns=model_turns,
            )

    def _finish_recorded_run(
        self,
        result: AgentRunResult,
        *,
        agent_run_id: int | None,
        metrics: AgentRunMetrics | None = None,
    ) -> AgentRunResult:
        """先持久化终态，再把同一运行结果交还调用方。"""

        if self._recorder is not None and agent_run_id is not None:
            self._recorder.finish_run(
                agent_run_id=agent_run_id,
                status=result.status,
                model_turns=result.model_turns,
                metrics=metrics,
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
