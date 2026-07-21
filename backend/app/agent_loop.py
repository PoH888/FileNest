"""正式 Agent Loop 的模型回合编排。"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .model_client import (
    ModelClient,
    ModelFinishReason,
    ModelMessage,
    ModelToolCall,
)
from .tool_registry import ToolRegistry


class AgentModelTurn(BaseModel):
    """模型完成一次响应后，Agent Loop 可继续处理的不可变状态。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: tuple[ModelMessage, ...]
    finish_reason: ModelFinishReason
    tool_calls: tuple[ModelToolCall, ...]


AgentRunStatus = Literal["completed", "max_steps_reached"]


class AgentRunResult(BaseModel):
    """一次 Agent Loop 运行的最终状态与完整消息证据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: AgentRunStatus
    messages: tuple[ModelMessage, ...]
    model_turns: int = Field(ge=1)
    final_answer: str | None = None

    @model_validator(mode="after")
    def validate_final_state(self) -> "AgentRunResult":
        """保证完成状态与最终回答存在性一致。"""

        if self.status == "completed" and not self.final_answer:
            raise ValueError("completed run must contain a final answer")
        if self.status == "max_steps_reached" and self.final_answer is not None:
            raise ValueError("stopped run must not contain a final answer")
        return self


class AgentLoop:
    """连接供应商无关模型客户端与程序侧工具注册表。"""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        tool_registry: ToolRegistry,
    ) -> None:
        self._model_client = model_client
        self._tool_registry = tool_registry

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
    ) -> AgentRunResult:
        """推进模型与工具回合，直到最终回答或模型调用预算耗尽。"""

        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        current_messages = tuple(messages)
        for model_turns in range(1, max_steps + 1):
            turn = self.request_model_turn(current_messages)
            current_messages = turn.messages

            if turn.finish_reason == "stop":
                return AgentRunResult(
                    status="completed",
                    messages=current_messages,
                    model_turns=model_turns,
                    final_answer=current_messages[-1].content,
                )

            if model_turns == max_steps:
                return AgentRunResult(
                    status="max_steps_reached",
                    messages=current_messages,
                    model_turns=model_turns,
                )

            current_messages = self.execute_tool_calls(turn)

        raise RuntimeError("Agent Loop reached an unreachable state")

    def execute_tool_calls(
        self,
        turn: AgentModelTurn,
    ) -> tuple[ModelMessage, ...]:
        """执行通过程序校验的工具，并把安全结果追加到消息历史。"""

        messages = list(turn.messages)
        for tool_call in turn.tool_calls:
            validation = self._tool_registry.validate(
                tool_call.name,
                tool_call.arguments,
            )
            result = (
                self._tool_registry.invoke(
                    tool_call.name,
                    tool_call.arguments,
                )
                if validation.ok
                else validation
            )
            messages.append(
                ModelMessage(
                    role="tool",
                    content=result.model_dump_json(),
                    tool_call_id=tool_call.id,
                )
            )

        return tuple(messages)
