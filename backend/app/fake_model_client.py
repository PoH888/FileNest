"""用于 Agent 测试的可预测模型客户端。"""

from collections import deque
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict

from .model_client import ModelMessage, ModelResponse
from .tool_registry import ToolDefinition


class FakeModelResponsesExhaustedError(RuntimeError):
    """Fake Model 收到的调用次数超过预设响应数量。"""


class FakeModelCall(BaseModel):
    """Fake Model 收到的一次不可变调用快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: tuple[ModelMessage, ...]
    tools: tuple[ToolDefinition, ...]


class FakeModelClient:
    """按预设顺序返回响应，并记录每次模型调用。"""

    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        self._responses = deque(responses)
        self._calls: list[FakeModelCall] = []

    @property
    def calls(self) -> tuple[FakeModelCall, ...]:
        """返回调用记录的只读视图。"""

        return tuple(self._calls)

    @property
    def remaining_responses(self) -> int:
        """返回尚未消费的预设响应数量。"""

        return len(self._responses)

    def complete(
        self,
        *,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        """记录调用快照，并返回下一条预设响应。"""

        self._calls.append(
            FakeModelCall(
                messages=tuple(messages),
                tools=tuple(tools),
            )
        )

        try:
            return self._responses.popleft()
        except IndexError as error:
            raise FakeModelResponsesExhaustedError(
                "Fake Model 没有剩余的预设响应"
            ) from error
