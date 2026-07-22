from threading import Event

import pytest
from pydantic import BaseModel, ConfigDict

import backend.app.agent_loop as agent_loop_module
from backend.app.agent_loop import AgentLoop
from backend.app.fake_model_client import FakeModelClient
from backend.app.model_client import (
    ModelClientRequestError,
    ModelMessage,
    ModelRequestErrorCode,
    ModelResponse,
    ModelToolCall,
)
from backend.app.tool_contracts import Tool, ToolResult
from backend.app.tool_registry import ToolRegistry


class EmptyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _tool_registry(handler_calls: list[bool]) -> ToolRegistry:
    def handle(_: BaseModel) -> ToolResult:
        handler_calls.append(True)
        return ToolResult.success({"items": [], "count": 0})

    return ToolRegistry(
        [
            Tool(
                name="list_workspaces",
                description="列出允许模型查看的工作区",
                arguments_model=EmptyArguments,
                handler=handle,
            )
        ]
    )


def _tool_call_response() -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(
            role="assistant",
            content="我需要先查询工作区和报告。",
            tool_calls=(
                ModelToolCall(
                    id="call_workspaces_1",
                    name="list_workspaces",
                    arguments={},
                ),
                ModelToolCall(
                    id="call_reports_1",
                    name="search_files",
                    arguments={"workspace_id": 1, "keyword": "report"},
                ),
            ),
        ),
        finish_reason="tool_calls",
    )


def _final_response() -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(
            role="assistant",
            content="没有找到匹配的报告。",
        ),
        finish_reason="stop",
    )


class ScriptedModelClient:
    """按顺序返回响应或抛出稳定模型错误的测试客户端。"""

    def __init__(
        self,
        actions: list[ModelResponse | ModelClientRequestError],
    ) -> None:
        self._actions = list(actions)
        self.calls = 0

    def complete(self, **_: object) -> ModelResponse:
        self.calls += 1
        action = self._actions.pop(0)
        if isinstance(action, ModelClientRequestError):
            raise action
        return action


def _model_request_error(
    *,
    retryable: bool,
    code: ModelRequestErrorCode = "model_connection_error",
) -> ModelClientRequestError:
    return ModelClientRequestError(
        code=code,
        message="模型请求失败",
        retryable=retryable,
    )


def test_model_turn_passes_messages_and_registered_tool_definitions() -> None:
    handler_calls: list[bool] = []
    model_client = FakeModelClient([_tool_call_response()])
    loop = AgentLoop(
        model_client=model_client,
        tool_registry=_tool_registry(handler_calls),
    )

    loop.request_model_turn([ModelMessage(role="user", content="查找报告")])

    assert len(model_client.calls) == 1
    assert [message.content for message in model_client.calls[0].messages] == [
        "查找报告"
    ]
    assert [tool.name for tool in model_client.calls[0].tools] == [
        "list_workspaces"
    ]
    assert handler_calls == []


def test_model_turn_appends_assistant_message_and_parses_tool_calls() -> None:
    handler_calls: list[bool] = []
    response = _tool_call_response()
    loop = AgentLoop(
        model_client=FakeModelClient([response]),
        tool_registry=_tool_registry(handler_calls),
    )
    original_messages = [ModelMessage(role="user", content="查找报告")]

    turn = loop.request_model_turn(original_messages)

    assert original_messages == [ModelMessage(role="user", content="查找报告")]
    assert turn.messages == (*original_messages, response.message)
    assert turn.finish_reason == "tool_calls"
    assert [tool_call.model_dump() for tool_call in turn.tool_calls] == [
        {
            "id": "call_workspaces_1",
            "name": "list_workspaces",
            "arguments": {},
        },
        {
            "id": "call_reports_1",
            "name": "search_files",
            "arguments": {"workspace_id": 1, "keyword": "report"},
        },
    ]
    assert handler_calls == []


def test_tool_results_are_executed_and_returned_to_the_model() -> None:
    handler_calls: list[bool] = []
    model_client = FakeModelClient([_tool_call_response(), _final_response()])
    loop = AgentLoop(
        model_client=model_client,
        tool_registry=_tool_registry(handler_calls),
    )

    first_turn = loop.request_model_turn(
        [ModelMessage(role="user", content="查找报告")]
    )
    messages_with_results = loop.execute_tool_calls(first_turn)
    loop.request_model_turn(messages_with_results)

    assert handler_calls == [True]
    assert len(model_client.calls) == 2
    returned_messages = model_client.calls[1].messages
    assert [message.role for message in returned_messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert [message.tool_call_id for message in returned_messages[-2:]] == [
        "call_workspaces_1",
        "call_reports_1",
    ]

    successful_result = ToolResult.model_validate_json(
        returned_messages[-2].content
    )
    rejected_result = ToolResult.model_validate_json(returned_messages[-1].content)
    assert successful_result == ToolResult.success({"items": [], "count": 0})
    assert rejected_result.ok is False
    assert rejected_result.error is not None
    assert rejected_result.error.code == "unknown_tool"


def test_run_returns_final_answer_after_tool_round() -> None:
    handler_calls: list[bool] = []
    model_client = FakeModelClient([_tool_call_response(), _final_response()])
    loop = AgentLoop(
        model_client=model_client,
        tool_registry=_tool_registry(handler_calls),
    )

    result = loop.run(
        [ModelMessage(role="user", content="查找报告")],
        max_steps=3,
    )

    assert result.status == "completed"
    assert result.final_answer == "没有找到匹配的报告。"
    assert result.model_turns == 2
    assert [message.role for message in result.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert handler_calls == [True]
    assert len(model_client.calls) == 2


def test_run_stops_before_executing_tools_that_cannot_be_returned() -> None:
    handler_calls: list[bool] = []
    model_client = FakeModelClient(
        [_tool_call_response(), _tool_call_response()]
    )
    loop = AgentLoop(
        model_client=model_client,
        tool_registry=_tool_registry(handler_calls),
    )

    result = loop.run(
        [ModelMessage(role="user", content="一直查找报告")],
        max_steps=2,
    )

    assert result.status == "max_steps_reached"
    assert result.final_answer is None
    assert result.model_turns == 2
    assert [message.role for message in result.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert handler_calls == [True]
    assert len(model_client.calls) == 2
    assert model_client.remaining_responses == 0


def test_run_returns_cancelled_before_requesting_model() -> None:
    cancel_event = Event()
    cancel_event.set()
    model_client = FakeModelClient([_final_response()])
    loop = AgentLoop(
        model_client=model_client,
        tool_registry=_tool_registry([]),
    )

    result = loop.run(
        [ModelMessage(role="user", content="查找报告")],
        cancel_event=cancel_event,
    )

    assert result.status == "cancelled"
    assert result.final_answer is None
    assert result.model_turns == 0
    assert [message.role for message in result.messages] == ["user"]
    assert model_client.calls == ()


def test_run_returns_timed_out_after_blocking_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values = iter([0.0, 0.25, 1.0])
    monkeypatch.setattr(
        agent_loop_module,
        "monotonic",
        lambda: next(clock_values),
    )
    model_client = FakeModelClient([_final_response()])
    loop = AgentLoop(
        model_client=model_client,
        tool_registry=_tool_registry([]),
    )

    result = loop.run(
        [ModelMessage(role="user", content="查找报告")],
        timeout_seconds=1.0,
    )

    assert result.status == "timed_out"
    assert result.final_answer is None
    assert result.model_turns == 1
    assert [message.role for message in result.messages] == [
        "user",
        "assistant",
    ]
    assert len(model_client.calls) == 1


def test_run_cancels_between_tool_calls() -> None:
    cancel_event = Event()
    handler_calls: list[bool] = []

    def handle(_: BaseModel) -> ToolResult:
        handler_calls.append(True)
        cancel_event.set()
        return ToolResult.success({"items": [], "count": 0})

    registry = ToolRegistry(
        [
            Tool(
                name="list_workspaces",
                description="列出允许模型查看的工作区",
                arguments_model=EmptyArguments,
                handler=handle,
            )
        ]
    )
    response = ModelResponse(
        message=ModelMessage(
            role="assistant",
            tool_calls=(
                ModelToolCall(
                    id="call_workspaces_1",
                    name="list_workspaces",
                    arguments={},
                ),
                ModelToolCall(
                    id="call_workspaces_2",
                    name="list_workspaces",
                    arguments={},
                ),
            ),
        ),
        finish_reason="tool_calls",
    )
    loop = AgentLoop(
        model_client=FakeModelClient([response]),
        tool_registry=registry,
    )

    result = loop.run(
        [ModelMessage(role="user", content="查询两次工作区")],
        cancel_event=cancel_event,
    )

    assert result.status == "cancelled"
    assert result.final_answer is None
    assert result.model_turns == 1
    assert [message.role for message in result.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assert result.messages[-1].tool_call_id == "call_workspaces_1"
    assert handler_calls == [True]


def test_run_retries_model_request_without_reexecuting_completed_tool() -> None:
    handler_calls: list[bool] = []
    model_client = ScriptedModelClient(
        [
            _tool_call_response(),
            _model_request_error(retryable=True),
            _final_response(),
        ]
    )
    loop = AgentLoop(
        model_client=model_client,
        tool_registry=_tool_registry(handler_calls),
    )

    result = loop.run(
        [ModelMessage(role="user", content="查找报告")],
        retry_base_delay_seconds=0,
    )

    assert result.status == "completed"
    assert result.final_answer == "没有找到匹配的报告。"
    assert result.model_turns == 2
    assert result.error is None
    assert model_client.calls == 3
    assert handler_calls == [True]


def test_run_stops_after_retryable_error_exhausts_retry_budget() -> None:
    model_client = ScriptedModelClient(
        [
            _model_request_error(retryable=True),
            _model_request_error(retryable=True),
            _model_request_error(retryable=True),
            _final_response(),
        ]
    )
    loop = AgentLoop(
        model_client=model_client,
        tool_registry=_tool_registry([]),
    )

    result = loop.run(
        [ModelMessage(role="user", content="查找报告")],
        max_model_retries=2,
        retry_base_delay_seconds=0,
    )

    assert result.status == "failed"
    assert result.final_answer is None
    assert result.model_turns == 0
    assert result.error is not None
    assert result.error.code == "model_connection_error"
    assert result.error.retryable is True
    assert result.error.attempts == 3
    assert model_client.calls == 3


def test_run_does_not_retry_non_retryable_model_error() -> None:
    model_client = ScriptedModelClient(
        [
            _model_request_error(
                retryable=False,
                code="model_request_rejected",
            ),
            _final_response(),
        ]
    )
    loop = AgentLoop(
        model_client=model_client,
        tool_registry=_tool_registry([]),
    )

    result = loop.run(
        [ModelMessage(role="user", content="查找报告")],
        max_model_retries=2,
    )

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "model_request_rejected"
    assert result.error.retryable is False
    assert result.error.attempts == 1
    assert model_client.calls == 1


def test_run_timeout_interrupts_retry_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr(
        agent_loop_module,
        "monotonic",
        lambda: clock["now"],
    )
    monkeypatch.setattr(
        agent_loop_module,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )
    model_client = ScriptedModelClient(
        [
            _model_request_error(retryable=True),
            _final_response(),
        ]
    )
    loop = AgentLoop(
        model_client=model_client,
        tool_registry=_tool_registry([]),
    )

    result = loop.run(
        [ModelMessage(role="user", content="查找报告")],
        timeout_seconds=0.5,
        retry_base_delay_seconds=1.0,
    )

    assert result.status == "timed_out"
    assert result.error is None
    assert result.model_turns == 0
    assert model_client.calls == 1


def test_run_cancellation_interrupts_retry_wait() -> None:
    class CancellingEvent(Event):
        def wait(self, timeout: float | None = None) -> bool:
            self.set()
            return True

    cancel_event = CancellingEvent()
    model_client = ScriptedModelClient(
        [
            _model_request_error(retryable=True),
            _final_response(),
        ]
    )
    loop = AgentLoop(
        model_client=model_client,
        tool_registry=_tool_registry([]),
    )

    result = loop.run(
        [ModelMessage(role="user", content="查找报告")],
        cancel_event=cancel_event,
        retry_base_delay_seconds=0.25,
    )

    assert result.status == "cancelled"
    assert result.error is None
    assert result.model_turns == 0
    assert model_client.calls == 1


@pytest.mark.parametrize("max_steps", [0, -1])
def test_run_rejects_non_positive_step_budget(max_steps: int) -> None:
    loop = AgentLoop(
        model_client=FakeModelClient([]),
        tool_registry=_tool_registry([]),
    )

    with pytest.raises(ValueError, match="max_steps must be at least 1"):
        loop.run(
            [ModelMessage(role="user", content="查找报告")],
            max_steps=max_steps,
        )


@pytest.mark.parametrize(
    "timeout_seconds",
    [0.0, -0.1, float("nan"), float("inf")],
)
def test_run_rejects_non_positive_timeout(timeout_seconds: float) -> None:
    loop = AgentLoop(
        model_client=FakeModelClient([]),
        tool_registry=_tool_registry([]),
    )

    with pytest.raises(
        ValueError,
        match="timeout_seconds must be greater than 0",
    ):
        loop.run(
            [ModelMessage(role="user", content="查找报告")],
            timeout_seconds=timeout_seconds,
        )


@pytest.mark.parametrize("max_model_retries", [-1, 6])
def test_run_rejects_model_retry_budget_outside_safe_range(
    max_model_retries: int,
) -> None:
    loop = AgentLoop(
        model_client=FakeModelClient([]),
        tool_registry=_tool_registry([]),
    )

    with pytest.raises(
        ValueError,
        match="max_model_retries must be between 0 and 5",
    ):
        loop.run(
            [ModelMessage(role="user", content="查找报告")],
            max_model_retries=max_model_retries,
        )


@pytest.mark.parametrize(
    "retry_base_delay_seconds",
    [-0.1, float("nan"), float("inf")],
)
def test_run_rejects_invalid_retry_base_delay(
    retry_base_delay_seconds: float,
) -> None:
    loop = AgentLoop(
        model_client=FakeModelClient([]),
        tool_registry=_tool_registry([]),
    )

    with pytest.raises(
        ValueError,
        match="retry_base_delay_seconds must be finite and non-negative",
    ):
        loop.run(
            [ModelMessage(role="user", content="查找报告")],
            retry_base_delay_seconds=retry_base_delay_seconds,
        )
