import pytest
from pydantic import BaseModel, ConfigDict

from backend.app.agent_loop import AgentLoop
from backend.app.fake_model_client import FakeModelClient
from backend.app.model_client import (
    ModelMessage,
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
