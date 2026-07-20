import pytest

from backend.app.fake_model_client import (
    FakeModelClient,
    FakeModelResponsesExhaustedError,
)
from backend.app.model_client import (
    ModelClient,
    ModelMessage,
    ModelResponse,
    ModelToolCall,
)
from backend.app.tool_registry import ToolDefinition


def _search_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="search_files",
        description="搜索已授权工作区中的文件索引",
        parameters={
            "type": "object",
            "properties": {
                "workspace_id": {"type": "integer"},
                "keyword": {"type": "string"},
            },
            "required": ["workspace_id", "keyword"],
            "additionalProperties": False,
        },
    )


def _tool_call_response() -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(
            role="assistant",
            tool_calls=(
                ModelToolCall(
                    id="call_search_1",
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


def test_fake_model_returns_scripted_responses_in_order() -> None:
    first_response = _tool_call_response()
    second_response = _final_response()
    client = FakeModelClient([first_response, second_response])

    first = client.complete(
        messages=[ModelMessage(role="user", content="查找报告")],
        tools=[_search_tool_definition()],
    )
    second = client.complete(
        messages=[first.message],
        tools=[_search_tool_definition()],
    )

    assert first == first_response
    assert second == second_response
    assert client.remaining_responses == 0


def test_fake_model_records_immutable_call_snapshots() -> None:
    messages = [ModelMessage(role="user", content="查找报告")]
    tools = [_search_tool_definition()]
    client = FakeModelClient([_tool_call_response()])

    client.complete(messages=messages, tools=tools)
    messages.append(ModelMessage(role="user", content="后来添加的消息"))
    tools.clear()

    assert len(client.calls) == 1
    assert [message.content for message in client.calls[0].messages] == [
        "查找报告"
    ]
    assert [tool.name for tool in client.calls[0].tools] == ["search_files"]


def test_fake_model_reports_an_unexpected_extra_call() -> None:
    client = FakeModelClient([])

    with pytest.raises(
        FakeModelResponsesExhaustedError,
        match="没有剩余的预设响应",
    ):
        client.complete(
            messages=[ModelMessage(role="user", content="查找报告")],
            tools=[],
        )

    assert len(client.calls) == 1


def test_fake_model_satisfies_the_model_client_protocol() -> None:
    client = FakeModelClient([_final_response()])

    assert isinstance(client, ModelClient)
