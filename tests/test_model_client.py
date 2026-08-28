from collections.abc import Sequence
from decimal import Decimal

import pytest
from pydantic import ValidationError

from backend.app.model_client import (
    COST_CALCULATION_VERSION,
    ModelClient,
    ModelMessage,
    ModelResponse,
    ModelTokenPricing,
    ModelTokenUsage,
    ModelToolCall,
    estimate_model_cost_usd,
)
from backend.app.tool_registry import ToolDefinition


def _search_tool_call() -> ModelToolCall:
    return ModelToolCall(
        id="call_search_1",
        name="search_files",
        arguments={"workspace_id": 1, "keyword": "report"},
    )


def test_message_contract_supports_a_tool_call_round_trip() -> None:
    assistant_message = ModelMessage(
        role="assistant",
        tool_calls=(_search_tool_call(),),
    )
    tool_message = ModelMessage(
        role="tool",
        content='{"ok":true,"data":{"items":[]}}',
        tool_call_id="call_search_1",
    )

    assert assistant_message.tool_calls[0].name == "search_files"
    assert tool_message.tool_call_id == assistant_message.tool_calls[0].id


@pytest.mark.parametrize(
    "payload",
    [
        {"role": "user", "content": "   "},
        {
            "role": "user",
            "content": "查找报告",
            "tool_calls": [_search_tool_call().model_dump()],
        },
        {"role": "assistant"},
        {
            "role": "assistant",
            "content": "完成",
            "tool_call_id": "call_search_1",
        },
        {"role": "tool", "content": "{}"},
        {
            "role": "tool",
            "content": "{}",
            "tool_call_id": "call_search_1",
            "tool_calls": [_search_tool_call().model_dump()],
        },
    ],
)
def test_message_contract_rejects_contradictory_role_states(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ModelMessage.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "message": {"role": "user", "content": "查找报告"},
            "finish_reason": "stop",
        },
        {
            "message": {"role": "assistant", "content": "需要调用工具"},
            "finish_reason": "tool_calls",
        },
        {
            "message": {
                "role": "assistant",
                "tool_calls": [_search_tool_call().model_dump()],
            },
            "finish_reason": "stop",
        },
    ],
)
def test_response_rejects_inconsistent_finish_reason(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ModelResponse.model_validate(payload)


def test_structural_client_can_receive_vendor_independent_contracts() -> None:
    class StubClient:
        def complete(
            self,
            *,
            messages: Sequence[ModelMessage],
            tools: Sequence[ToolDefinition],
        ) -> ModelResponse:
            assert messages == [ModelMessage(role="user", content="你好")]
            assert tools == []
            return ModelResponse(
                message=ModelMessage(role="assistant", content="你好"),
                finish_reason="stop",
            )

    client: ModelClient = StubClient()

    response = client.complete(
        messages=[ModelMessage(role="user", content="你好")],
        tools=[],
    )

    assert isinstance(client, ModelClient)
    assert response.message.content == "你好"


def test_model_cost_calculation_is_versioned_and_uses_observed_usage() -> None:
    assert COST_CALCULATION_VERSION == "token-pricing-v1"
    assert estimate_model_cost_usd(
        ModelTokenUsage(
            input_tokens=1_000,
            output_tokens=200,
            total_tokens=1_200,
        ),
        ModelTokenPricing(
            input_usd_per_million_tokens=1.25,
            output_usd_per_million_tokens=2.50,
        ),
    ) == Decimal("0.00175")
    assert estimate_model_cost_usd(None, None) is None
