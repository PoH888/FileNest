from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAIError,
    RateLimitError,
)

import backend.app.openai_compatible_model_client as client_module
from backend.app.model_client import (
    ModelMessage,
    ModelTokenPricing,
    ModelToolCall,
)
from backend.app.model_settings import ModelSettings
from backend.app.openai_compatible_model_client import (
    InvalidModelProviderResponseError,
    ModelProviderRequestError,
    OpenAICompatibleModelClient,
    UnsupportedModelProviderError,
)
from backend.app.tool_registry import ToolDefinition


class StubCompletions:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> object:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response


class StubSdkClient:
    def __init__(self, completions: StubCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _settings(provider: str = "deepseek") -> ModelSettings:
    return ModelSettings(
        provider=provider,
        name="example-model",
        api_key="secret-for-test",
    )


def _text_response(content: str = "找到报告。", *, with_usage: bool = True) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content, tool_calls=None),
            )
        ],
        usage=(
            SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200,
            )
            if with_usage
            else None
        ),
    )


@pytest.mark.parametrize(
    ("provider", "expected_base_url"),
    [
        ("deepseek", "https://api.deepseek.com"),
        ("openai", None),
        ("google", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    ],
)
def test_client_uses_only_reviewed_provider_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    expected_base_url: str | None,
) -> None:
    captured: dict[str, Any] = {}

    def fake_openai(**options: Any) -> StubSdkClient:
        captured.update(options)
        return StubSdkClient(StubCompletions(_text_response()))

    monkeypatch.setattr(client_module, "OpenAI", fake_openai)

    OpenAICompatibleModelClient(_settings(provider))

    assert captured["api_key"] == "secret-for-test"
    assert captured.get("base_url") == expected_base_url
    assert captured["max_retries"] == 0
    assert captured["timeout"] == 30.0


def test_client_rejects_an_unknown_provider_before_creating_sdk_client() -> None:
    with pytest.raises(
        UnsupportedModelProviderError,
        match="不支持的模型供应商",
    ):
        OpenAICompatibleModelClient(_settings("unknown-provider"))


def test_client_converts_messages_tools_and_text_response() -> None:
    completions = StubCompletions(_text_response())
    client = OpenAICompatibleModelClient(
        _settings(),
        sdk_client=StubSdkClient(completions),
    )
    tool = ToolDefinition(
        name="search_files",
        description="搜索文件",
        parameters={"type": "object", "additionalProperties": False},
    )

    response = client.complete(
        messages=[ModelMessage(role="user", content="查找报告")],
        tools=[tool],
    )

    assert response.message.content == "找到报告。"
    assert response.finish_reason == "stop"
    assert response.metrics is not None
    assert response.metrics.token_usage is not None
    assert response.metrics.estimated_cost_usd is None
    assert completions.requests == [
        {
            "model": "example-model",
            "messages": [{"role": "user", "content": "查找报告"}],
            "max_tokens": 512,
            "stream": False,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_files",
                        "description": "搜索文件",
                        "parameters": {
                            "type": "object",
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        }
    ]


def test_client_records_latency_tokens_and_estimated_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([10.0, 10.25])
    monkeypatch.setattr(client_module, "perf_counter", lambda: next(timestamps))
    client = OpenAICompatibleModelClient(
        _settings(),
        sdk_client=StubSdkClient(StubCompletions(_text_response())),
        token_pricing=ModelTokenPricing(
            input_usd_per_million_tokens=Decimal("1.25"),
            output_usd_per_million_tokens=Decimal("2.50"),
        ),
    )

    response = client.complete(
        messages=[ModelMessage(role="user", content="查找报告")],
        tools=[],
    )

    assert response.metrics is not None
    assert response.metrics.latency_ms == 250.0
    assert response.metrics.requested_max_output_tokens == 512
    assert response.metrics.token_usage is not None
    assert response.metrics.token_usage.model_dump() == {
        "input_tokens": 1000,
        "output_tokens": 200,
        "total_tokens": 1200,
    }
    assert response.metrics.estimated_cost_usd == Decimal("0.00175")


def test_client_preserves_metrics_when_provider_omits_usage() -> None:
    client = OpenAICompatibleModelClient(
        _settings(),
        sdk_client=StubSdkClient(
            StubCompletions(_text_response(with_usage=False))
        ),
    )

    response = client.complete(
        messages=[ModelMessage(role="user", content="查找报告")],
        tools=[],
    )

    assert response.metrics is not None
    assert response.metrics.token_usage is None
    assert response.metrics.estimated_cost_usd is None


def test_client_converts_tool_call_response_and_history() -> None:
    sdk_tool_call = SimpleNamespace(
        id="call_search_1",
        type="function",
        function=SimpleNamespace(
            name="search_files",
            arguments='{"workspace_id":1,"keyword":"报告"}',
        ),
    )
    completions = StubCompletions(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[sdk_tool_call],
                    ),
                )
            ]
        )
    )
    client = OpenAICompatibleModelClient(
        _settings(),
        sdk_client=StubSdkClient(completions),
    )
    previous_call = ModelToolCall(
        id="call_previous",
        name="list_workspaces",
        arguments={},
    )

    response = client.complete(
        messages=[
            ModelMessage(role="assistant", tool_calls=(previous_call,)),
            ModelMessage(
                role="tool",
                content='{"ok":true}',
                tool_call_id="call_previous",
            ),
        ],
        tools=[],
    )

    assert response.message.tool_calls[0].model_dump() == {
        "id": "call_search_1",
        "name": "search_files",
        "arguments": {"workspace_id": 1, "keyword": "报告"},
    }
    assert completions.requests[0]["messages"] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_previous",
                    "type": "function",
                    "function": {
                        "name": "list_workspaces",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"ok":true}',
            "tool_call_id": "call_previous",
        },
    ]


def test_client_rejects_invalid_tool_arguments_from_provider() -> None:
    sdk_tool_call = SimpleNamespace(
        id="call_search_1",
        type="function",
        function=SimpleNamespace(name="search_files", arguments="not-json"),
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(content=None, tool_calls=[sdk_tool_call]),
            )
        ]
    )
    client = OpenAICompatibleModelClient(
        _settings(),
        sdk_client=StubSdkClient(StubCompletions(response)),
    )

    with pytest.raises(
        InvalidModelProviderResponseError,
        match="响应不符合预期结构",
    ):
        client.complete(messages=[ModelMessage(role="user", content="查找")], tools=[])


def test_client_hides_sdk_error_details() -> None:
    exposed_secret = "secret-that-must-stay-hidden"
    client = OpenAICompatibleModelClient(
        _settings(),
        sdk_client=StubSdkClient(
            StubCompletions(error=OpenAIError(exposed_secret))
        ),
    )

    with pytest.raises(ModelProviderRequestError) as error_info:
        client.complete(messages=[ModelMessage(role="user", content="你好")], tools=[])

    assert exposed_secret not in str(error_info.value)


@pytest.mark.parametrize(
    ("sdk_error_name", "expected_code", "expected_retryable"),
    [
        ("timeout", "model_timeout", True),
        ("connection", "model_connection_error", True),
        ("rate_limit", "model_rate_limited", True),
        ("server", "model_server_error", True),
        ("authentication", "model_request_rejected", False),
        ("provider", "model_provider_error", False),
    ],
)
def test_client_classifies_sdk_errors_without_exposing_details(
    sdk_error_name: str,
    expected_code: str,
    expected_retryable: bool,
) -> None:
    exposed_secret = f"secret-from-{sdk_error_name}"
    request = httpx.Request(
        "POST",
        "https://provider.example/v1/chat/completions",
    )
    status_errors: dict[str, APIStatusError] = {
        "rate_limit": RateLimitError(
            exposed_secret,
            response=httpx.Response(429, request=request),
            body={"message": exposed_secret},
        ),
        "server": APIStatusError(
            exposed_secret,
            response=httpx.Response(503, request=request),
            body={"message": exposed_secret},
        ),
        "authentication": APIStatusError(
            exposed_secret,
            response=httpx.Response(401, request=request),
            body={"message": exposed_secret},
        ),
    }
    sdk_errors: dict[str, OpenAIError] = {
        "timeout": APITimeoutError(request),
        "connection": APIConnectionError(
            message=exposed_secret,
            request=request,
        ),
        **status_errors,
        "provider": OpenAIError(exposed_secret),
    }
    client = OpenAICompatibleModelClient(
        _settings(),
        sdk_client=StubSdkClient(
            StubCompletions(error=sdk_errors[sdk_error_name])
        ),
    )

    with pytest.raises(ModelProviderRequestError) as error_info:
        client.complete(
            messages=[ModelMessage(role="user", content="你好")],
            tools=[],
        )

    assert error_info.value.code == expected_code
    assert error_info.value.retryable is expected_retryable
    assert exposed_secret not in str(error_info.value)
