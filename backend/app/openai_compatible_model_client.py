"""DeepSeek、OpenAI 与 Google 共用的 OpenAI 兼容模型适配器。"""

import json
from collections.abc import Mapping, Sequence
from time import perf_counter
from types import MappingProxyType
from typing import Any, Protocol, cast

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from .model_client import (
    ModelCallMetrics,
    ModelClientRequestError,
    ModelMessage,
    ModelRequestErrorCode,
    ModelResponse,
    ModelTokenPricing,
    ModelTokenUsage,
    ModelToolCall,
    estimate_model_cost_usd,
)
from .model_settings import ModelSettings
from .tool_registry import ToolDefinition


class UnsupportedModelProviderError(ValueError):
    """配置的供应商不属于已审核的 OpenAI 兼容地址。"""


class ModelProviderRequestError(ModelClientRequestError):
    """供应商请求失败，且不向上层公开 SDK 内部信息。"""

    def __init__(
        self,
        *,
        code: ModelRequestErrorCode,
        retryable: bool,
    ) -> None:
        super().__init__(
            code=code,
            message="模型供应商请求失败",
            retryable=retryable,
        )


class InvalidModelProviderResponseError(RuntimeError):
    """供应商响应无法转换为统一模型协议。"""


class _CompatibleSdkClient(Protocol):
    chat: Any


PROVIDER_BASE_URLS: Mapping[str, str | None] = MappingProxyType(
    {
        "deepseek": "https://api.deepseek.com",
        "openai": None,
        "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    }
)
MAX_OUTPUT_TOKENS = 512


class OpenAICompatibleModelClient:
    """将 OpenAI 兼容 SDK 的请求和响应转换为 FileNest 统一契约。"""

    def __init__(
        self,
        settings: ModelSettings,
        *,
        sdk_client: _CompatibleSdkClient | None = None,
        token_pricing: ModelTokenPricing | None = None,
    ) -> None:
        provider = settings.provider.casefold()
        if provider not in PROVIDER_BASE_URLS:
            raise UnsupportedModelProviderError(
                f"不支持的模型供应商: {settings.provider}"
            )

        self._model_provider = provider
        self._model_name = settings.name
        self._client = sdk_client or _build_sdk_client(settings, provider)
        self._token_pricing = token_pricing

    def complete(
        self,
        *,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        """调用已选择的供应商，并返回统一助手响应。"""

        request: dict[str, Any] = {
            "model": self._model_name,
            "messages": [_message_payload(message) for message in messages],
            "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": False,
        }
        if tools:
            request["tools"] = [_tool_payload(tool) for tool in tools]

        started_at = perf_counter()
        try:
            response = self._client.chat.completions.create(**request)
        except APITimeoutError:
            raise ModelProviderRequestError(
                code="model_timeout",
                retryable=True,
            ) from None
        except APIConnectionError:
            raise ModelProviderRequestError(
                code="model_connection_error",
                retryable=True,
            ) from None
        except RateLimitError:
            raise ModelProviderRequestError(
                code="model_rate_limited",
                retryable=True,
            ) from None
        except APIStatusError as error:
            is_server_error = error.status_code >= 500
            raise ModelProviderRequestError(
                code=(
                    "model_server_error"
                    if is_server_error
                    else "model_request_rejected"
                ),
                retryable=is_server_error,
            ) from None
        except OpenAIError:
            # 未知 SDK 异常默认不重试，避免错误配置触发重复外部请求。
            raise ModelProviderRequestError(
                code="model_provider_error",
                retryable=False,
            ) from None

        latency_ms = (perf_counter() - started_at) * 1000
        return _model_response(
            response,
            latency_ms=latency_ms,
            token_pricing=self._token_pricing,
            model_provider=self._model_provider,
            model_name=self._model_name,
        )


def _build_sdk_client(
    settings: ModelSettings,
    provider: str,
) -> _CompatibleSdkClient:
    api_key = settings.api_key.get_secret_value()
    base_url = PROVIDER_BASE_URLS[provider]

    if base_url is None:
        return cast(
            _CompatibleSdkClient,
            OpenAI(api_key=api_key, max_retries=0, timeout=30.0),
        )

    return cast(
        _CompatibleSdkClient,
        OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=30.0,
        ),
    )


def _message_payload(message: ModelMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role}

    if message.content is not None:
        payload["content"] = message.content
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(
                        tool_call.arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
            for tool_call in message.tool_calls
        ]

    return payload


def _tool_payload(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _model_response(
    response: object,
    *,
    latency_ms: float,
    token_pricing: ModelTokenPricing | None,
    model_provider: str,
    model_name: str,
) -> ModelResponse:
    try:
        choices = getattr(response, "choices")
        if not choices:
            raise InvalidModelProviderResponseError("模型供应商未返回候选响应")

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason")
        if finish_reason not in {"stop", "tool_calls"}:
            raise InvalidModelProviderResponseError("模型供应商返回了不支持的结束原因")

        sdk_message = getattr(choice, "message")
        content = getattr(sdk_message, "content", None)
        if content is not None and not isinstance(content, str):
            raise InvalidModelProviderResponseError("模型供应商返回了非文本内容")

        tool_calls = tuple(
            _model_tool_call(tool_call)
            for tool_call in (getattr(sdk_message, "tool_calls", None) or [])
        )
        token_usage = _model_token_usage(response)
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
            ),
            finish_reason=finish_reason,
            model_provider=model_provider,
            model_name=model_name,
            metrics=ModelCallMetrics(
                latency_ms=latency_ms,
                requested_max_output_tokens=MAX_OUTPUT_TOKENS,
                token_usage=token_usage,
                estimated_cost_usd=estimate_model_cost_usd(
                    token_usage,
                    token_pricing,
                ),
            ),
        )
    except InvalidModelProviderResponseError:
        raise
    except (AttributeError, IndexError, TypeError, ValueError):
        raise InvalidModelProviderResponseError(
            "模型供应商响应不符合预期结构"
        ) from None


def _model_token_usage(response: object) -> ModelTokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    return ModelTokenUsage(
        input_tokens=getattr(usage, "prompt_tokens"),
        output_tokens=getattr(usage, "completion_tokens"),
        total_tokens=getattr(usage, "total_tokens"),
    )


def _model_tool_call(tool_call: object) -> ModelToolCall:
    if getattr(tool_call, "type", None) != "function":
        raise InvalidModelProviderResponseError("模型供应商返回了不支持的工具类型")

    function = getattr(tool_call, "function")
    arguments = json.loads(getattr(function, "arguments"))
    if not isinstance(arguments, dict):
        raise InvalidModelProviderResponseError("模型工具参数必须是 JSON 对象")

    return ModelToolCall(
        id=getattr(tool_call, "id"),
        name=getattr(function, "name"),
        arguments=arguments,
    )
