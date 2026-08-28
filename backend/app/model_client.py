"""供应商无关的模型消息、响应与客户端协议。"""

from collections.abc import Sequence
from decimal import Decimal
from typing import Literal, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from .tool_registry import ToolDefinition


ModelRole = Literal["system", "user", "assistant", "tool"]
ModelFinishReason = Literal["stop", "tool_calls"]
ModelRequestErrorCode = Literal[
    "model_timeout",
    "model_connection_error",
    "model_rate_limited",
    "model_server_error",
    "model_request_rejected",
    "model_provider_error",
]
COST_CALCULATION_VERSION = "token-pricing-v1"


class ModelClientRequestError(RuntimeError):
    """所有模型适配器向 Agent Loop 暴露的稳定请求错误。"""

    def __init__(
        self,
        *,
        code: ModelRequestErrorCode,
        message: str,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class ModelTokenPricing(BaseModel):
    """调用方显式提供的每百万 token 美元单价。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_usd_per_million_tokens: Decimal = Field(ge=0)
    output_usd_per_million_tokens: Decimal = Field(ge=0)


class ModelTokenUsage(BaseModel):
    """供应商返回的一次调用 token 用量。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ModelCallMetrics(BaseModel):
    """一次真实模型调用的可观察指标与费用估算边界。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    latency_ms: float = Field(ge=0)
    requested_max_output_tokens: int = Field(gt=0)
    token_usage: ModelTokenUsage | None = None
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)


def estimate_model_cost_usd(
    token_usage: ModelTokenUsage | None,
    token_pricing: ModelTokenPricing | None,
) -> Decimal | None:
    """按当前版本价目计算一次调用的美元成本；缺少依据时返回 None。"""

    if token_usage is None or token_pricing is None:
        return None

    input_cost = (
        Decimal(token_usage.input_tokens)
        * token_pricing.input_usd_per_million_tokens
    )
    output_cost = (
        Decimal(token_usage.output_tokens)
        * token_pricing.output_usd_per_million_tokens
    )
    return (input_cost + output_cost) / Decimal(1_000_000)


class ModelToolCall(BaseModel):
    """模型提出的一次供应商无关工具调用。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    arguments: dict[str, JsonValue]

    @field_validator("id", "name")
    @classmethod
    def reject_surrounding_whitespace(cls, value: str) -> str:
        """保持调用标识稳定，避免不同客户端产生含糊匹配。"""

        if value != value.strip():
            raise ValueError("must not contain surrounding whitespace")
        return value


class ModelMessage(BaseModel):
    """Agent Loop 与任意模型客户端之间传递的统一消息。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: ModelRole
    content: str | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()
    tool_call_id: str | None = None

    @field_validator("tool_call_id")
    @classmethod
    def validate_tool_call_id(cls, value: str | None) -> str | None:
        """工具结果必须使用非空且无首尾空白的调用标识。"""

        if value is not None and (not value or value != value.strip()):
            raise ValueError("must be non-empty without surrounding whitespace")
        return value

    @model_validator(mode="after")
    def validate_role_state(self) -> "ModelMessage":
        """限制各角色可携带的数据，防止客户端猜测矛盾消息。"""

        has_content = self.content is not None and bool(self.content.strip())

        if self.role in {"system", "user"}:
            if not has_content:
                raise ValueError(f"{self.role} message must contain text")
            if self.tool_calls or self.tool_call_id is not None:
                raise ValueError(
                    f"{self.role} message must not contain tool metadata"
                )

        if self.role == "assistant":
            if not has_content and not self.tool_calls:
                raise ValueError(
                    "assistant message must contain text or tool calls"
                )
            if self.tool_call_id is not None:
                raise ValueError(
                    "assistant message must not contain a tool call result id"
                )

        if self.role == "tool":
            if not has_content:
                raise ValueError("tool message must contain a serialized result")
            if self.tool_call_id is None:
                raise ValueError("tool message must contain a tool call id")
            if self.tool_calls:
                raise ValueError("tool message must not request tool calls")

        return self


class ModelResponse(BaseModel):
    """一次模型调用产生的统一助手响应。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: ModelMessage
    finish_reason: ModelFinishReason
    model_provider: str | None = None
    model_name: str | None = None
    metrics: ModelCallMetrics | None = None

    @field_validator("model_provider", "model_name")
    @classmethod
    def reject_blank_model_identity(cls, value: str | None) -> str | None:
        """模型身份必须是可解释且无首尾空格的标识。"""

        if value is not None and (not value or value != value.strip()):
            raise ValueError("model identity must be non-empty without whitespace")
        return value

    @model_validator(mode="after")
    def validate_response_state(self) -> "ModelResponse":
        """保证结束原因与助手消息内容一致。"""

        if self.message.role != "assistant":
            raise ValueError("model response message must use the assistant role")
        if self.finish_reason == "tool_calls" and not self.message.tool_calls:
            raise ValueError("tool_calls finish reason requires at least one call")
        if self.finish_reason == "stop" and self.message.tool_calls:
            raise ValueError("stop finish reason must not contain tool calls")
        return self


@runtime_checkable
class ModelClient(Protocol):
    """所有 Fake 或真实模型客户端都必须提供的最小同步接口。"""

    def complete(
        self,
        *,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        """根据完整对话历史和允许工具生成下一条助手响应。"""

        ...
