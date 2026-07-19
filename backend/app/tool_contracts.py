"""Agent 只读工具的统一调用与结果契约。"""

from collections.abc import Callable
import logging
from typing import Any, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)


logger = logging.getLogger(__name__)


class ToolError(BaseModel):
    """可安全返回给 Agent 的结构化工具错误。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    message: str = Field(min_length=1)
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("code", "message")
    @classmethod
    def reject_surrounding_whitespace(cls, value: str) -> str:
        """稳定错误码和消息，避免调用方依赖含糊的空白差异。"""

        if value != value.strip():
            raise ValueError("must not contain surrounding whitespace")
        return value


class ToolResult(BaseModel):
    """所有工具都必须返回的成功或失败信封。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    data: JsonValue | None = None
    error: ToolError | None = None

    @model_validator(mode="after")
    def validate_result_state(self) -> "ToolResult":
        """保证成功和失败状态互斥，避免 Agent 猜测结果含义。"""

        if self.ok and self.error is not None:
            raise ValueError("successful tool result must not contain an error")
        if not self.ok and self.error is None:
            raise ValueError("failed tool result must contain an error")
        if not self.ok and self.data is not None:
            raise ValueError("failed tool result must not contain data")
        return self

    @classmethod
    def success(cls, data: JsonValue = None) -> "ToolResult":
        """创建成功结果。"""

        return cls(ok=True, data=data)

    @classmethod
    def failure(
        cls,
        *,
        code: str,
        message: str,
        details: dict[str, JsonValue] | None = None,
    ) -> "ToolResult":
        """创建不携带业务数据的失败结果。"""

        return cls(
            ok=False,
            error=ToolError(
                code=code,
                message=message,
                details=details or {},
            ),
        )


ToolHandler = Callable[[BaseModel], ToolResult]


class Tool(BaseModel):
    """工具名称、参数模型和执行函数组成的统一只读工具契约。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    arguments_model: type[BaseModel]
    handler: ToolHandler

    @field_validator("name", "description")
    @classmethod
    def reject_surrounding_whitespace(cls, value: str) -> str:
        """工具标识必须稳定，说明也不能用纯空白蒙混通过。"""

        if value != value.strip():
            raise ValueError("must not contain surrounding whitespace")
        return value

    @model_validator(mode="after")
    def require_strict_arguments_model(self) -> "Tool":
        """未知参数必须被拒绝，不能在安全边界中静默忽略。"""

        if self.arguments_model.model_config.get("extra") != "forbid":
            raise ValueError("tool arguments model must set extra='forbid'")
        return self

    def input_schema(self) -> dict[str, JsonValue]:
        """返回后续模型客户端可使用的 JSON Schema。"""

        return cast(dict[str, JsonValue], self.arguments_model.model_json_schema())

    def invoke(self, arguments: object) -> ToolResult:
        """验证不可信参数，并把执行边界内的失败统一为安全结果。"""

        try:
            validated_arguments = self.arguments_model.model_validate(arguments)
        except ValidationError as error:
            return ToolResult.failure(
                code="invalid_arguments",
                message="工具参数不符合契约",
                details={"errors": _safe_validation_errors(error)},
            )

        try:
            result = self.handler(validated_arguments)
        except Exception:
            # 记录真实异常供程序排查，但不把路径、SQL 等内部细节返回给模型。
            logger.exception("Tool %s execution failed", self.name)
            return ToolResult.failure(
                code="tool_execution_failed",
                message="工具执行失败",
                details={"tool": self.name},
            )

        if not isinstance(result, ToolResult):
            logger.error("Tool %s returned an invalid result", self.name)
            return ToolResult.failure(
                code="invalid_tool_result",
                message="工具返回结果不符合契约",
                details={"tool": self.name},
            )

        return result


def _safe_validation_errors(error: ValidationError) -> list[JsonValue]:
    """只公开字段位置与错误类型，不回显可能敏感的原始参数。"""

    errors: list[JsonValue] = []
    for item in error.errors():
        errors.append(
            {
                "type": item["type"],
                "loc": [str(part) for part in item["loc"]],
                "message": item["msg"],
            }
        )
    return errors
