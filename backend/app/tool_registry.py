"""Agent 可见工具的程序侧白名单与分发入口。"""

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, JsonValue
from sqlalchemy.orm import Session

from .read_tools import (
    build_get_file_metadata_tool,
    build_list_workspaces_tool,
    build_search_files_tool,
)
from .tool_contracts import Tool, ToolResult


class ToolDefinition(BaseModel):
    """可交给模型客户端的供应商无关工具定义。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    parameters: dict[str, JsonValue]


class ToolRegistry:
    """只允许按精确名称调用启动时注册的工具。"""

    def __init__(self, tools: Iterable[Tool]) -> None:
        registered: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in registered:
                raise ValueError(f"duplicate tool name: {tool.name}")
            registered[tool.name] = tool

        # 注册完成后冻结映射，运行中的模型输出不能替换工具实现。
        self._tools: Mapping[str, Tool] = MappingProxyType(registered)

    @property
    def names(self) -> tuple[str, ...]:
        """按注册顺序返回允许调用的工具名称。"""

        return tuple(self._tools)

    def definitions(self) -> list[ToolDefinition]:
        """返回模型所需的说明和参数 Schema，不暴露 Python handler。"""

        return [
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.input_schema(),
            )
            for tool in self._tools.values()
        ]

    def validate(self, name: object, arguments: object) -> ToolResult:
        """校验工具白名单权限与参数，不执行工具处理函数。"""

        tool = self._tools.get(name) if isinstance(name, str) else None
        if tool is None:
            return ToolResult.failure(
                code="unknown_tool",
                message="请求的工具未注册",
            )

        return tool.validate_arguments(arguments)

    def invoke(self, name: object, arguments: object) -> ToolResult:
        """调用白名单工具；未知或非法名称一律拒绝。"""

        tool = self._tools.get(name) if isinstance(name, str) else None
        if tool is None:
            return ToolResult.failure(
                code="unknown_tool",
                message="请求的工具未注册",
            )

        return tool.invoke(arguments)


def build_read_tool_registry(session: Session) -> ToolRegistry:
    """为一个数据库会话构建 FileNest 正式只读工具白名单。"""

    return ToolRegistry(
        [
            build_list_workspaces_tool(session),
            build_search_files_tool(session),
            build_get_file_metadata_tool(session),
        ]
    )
