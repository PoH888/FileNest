from typing import cast

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.app.tool_contracts import Tool, ToolResult


class SearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str = Field(min_length=1)
    limit: int = Field(default=20, ge=1, le=50)


def test_tool_validates_arguments_and_returns_success_result() -> None:
    def handler(arguments: BaseModel) -> ToolResult:
        search = cast(SearchArguments, arguments)
        return ToolResult.success(
            {"keyword": search.keyword, "limit": search.limit}
        )

    tool = Tool(
        name="search_files",
        description="搜索已授权工作区中的文件索引",
        arguments_model=SearchArguments,
        handler=handler,
    )

    result = tool.invoke({"keyword": "report", "limit": 10})

    assert result.model_dump() == {
        "ok": True,
        "data": {"keyword": "report", "limit": 10},
        "error": None,
    }
    assert tool.input_schema()["additionalProperties"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        {"keyword": "report", "limit": 0},
        {"keyword": "report", "unknown": True},
        ["report"],
    ],
)
def test_tool_rejects_invalid_arguments_before_calling_handler(
    arguments: object,
) -> None:
    handler_called = False

    def handler(_: BaseModel) -> ToolResult:
        nonlocal handler_called
        handler_called = True
        return ToolResult.success([])

    tool = Tool(
        name="search_files",
        description="搜索文件",
        arguments_model=SearchArguments,
        handler=handler,
    )

    result = tool.invoke(arguments)

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "invalid_arguments"
    assert result.error.details["errors"]
    assert handler_called is False


def test_validation_error_does_not_echo_argument_values() -> None:
    tool = Tool(
        name="search_files",
        description="搜索文件",
        arguments_model=SearchArguments,
        handler=lambda _: ToolResult.success([]),
    )

    result = tool.invoke({"keyword": "private-name.txt", "limit": 0})

    assert "private-name.txt" not in result.model_dump_json()


def test_tool_converts_unexpected_exception_to_safe_failure() -> None:
    def handler(_: BaseModel) -> ToolResult:
        raise RuntimeError("D:/Secret/private.db")

    tool = Tool(
        name="search_files",
        description="搜索文件",
        arguments_model=SearchArguments,
        handler=handler,
    )

    result = tool.invoke({"keyword": "report"})

    assert result.model_dump() == {
        "ok": False,
        "data": None,
        "error": {
            "code": "tool_execution_failed",
            "message": "工具执行失败",
            "details": {"tool": "search_files"},
        },
    }
    assert "private.db" not in result.model_dump_json()


def test_tool_converts_non_contract_result_to_failure() -> None:
    tool = Tool(
        name="search_files",
        description="搜索文件",
        arguments_model=SearchArguments,
        handler=lambda _: cast(ToolResult, ["report.txt"]),
    )

    result = tool.invoke({"keyword": "report"})

    assert result.error is not None
    assert result.error.code == "invalid_tool_result"


@pytest.mark.parametrize(
    "payload",
    [
        {"ok": True, "error": {"code": "failed", "message": "失败"}},
        {"ok": False},
        {
            "ok": False,
            "data": [],
            "error": {"code": "failed", "message": "失败"},
        },
    ],
)
def test_tool_result_rejects_contradictory_states(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ToolResult.model_validate(payload)


def test_tool_requires_arguments_model_to_reject_unknown_fields() -> None:
    class PermissiveArguments(BaseModel):
        keyword: str

    with pytest.raises(
        ValidationError,
        match="tool arguments model must set extra='forbid'",
    ):
        Tool(
            name="search_files",
            description="搜索文件",
            arguments_model=PermissiveArguments,
            handler=lambda _: ToolResult.success([]),
        )
