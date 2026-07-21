from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import Workspace
from backend.app.tool_contracts import Tool, ToolResult
from backend.app.tool_registry import ToolRegistry, build_read_tool_registry


class EmptyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'tool-registry.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _fake_tool(name: str, handler_called: list[bool] | None = None) -> Tool:
    def handle(_: BaseModel) -> ToolResult:
        if handler_called is not None:
            handler_called.append(True)
        return ToolResult.success({"tool": name})

    return Tool(
        name=name,
        description=f"测试工具 {name}",
        arguments_model=EmptyArguments,
        handler=handle,
    )


def test_read_tool_registry_contains_only_formal_read_tools() -> None:
    with Session() as session:
        registry = build_read_tool_registry(session)

    assert registry.names == (
        "list_workspaces",
        "search_files",
        "get_file_metadata",
    )


def test_registry_definitions_expose_schema_but_not_handlers() -> None:
    with Session() as session:
        registry = build_read_tool_registry(session)

    definitions = [definition.model_dump() for definition in registry.definitions()]

    assert [definition["name"] for definition in definitions] == [
        "list_workspaces",
        "search_files",
        "get_file_metadata",
    ]
    assert all(
        definition["parameters"]["additionalProperties"] is False
        for definition in definitions
    )
    assert all("handler" not in definition for definition in definitions)


def test_registry_dispatches_registered_list_workspaces_tool(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        session.add(
            Workspace(
                name="注册表工作区",
                root_path="D:/Private/RegistryWorkspace",
            )
        )
        session.commit()
        registry = build_read_tool_registry(session)

        result = registry.invoke("list_workspaces", {})

    assert result.ok is True
    assert result.data == {
        "items": [{"id": 1, "name": "注册表工作区"}],
        "count": 1,
    }
    assert "root_path" not in result.model_dump_json()


def test_registry_validates_registered_call_without_running_handler() -> None:
    handler_calls: list[bool] = []
    registry = ToolRegistry([_fake_tool("safe_tool", handler_calls)])

    result = registry.validate("safe_tool", {})

    assert result == ToolResult.success()
    assert handler_calls == []


def test_registry_rejects_unregistered_write_tool_during_validation() -> None:
    handler_calls: list[bool] = []
    registry = ToolRegistry([_fake_tool("safe_tool", handler_calls)])

    result = registry.validate("delete_file", {"path": "private.txt"})

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "unknown_tool"
    assert handler_calls == []


def test_registry_rejects_invalid_arguments_without_running_handler() -> None:
    handler_calls: list[bool] = []
    registry = ToolRegistry([_fake_tool("safe_tool", handler_calls)])

    result = registry.validate("safe_tool", {"unexpected": True})

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "invalid_arguments"
    assert handler_calls == []


@pytest.mark.parametrize("unknown_name", ["delete_all_files", "", 123, None])
def test_registry_rejects_unknown_tool_without_calling_handler(
    unknown_name: object,
) -> None:
    handler_calls: list[bool] = []
    registry = ToolRegistry([_fake_tool("safe_tool", handler_calls)])

    result = registry.invoke(unknown_name, {})

    assert result.model_dump() == {
        "ok": False,
        "data": None,
        "error": {
            "code": "unknown_tool",
            "message": "请求的工具未注册",
            "details": {},
        },
    }
    assert handler_calls == []


def test_registry_rejects_duplicate_tool_names() -> None:
    with pytest.raises(ValueError, match="duplicate tool name: same_tool"):
        ToolRegistry([_fake_tool("same_tool"), _fake_tool("same_tool")])


def test_registry_keeps_registered_tool_argument_validation() -> None:
    registry = ToolRegistry([_fake_tool("safe_tool")])

    result = registry.invoke("safe_tool", {"unknown": True})

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "invalid_arguments"
