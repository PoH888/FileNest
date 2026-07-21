from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from backend.app.agent_loop import AgentLoop
from backend.app.database import Base
from backend.app.fake_model_client import FakeModelClient
from backend.app.model_client import (
    ModelMessage,
    ModelResponse,
    ModelToolCall,
)
from backend.app.models import FileEntry, Workspace
from backend.app.tool_contracts import ToolResult
from backend.app.tool_registry import build_read_tool_registry


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'agent-loop-end-to-end.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _seed_file_index(session: Session) -> tuple[int, int]:
    workspace = Workspace(
        name="Agent 查询工作区",
        root_path="D:/Private/AgentLoop",
    )
    session.add(workspace)
    session.flush()

    file_entry = FileEntry(
        workspace_id=workspace.id,
        relative_path="reports/quarterly-report.txt",
        name="quarterly-report.txt",
        extension=".txt",
        size_bytes=128,
        mtime_ns=1_700_000_000_000_000_000,
    )
    session.add(file_entry)
    session.commit()
    return workspace.id, file_entry.id


def _tool_call_response(
    *,
    call_id: str,
    name: str,
    arguments: dict[str, object],
) -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(
            role="assistant",
            tool_calls=(
                ModelToolCall(
                    id=call_id,
                    name=name,
                    arguments=arguments,
                ),
            ),
        ),
        finish_reason="tool_calls",
    )


def _final_response(content: str) -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(role="assistant", content=content),
        finish_reason="stop",
    )


def test_agent_loop_runs_formal_read_only_file_search(engine: Engine) -> None:
    with Session(engine) as session:
        workspace_id, _ = _seed_file_index(session)
        model_client = FakeModelClient(
            [
                _tool_call_response(
                    call_id="call_search_1",
                    name="search_files",
                    arguments={
                        "workspace_id": workspace_id,
                        "keyword": "quarterly",
                    },
                ),
                _final_response("找到季度报告。"),
            ]
        )
        loop = AgentLoop(
            model_client=model_client,
            tool_registry=build_read_tool_registry(session),
        )

        result = loop.run(
            [ModelMessage(role="user", content="查找季度报告")],
            max_steps=3,
        )

    returned_tool_message = model_client.calls[1].messages[-1]
    tool_result = ToolResult.model_validate_json(returned_tool_message.content)
    assert result.status == "completed"
    assert result.final_answer == "找到季度报告。"
    assert returned_tool_message.tool_call_id == "call_search_1"
    assert tool_result.ok is True
    assert isinstance(tool_result.data, dict)
    assert tool_result.data["total"] == 1
    assert tool_result.data["items"][0]["relative_path"] == (
        "reports/quarterly-report.txt"
    )
    assert "root_path" not in returned_tool_message.content
    assert "D:/Private" not in returned_tool_message.content


def test_agent_loop_rejects_unregistered_write_tool(engine: Engine) -> None:
    with Session(engine) as session:
        workspace_id, file_id = _seed_file_index(session)
        registry = build_read_tool_registry(session)
        model_client = FakeModelClient(
            [
                _tool_call_response(
                    call_id="call_delete_1",
                    name="delete_file",
                    arguments={
                        "workspace_id": workspace_id,
                        "file_id": file_id,
                    },
                ),
                _final_response("写工具未获授权，文件没有被删除。"),
            ]
        )
        loop = AgentLoop(
            model_client=model_client,
            tool_registry=registry,
        )

        result = loop.run(
            [ModelMessage(role="user", content="删除季度报告")],
            max_steps=3,
        )

        returned_tool_message = model_client.calls[1].messages[-1]
        tool_result = ToolResult.model_validate_json(returned_tool_message.content)
        persisted_file = session.get(FileEntry, file_id)

    assert "delete_file" not in registry.names
    assert tool_result.ok is False
    assert tool_result.error is not None
    assert tool_result.error.code == "unknown_tool"
    assert returned_tool_message.tool_call_id == "call_delete_1"
    assert persisted_file is not None
    assert persisted_file.relative_path == "reports/quarterly-report.txt"
    assert result.status == "completed"
    assert result.final_answer == "写工具未获授权，文件没有被删除。"
