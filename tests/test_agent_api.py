from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_api import (
    AgentRunResponse,
    ReadOnlyAgentRunExecutor,
    get_agent_run_executor,
)
from backend.app.database import Base, get_session
from backend.app.fake_model_client import FakeModelClient
from backend.app.main import app
from backend.app.model_client import ModelMessage, ModelResponse, ModelToolCall
from backend.app.models import AgentRun, FileEntry, Workspace
from backend.app.tool_contracts import ToolResult


@pytest.fixture
def agent_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, sessionmaker[Session]]]:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'agent-api.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    def override_get_session() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    with TestClient(app) as client:
        yield client, session_factory

    app.dependency_overrides.clear()
    engine.dispose()


def _seed_workspace(
    session_factory: sessionmaker[Session],
    *,
    name: str,
) -> tuple[int, int]:
    with session_factory() as session:
        workspace = Workspace(name=name, root_path=f"D:/Test/{name}")
        session.add(workspace)
        session.flush()
        file_entry = FileEntry(
            workspace_id=workspace.id,
            relative_path="reports/quarterly.txt",
            name="quarterly.txt",
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


def test_agent_run_api_returns_read_only_answer_and_run_id(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id, _ = _seed_workspace(session_factory, name="只读请求")
    model_client = FakeModelClient(
        [
            _tool_call_response(
                call_id="call_search",
                name="search_files",
                arguments={
                    "workspace_id": workspace_id,
                    "keyword": "quarterly",
                },
            ),
            _final_response("找到季度报告。"),
        ]
    )
    app.dependency_overrides[get_agent_run_executor] = lambda: (
        ReadOnlyAgentRunExecutor(lambda: model_client)
    )

    response = client.post(
        "/api/v1/agent-runs",
        json={
            "workspace_id": workspace_id,
            "request_text": "  查找季度报告  ",
        },
    )

    assert response.status_code == 200
    assert AgentRunResponse.model_validate(response.json()).model_dump() == {
        "run_id": 1,
        "status": "completed",
        "final_answer": "找到季度报告。",
        "error": None,
        "sources": (
            {
                "workspace_id": workspace_id,
                "file_id": 1,
                "name": "quarterly.txt",
                "relative_path": "reports/quarterly.txt",
            },
        ),
    }
    assert model_client.calls[0].messages[-1].content == "查找季度报告"
    with session_factory() as session:
        persisted_run = session.get(AgentRun, 1)
        assert persisted_run is not None
        assert persisted_run.status == "completed"


def test_agent_run_api_rejects_missing_workspace_before_model_call(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, _ = agent_client

    class UnexpectedExecutor:
        def run(self, *args: object, **kwargs: object) -> AgentRunResponse:
            raise AssertionError("missing workspace must not call the Agent")

    app.dependency_overrides[get_agent_run_executor] = UnexpectedExecutor

    response = client.post(
        "/api/v1/agent-runs",
        json={"workspace_id": 999, "request_text": "查找报告"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": {
            "code": "workspace_not_found",
            "message": "工作区不存在。",
        }
    }


def test_agent_run_api_rejects_model_access_to_another_workspace(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    allowed_workspace_id, _ = _seed_workspace(
        session_factory,
        name="允许工作区",
    )
    other_workspace_id, _ = _seed_workspace(
        session_factory,
        name="其他工作区",
    )
    model_client = FakeModelClient(
        [
            _tool_call_response(
                call_id="call_out_of_scope",
                name="search_files",
                arguments={
                    "workspace_id": other_workspace_id,
                    "keyword": "quarterly",
                },
            ),
            _final_response("请求被限制在已授权工作区。"),
        ]
    )
    app.dependency_overrides[get_agent_run_executor] = lambda: (
        ReadOnlyAgentRunExecutor(lambda: model_client)
    )

    response = client.post(
        "/api/v1/agent-runs",
        json={
            "workspace_id": allowed_workspace_id,
            "request_text": "查找其他工作区的报告",
        },
    )

    returned_tool_message = model_client.calls[1].messages[-1]
    tool_result = ToolResult.model_validate_json(returned_tool_message.content)
    assert response.status_code == 200
    assert tool_result.ok is False
    assert tool_result.error is not None
    assert tool_result.error.code == "invalid_arguments"
    assert response.json()["sources"] == []
