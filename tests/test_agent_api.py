from collections.abc import Iterator
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_api import (
    _stream_agent_run_events,
    AgentRunResponse,
    ReadOnlyAgentRunExecutor,
    get_agent_run_executor,
)
from backend.app.agent_observability import SqlAlchemyAgentRunRecorder
from backend.app.database import Base, get_session
from backend.app.document_chunker import chunk_document
from backend.app.document_contracts import Document
from backend.app.fake_model_client import FakeModelClient
from backend.app.main import app
from backend.app.model_client import ModelMessage, ModelResponse, ModelToolCall
from backend.app.models import (
    AgentRun,
    AgentToolCall,
    ChunkRecord,
    DocumentRecord,
    FileEntry,
    Workspace,
)
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


def _seed_knowledge_document(
    session_factory: sessionmaker[Session],
    *,
    workspace_id: int,
    file_id: int,
) -> None:
    text = "第一行：审批流程\n第二行：批准后才能移动文件。"
    with session_factory() as session:
        document = Document(
            document_id=uuid4(),
            workspace_id=workspace_id,
            file_entry_id=file_id,
            source_relative_path="reports/quarterly.txt",
            source_format="text",
            normalized_text=text,
            source_version="a" * 64,
            source_updated_at="2026-09-01T00:00:00+00:00",
        )
        session.add(DocumentRecord.from_contract(document))
        session.add_all(
            ChunkRecord.from_contract(chunk)
            for chunk in chunk_document(document)
        )
        session.commit()


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
                "start_line": None,
                "end_line": None,
                "start_offset": None,
                "end_offset": None,
            },
        ),
    }
    assert model_client.calls[0].messages[-1].content == "查找季度报告"
    with session_factory() as session:
        persisted_run = session.get(AgentRun, 1)
        assert persisted_run is not None
        assert persisted_run.status == "completed"


def test_agent_run_api_returns_knowledge_source_location(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id, file_id = _seed_workspace(
        session_factory,
        name="知识出处请求",
    )
    _seed_knowledge_document(
        session_factory,
        workspace_id=workspace_id,
        file_id=file_id,
    )
    model_client = FakeModelClient(
        [
            _tool_call_response(
                call_id="call_knowledge_search",
                name="knowledge_search",
                arguments={
                    "workspace_id": workspace_id,
                    "query": "审批",
                },
            ),
            _final_response("文档显示需要先完成审批。"),
        ]
    )
    app.dependency_overrides[get_agent_run_executor] = lambda: (
        ReadOnlyAgentRunExecutor(lambda: model_client)
    )

    response = client.post(
        "/api/v1/agent-runs",
        json={
            "workspace_id": workspace_id,
            "request_text": "审批流程是什么？",
        },
    )

    assert response.status_code == 200
    result = AgentRunResponse.model_validate(response.json())
    assert result.final_answer == "文档显示需要先完成审批。"
    assert len(result.sources) == 1
    source = result.sources[0]
    assert source.workspace_id == workspace_id
    assert source.file_id == file_id
    assert source.name == "quarterly.txt"
    assert source.relative_path == "reports/quarterly.txt"
    assert (source.start_line, source.end_line) == (1, 2)
    assert source.start_offset == 0
    assert source.end_offset == len("第一行：审批流程\n第二行：批准后才能移动文件。")


def test_agent_run_api_refuses_knowledge_answer_without_evidence(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id, _ = _seed_workspace(
        session_factory,
        name="无知识证据请求",
    )
    model_client = FakeModelClient(
        [
            _tool_call_response(
                call_id="call_empty_knowledge_search",
                name="knowledge_search",
                arguments={
                    "workspace_id": workspace_id,
                    "query": "不存在的审批证据",
                },
            ),
            _final_response("根据常识可以直接回答。"),
        ]
    )
    app.dependency_overrides[get_agent_run_executor] = lambda: (
        ReadOnlyAgentRunExecutor(lambda: model_client)
    )

    response = client.post(
        "/api/v1/agent-runs",
        json={
            "workspace_id": workspace_id,
            "request_text": "这个问题应该怎么回答？",
        },
    )

    assert response.status_code == 200
    result = AgentRunResponse.model_validate(response.json())
    assert result.status == "completed"
    assert result.final_answer == "没有足够的文档证据，无法回答该问题。"
    assert result.sources == ()
    assert "根据常识可以直接回答。" not in response.text


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


def test_agent_run_events_stream_persisted_statuses_with_ordered_ids(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    started_at = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
    finished_at = datetime(2026, 8, 31, 12, 1, tzinfo=timezone.utc)

    with session_factory() as session:
        agent_run = AgentRun(
            status="completed",
            started_at=started_at,
            finished_at=finished_at,
            model_turns=2,
        )
        session.add(agent_run)
        session.flush()
        session.add(
            AgentToolCall(
                agent_run_id=agent_run.id,
                sequence_no=1,
                model_call_id="call_1",
                tool_name="search_files",
                status="succeeded",
                started_at=started_at,
                finished_at=finished_at,
            )
        )
        session.commit()
        run_id = agent_run.id

    response = client.get(f"/api/v1/agent-runs/{run_id}/events")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    blocks = [block for block in response.text.split("\n\n") if block]
    assert len(blocks) == 4
    event_lines = [block.splitlines() for block in blocks]
    assert [lines[0] for lines in event_lines] == [
        "id: 1",
        "id: 2",
        "id: 3",
        "id: 4",
    ]
    assert [lines[1] for lines in event_lines] == [
        "event: agent_run.status_changed",
        "event: agent_tool_call.status_changed",
        "event: agent_tool_call.status_changed",
        "event: agent_run.status_changed",
    ]
    payloads = [json.loads(lines[2].removeprefix("data: ")) for lines in event_lines]
    assert [payload["status"] for payload in payloads] == [
        "running",
        "requested",
        "succeeded",
        "completed",
    ]
    assert payloads[-1]["run_id"] == run_id


def test_agent_run_recovery_fetches_state_and_resumes_after_last_event_id(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    started_at = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
    finished_at = datetime(2026, 8, 31, 12, 1, tzinfo=timezone.utc)

    with session_factory() as session:
        agent_run = AgentRun(
            status="completed",
            started_at=started_at,
            finished_at=finished_at,
            model_turns=2,
        )
        session.add(agent_run)
        session.flush()
        session.add(
            AgentToolCall(
                agent_run_id=agent_run.id,
                sequence_no=1,
                model_call_id="call_1",
                tool_name="search_files",
                status="succeeded",
                started_at=started_at,
                finished_at=finished_at,
            )
        )
        session.commit()
        run_id = agent_run.id

    state_response = client.get(f"/api/v1/agent-runs/{run_id}")
    event_response = client.get(
        f"/api/v1/agent-runs/{run_id}/events",
        headers={"Last-Event-ID": "2"},
    )

    assert state_response.status_code == 200
    assert state_response.json() == {
        "run_id": run_id,
        "status": "completed",
        "model_turns": 2,
        "error_code": None,
    }
    blocks = [block for block in event_response.text.split("\n\n") if block]
    event_lines = [block.splitlines() for block in blocks]
    assert event_response.status_code == 200
    assert [lines[0] for lines in event_lines] == ["id: 3", "id: 4"]
    payloads = [
        json.loads(lines[2].removeprefix("data: ")) for lines in event_lines
    ]
    assert [payload["status"] for payload in payloads] == [
        "succeeded",
        "completed",
    ]


def test_agent_run_events_rejects_unknown_run(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, _ = agent_client

    response = client.get("/api/v1/agent-runs/999/events")

    assert response.status_code == 404
    assert response.json() == {
        "detail": {
            "code": "agent_run_not_found",
            "message": "Agent 运行记录不存在。",
        }
    }


def test_agent_run_stream_disconnect_does_not_lose_background_run(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    started_at = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
    finished_at = datetime(2026, 8, 31, 12, 1, tzinfo=timezone.utc)

    with session_factory() as session:
        recorder = SqlAlchemyAgentRunRecorder(
            session,
            clock=lambda: started_at,
        )
        run_id = recorder.start_run()

    with session_factory() as stream_session:
        event_stream = _stream_agent_run_events(stream_session, run_id)
        first_event = next(event_stream)
        event_stream.close()

    with session_factory() as session:
        persisted_run = session.get(AgentRun, run_id)
        assert persisted_run is not None
        assert persisted_run.status == "running"

    with session_factory() as session:
        recorder = SqlAlchemyAgentRunRecorder(
            session,
            clock=lambda: finished_at,
        )
        recorder.finish_run(
            agent_run_id=run_id,
            status="completed",
            model_turns=1,
            error_code=None,
        )

    state_response = client.get(f"/api/v1/agent-runs/{run_id}")
    first_payload = json.loads(first_event.split("data: ", 1)[1])

    assert first_payload["status"] == "running"
    assert state_response.status_code == 200
    assert state_response.json() == {
        "run_id": run_id,
        "status": "completed",
        "model_turns": 1,
        "error_code": None,
    }
