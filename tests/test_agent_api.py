from collections.abc import Iterator
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
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
from backend.app.model_client import (
    ModelClientRequestError,
    ModelMessage,
    ModelResponse,
    ModelToolCall,
)
from backend.app.models import (
    AgentRun,
    AgentToolCall,
    ChunkRecord,
    DocumentRecord,
    FileEntry,
    OperationPlanRecord,
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


def _wait_for_agent_run_completion(
    client: TestClient,
    run_id: int,
) -> object:
    terminal_statuses = {
        "completed",
        "max_steps_reached",
        "timed_out",
        "cancelled",
        "failed",
    }
    deadline = monotonic() + 5
    while True:
        response = client.get(f"/api/v1/agent-runs/{run_id}")
        assert response.status_code == 200
        if response.json()["status"] in terminal_statuses:
            return response
        if monotonic() >= deadline:
            pytest.fail("Agent Run did not reach a terminal state")
        sleep(0.01)


def _seed_disk_workspace(
    session_factory: sessionmaker[Session],
    workspace_root: Path,
    *,
    name: str,
    relative_path: str,
    contents: bytes,
) -> tuple[int, int, Path]:
    source_path = workspace_root / Path(relative_path)
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(contents)
    with session_factory() as session:
        workspace = Workspace(name=name, root_path=str(workspace_root))
        session.add(workspace)
        session.flush()
        metadata = source_path.stat()
        file_entry = FileEntry(
            workspace_id=workspace.id,
            relative_path=source_path.relative_to(workspace_root).as_posix(),
            name=source_path.name,
            extension=source_path.suffix,
            size_bytes=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
        )
        session.add(file_entry)
        session.commit()
        return workspace.id, file_entry.id, source_path


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

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    state_response = _wait_for_agent_run_completion(client, run_id)
    assert state_response.json()["status"] == "completed"
    assert model_client.calls[0].messages[-1].content == "查找季度报告"
    system_prompt = model_client.calls[0].messages[0].content
    assert system_prompt is not None
    assert "理解用户的整理意图" in system_prompt
    assert "search_files" in system_prompt
    assert "propose_move" in system_prompt
    assert "propose_rename" in system_prompt
    assert "propose_quarantine" in system_prompt
    assert "不得审批" in system_prompt
    assert "不得执行" in system_prompt
    with session_factory() as session:
        persisted_run = session.get(AgentRun, run_id)
        assert persisted_run is not None
        assert persisted_run.status == "completed"


def test_agent_run_api_returns_before_background_executor_finishes(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id, _ = _seed_workspace(
        session_factory,
        name="异步快速创建",
    )
    started = Event()
    release = Event()

    class BlockingExecutor:
        def run(
            self,
            session: Session,
            *,
            workspace_id: int,
            request_text: str,
            run_id: int | None = None,
            cancel_event: Event | None = None,
        ) -> AgentRunResponse:
            assert run_id is not None
            started.set()
            assert release.wait(timeout=5)
            SqlAlchemyAgentRunRecorder(session).finish_run(
                agent_run_id=run_id,
                status="completed",
                model_turns=1,
                error_code=None,
            )
            return AgentRunResponse(
                run_id=run_id,
                status="completed",
                final_answer="完成",
            )

    app.dependency_overrides[get_agent_run_executor] = BlockingExecutor

    try:
        response = client.post(
            "/api/v1/agent-runs",
            json={
                "workspace_id": workspace_id,
                "request_text": "执行异步请求",
            },
        )

        assert response.status_code == 202
        run_id = response.json()["run_id"]
        assert started.wait(timeout=2)
        state_response = client.get(f"/api/v1/agent-runs/{run_id}")
        assert state_response.status_code == 200
        assert state_response.json()["status"] == "running"
    finally:
        release.set()

    assert _wait_for_agent_run_completion(client, run_id).json()["status"] == (
        "completed"
    )


def test_agent_run_api_records_background_failure(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id, _ = _seed_workspace(
        session_factory,
        name="异步失败记录",
    )
    started = Event()

    class FailingExecutor:
        def run(
            self,
            session: Session,
            *,
            workspace_id: int,
            request_text: str,
            run_id: int | None = None,
            cancel_event: Event | None = None,
        ) -> AgentRunResponse:
            started.set()
            raise RuntimeError("background failure")

    app.dependency_overrides[get_agent_run_executor] = FailingExecutor
    response = client.post(
        "/api/v1/agent-runs",
        json={
            "workspace_id": workspace_id,
            "request_text": "触发后台失败",
        },
    )

    assert response.status_code == 202
    assert started.wait(timeout=2)
    state_response = _wait_for_agent_run_completion(
        client,
        response.json()["run_id"],
    )
    assert state_response.json()["status"] == "failed"
    assert state_response.json()["error_code"] == "model_provider_error"


def test_cancel_agent_run_stops_background_execution(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id, _ = _seed_workspace(
        session_factory,
        name="取消后台运行",
    )
    started = Event()

    class CancellableExecutor:
        def run(
            self,
            session: Session,
            *,
            workspace_id: int,
            request_text: str,
            run_id: int | None = None,
            cancel_event: Event | None = None,
        ) -> AgentRunResponse:
            assert run_id is not None
            assert cancel_event is not None
            started.set()
            assert cancel_event.wait(timeout=5)
            SqlAlchemyAgentRunRecorder(session).finish_run(
                agent_run_id=run_id,
                status="cancelled",
                model_turns=0,
                error_code=None,
            )
            return AgentRunResponse(
                run_id=run_id,
                status="cancelled",
            )

    app.dependency_overrides[get_agent_run_executor] = CancellableExecutor
    response = client.post(
        "/api/v1/agent-runs",
        json={
            "workspace_id": workspace_id,
            "request_text": "取消这个请求",
        },
    )

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert started.wait(timeout=2)

    cancel_response = client.post(f"/api/v1/agent-runs/{run_id}/cancel")

    assert cancel_response.status_code == 200
    assert _wait_for_agent_run_completion(client, run_id).json()["status"] == (
        "cancelled"
    )
    repeated_cancel_response = client.post(
        f"/api/v1/agent-runs/{run_id}/cancel"
    )

    assert repeated_cancel_response.status_code == 200
    assert repeated_cancel_response.json()["status"] == "cancelled"


@pytest.mark.parametrize(
    ("status", "error_code", "model_turns"),
    [
        ("completed", None, 1),
        ("failed", "model_provider_error", 0),
    ],
)
def test_cancel_agent_run_preserves_terminal_status(
    agent_client: tuple[TestClient, sessionmaker[Session]],
    status: str,
    error_code: str | None,
    model_turns: int,
) -> None:
    client, session_factory = agent_client
    with session_factory() as session:
        agent_run = AgentRun(
            status=status,
            model_turns=model_turns,
            error_code=error_code,
        )
        session.add(agent_run)
        session.commit()
        run_id = agent_run.id

    response = client.post(f"/api/v1/agent-runs/{run_id}/cancel")

    assert response.status_code == 200
    assert response.json() == {
        "run_id": run_id,
        "status": status,
        "model_turns": model_turns,
        "error_code": error_code,
    }
    with session_factory() as session:
        persisted_run = session.get(AgentRun, run_id)
        assert persisted_run is not None
        assert persisted_run.status == status
        assert persisted_run.error_code == error_code


def test_cancel_agent_run_rejects_unknown_run(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, _ = agent_client

    response = client.post("/api/v1/agent-runs/999/cancel")

    assert response.status_code == 404
    assert response.json() == {
        "detail": {
            "code": "agent_run_not_found",
            "message": "Agent 运行记录不存在。",
        }
    }


def test_resume_agent_run_returns_accepted_and_restarts_run(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id, _ = _seed_workspace(
        session_factory,
        name="恢复后台运行",
    )
    initial_messages = (
        ModelMessage(role="system", content="系统上下文"),
        ModelMessage(role="user", content="恢复这个请求"),
    )
    with session_factory() as session:
        recorder = SqlAlchemyAgentRunRecorder(session)
        run_id = recorder.start_pending_run(
            workspace_id=workspace_id,
            request_text="恢复这个请求",
            messages=initial_messages,
        )
        recorder.finish_run(
            agent_run_id=run_id,
            status="cancelled",
            model_turns=0,
            error_code=None,
        )
    expected_run_id = run_id

    started = Event()

    class ResumeExecutor:
        def run(
            self,
            session: Session,
            *,
            workspace_id: int,
            request_text: str,
            run_id: int | None = None,
            cancel_event: Event | None = None,
        ) -> AgentRunResponse:
            assert workspace_id > 0
            assert request_text == "恢复这个请求"
            assert run_id == expected_run_id
            started.set()
            assert run_id is not None
            SqlAlchemyAgentRunRecorder(session).finish_run(
                agent_run_id=run_id,
                status="completed",
                model_turns=1,
                error_code=None,
            )
            return AgentRunResponse(
                run_id=run_id,
                status="completed",
                final_answer="恢复完成",
            )

    app.dependency_overrides[get_agent_run_executor] = ResumeExecutor
    response = client.post(f"/api/v1/agent-runs/{run_id}/resume")

    assert response.status_code == 202
    assert response.json() == {"run_id": run_id}
    assert started.wait(timeout=2)
    assert _wait_for_agent_run_completion(client, run_id).json()["status"] == (
        "completed"
    )


@pytest.mark.parametrize(
    "status",
    ["pending", "running", "waiting_approval", "completed"],
)
def test_resume_agent_run_rejects_non_resumable_status(
    agent_client: tuple[TestClient, sessionmaker[Session]],
    status: str,
) -> None:
    client, session_factory = agent_client
    with session_factory() as session:
        agent_run = AgentRun(status=status)
        session.add(agent_run)
        session.commit()
        run_id = agent_run.id

    response = client.post(f"/api/v1/agent-runs/{run_id}/resume")

    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "code": "agent_run_resume_not_allowed",
            "message": "Agent 运行当前状态不允许恢复。",
        }
    }


def test_resume_agent_run_restores_persisted_messages(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id, _ = _seed_workspace(
        session_factory,
        name="恢复持久上下文",
    )

    class FailOnceAfterToolModelClient:
        def __init__(self) -> None:
            self.calls: list[tuple[ModelMessage, ...]] = []

        def complete(self, *, messages, tools):
            self.calls.append(tuple(messages))
            if len(self.calls) == 1:
                return _tool_call_response(
                    call_id="call_before_resume",
                    name="search_files",
                    arguments={
                        "workspace_id": workspace_id,
                        "keyword": "quarterly",
                    },
                )
            if len(self.calls) == 2:
                raise ModelClientRequestError(
                    code="model_connection_error",
                    message="temporary provider failure",
                    retryable=False,
                )
            return _final_response("恢复后的回答")

    model_client = FailOnceAfterToolModelClient()
    app.dependency_overrides[get_agent_run_executor] = lambda: (
        ReadOnlyAgentRunExecutor(lambda: model_client)
    )

    response = client.post(
        "/api/v1/agent-runs",
        json={
            "workspace_id": workspace_id,
            "request_text": "查找季度报告并继续",
        },
    )

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    failed_response = _wait_for_agent_run_completion(client, run_id)
    assert failed_response.json()["status"] == "failed"
    assert failed_response.json()["model_turns"] == 1

    resume_response = client.post(f"/api/v1/agent-runs/{run_id}/resume")

    assert resume_response.status_code == 202
    completed_response = _wait_for_agent_run_completion(client, run_id)
    assert completed_response.json()["status"] == "completed"
    assert completed_response.json()["model_turns"] == 2
    assert model_client.calls[2] == model_client.calls[1]
    assert [message.role for message in model_client.calls[2]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]


def test_resume_agent_run_rejects_missing_persisted_context(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id, _ = _seed_workspace(
        session_factory,
        name="缺少恢复上下文",
    )
    with session_factory() as session:
        agent_run = AgentRun(
            status="failed",
            workspace_id=workspace_id,
            request_text="无法恢复",
            model_turns=0,
            error_code="model_provider_error",
        )
        session.add(agent_run)
        session.commit()
        run_id = agent_run.id

    response = client.post(f"/api/v1/agent-runs/{run_id}/resume")

    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "code": "agent_run_resume_unavailable",
            "message": "Agent 运行缺少可恢复的持久状态。",
        }
    }


def test_resume_agent_run_rejects_corrupt_persisted_context(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id, _ = _seed_workspace(
        session_factory,
        name="损坏恢复上下文",
    )
    with session_factory() as session:
        agent_run = AgentRun(
            status="failed",
            workspace_id=workspace_id,
            request_text="损坏上下文",
            context_json="not-json",
            model_turns=0,
            error_code="model_provider_error",
        )
        session.add(agent_run)
        session.commit()
        run_id = agent_run.id

    response = client.post(f"/api/v1/agent-runs/{run_id}/resume")

    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "code": "agent_run_resume_unavailable",
            "message": "Agent 运行缺少可恢复的持久状态。",
        }
    }


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

    assert response.status_code == 202
    state_response = _wait_for_agent_run_completion(
        client,
        response.json()["run_id"],
    )
    assert state_response.json()["status"] == "completed"
    assert model_client.calls[1].messages[-1].content is not None


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

    assert response.status_code == 202
    state_response = _wait_for_agent_run_completion(
        client,
        response.json()["run_id"],
    )
    assert state_response.json()["status"] == "completed"


def test_agent_run_api_turns_natural_language_into_waiting_move_proposal(
    agent_client: tuple[TestClient, sessionmaker[Session]],
    tmp_path: Path,
) -> None:
    client, session_factory = agent_client
    workspace_root = tmp_path / "natural-language-workspace"
    workspace_id, file_id, source_path = _seed_disk_workspace(
        session_factory,
        workspace_root,
        name="自然语言整理工作区",
        relative_path="downloads/calculus-course.pdf",
        contents=b"course pdf",
    )
    target_directory = workspace_root / "subjects" / "mathematics"
    target_directory.mkdir(parents=True)
    original_contents = source_path.read_bytes()
    model_client = FakeModelClient(
        [
            _tool_call_response(
                call_id="call_search_courses",
                name="search_files",
                arguments={
                    "workspace_id": workspace_id,
                    "keyword": "课程 PDF",
                },
            ),
            _tool_call_response(
                call_id="call_propose_course_move",
                name="propose_move",
                arguments={
                    "workspace_id": workspace_id,
                    "source_file_id": file_id,
                    "destination": "subjects/mathematics",
                },
            ),
            _final_response("已提出按科目整理课程 PDF 的待审批计划。"),
        ]
    )
    app.dependency_overrides[get_agent_run_executor] = lambda: (
        ReadOnlyAgentRunExecutor(lambda: model_client)
    )

    response = client.post(
        "/api/v1/agent-runs",
        json={
            "workspace_id": workspace_id,
            "request_text": "把下载目录里的课程 PDF 按科目整理",
        },
    )

    assert response.status_code == 202
    _wait_for_agent_run_completion(client, response.json()["run_id"])
    proposal_message = model_client.calls[2].messages[-1]
    proposal_result = ToolResult.model_validate_json(proposal_message.content)
    assert proposal_result.ok is True
    assert proposal_result.data is not None
    plan_id = proposal_result.data["plan_id"]
    with session_factory() as session:
        plan = session.get(OperationPlanRecord, plan_id)
        assert plan is not None
        assert plan.status == "WAITING_APPROVAL"
        assert plan.workspace_id == workspace_id
        assert plan.items[0].source_file_id == file_id
        assert plan.items[0].target_relative_path == (
            "subjects/mathematics/calculus-course.pdf"
        )
    assert source_path.exists()
    assert source_path.read_bytes() == original_contents
    assert not (target_directory / source_path.name).exists()


def test_agent_run_api_does_not_create_move_proposal_without_search_results(
    agent_client: tuple[TestClient, sessionmaker[Session]],
    tmp_path: Path,
) -> None:
    client, session_factory = agent_client
    workspace_id, _, source_path = _seed_disk_workspace(
        session_factory,
        tmp_path / "empty-search-workspace",
        name="无匹配整理工作区",
        relative_path="downloads/physics-course.pdf",
        contents=b"course pdf",
    )
    model_client = FakeModelClient(
        [
            _tool_call_response(
                call_id="call_search_missing",
                name="search_files",
                arguments={
                    "workspace_id": workspace_id,
                    "keyword": "不存在的科目",
                },
            ),
            _final_response("没有找到匹配文件，无法提出安全计划。"),
        ]
    )
    app.dependency_overrides[get_agent_run_executor] = lambda: (
        ReadOnlyAgentRunExecutor(lambda: model_client)
    )

    response = client.post(
        "/api/v1/agent-runs",
        json={
            "workspace_id": workspace_id,
            "request_text": "把下载目录里不存在的科目课程整理到对应目录",
        },
    )

    assert response.status_code == 202
    _wait_for_agent_run_completion(client, response.json()["run_id"])
    with session_factory() as session:
        assert session.scalar(select(OperationPlanRecord)) is None
    assert source_path.exists()


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

    assert response.status_code == 202
    _wait_for_agent_run_completion(client, response.json()["run_id"])
    returned_tool_message = model_client.calls[1].messages[-1]
    tool_result = ToolResult.model_validate_json(returned_tool_message.content)
    assert tool_result.ok is False
    assert tool_result.error is not None
    assert tool_result.error.code == "invalid_arguments"


def test_agent_run_api_rejects_approval_tool_request(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id, _ = _seed_workspace(
        session_factory,
        name="拒绝审批请求",
    )
    model_client = FakeModelClient(
        [
            _tool_call_response(
                call_id="call_approve",
                name="approve",
                arguments={
                    "workspace_id": workspace_id,
                    "plan_id": 1,
                },
            ),
            _final_response("审批工具不可用。"),
        ]
    )
    app.dependency_overrides[get_agent_run_executor] = lambda: (
        ReadOnlyAgentRunExecutor(lambda: model_client)
    )

    response = client.post(
        "/api/v1/agent-runs",
        json={
            "workspace_id": workspace_id,
            "request_text": "批准这个整理计划",
        },
    )

    assert response.status_code == 202
    _wait_for_agent_run_completion(client, response.json()["run_id"])
    returned_tool_message = model_client.calls[1].messages[-1]
    tool_result = ToolResult.model_validate_json(returned_tool_message.content)
    assert tool_result.ok is False
    assert tool_result.error is not None
    assert tool_result.error.code == "unknown_tool"


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
