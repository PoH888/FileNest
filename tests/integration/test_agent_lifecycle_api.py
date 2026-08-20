from collections.abc import Iterator
from pathlib import Path
from threading import Event
from time import monotonic, sleep

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_api import AgentRunResponse, get_agent_run_executor
from backend.app.agent_observability import SqlAlchemyAgentRunRecorder
from backend.app.agent_recovery import recover_unfinished_agent_runs
from backend.app.database import Base, get_session
from backend.app.main import app
from backend.app.model_client import ModelMessage
from backend.app.models import AgentRun, Workspace


@pytest.fixture
def agent_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, sessionmaker[Session]]]:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'agent-lifecycle.db').as_posix()}",
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
) -> int:
    with session_factory() as session:
        workspace = Workspace(name=name, root_path=f"D:/Test/{name}")
        session.add(workspace)
        session.flush()
        workspace_id = workspace.id
        session.commit()
        return workspace_id


def _wait_for_agent_run_status(
    client: TestClient,
    run_id: int,
    expected_status: str,
) -> object:
    deadline = monotonic() + 5
    while True:
        response = client.get(f"/api/v1/agent-runs/{run_id}")
        assert response.status_code == 200
        if response.json()["status"] == expected_status:
            return response
        if monotonic() >= deadline:
            pytest.fail(
                f"Agent Run did not reach status {expected_status!r}"
            )
        sleep(0.01)


def test_create_agent_run_returns_accepted_run_id_and_starts_background_run(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id = _seed_workspace(session_factory, name="创建生命周期")
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
            assert workspace_id > 0
            assert request_text == "创建请求"
            assert run_id is not None
            assert cancel_event is not None
            started.set()
            assert release.wait(timeout=5)
            SqlAlchemyAgentRunRecorder(session).finish_run(
                agent_run_id=run_id,
                status="completed",
                model_turns=0,
                error_code=None,
            )
            return AgentRunResponse(
                run_id=run_id,
                status="completed",
            )

    app.dependency_overrides[get_agent_run_executor] = BlockingExecutor
    try:
        response = client.post(
            "/api/v1/agent-runs",
            json={
                "workspace_id": workspace_id,
                "request_text": "  创建请求  ",
            },
        )

        assert response.status_code == 202
        run_id = response.json()["run_id"]
        assert isinstance(run_id, int)
        assert started.wait(timeout=2)

        state_response = client.get(f"/api/v1/agent-runs/{run_id}")
        assert state_response.status_code == 200
        assert state_response.json()["status"] == "running"
    finally:
        release.set()

    assert _wait_for_agent_run_status(client, run_id, "completed").json()[
        "status"
    ] == "completed"
    with session_factory() as session:
        persisted_run = session.get(AgentRun, run_id)
        assert persisted_run is not None
        assert persisted_run.status == "completed"


def test_create_agent_run_rejects_missing_workspace_before_executor(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, _ = agent_client

    class UnexpectedExecutor:
        def run(self, *args: object, **kwargs: object) -> AgentRunResponse:
            raise AssertionError("missing workspace must not call the Agent")

    app.dependency_overrides[get_agent_run_executor] = UnexpectedExecutor

    response = client.post(
        "/api/v1/agent-runs",
        json={"workspace_id": 999, "request_text": "创建请求"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": {
            "code": "workspace_not_found",
            "message": "工作区不存在。",
        }
    }


def test_cancel_agent_run_stops_background_execution(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id = _seed_workspace(session_factory, name="取消生命周期")
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
            assert workspace_id > 0
            assert request_text == "取消请求"
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
            "request_text": "取消请求",
        },
    )

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert started.wait(timeout=2)

    cancel_response = client.post(f"/api/v1/agent-runs/{run_id}/cancel")

    assert cancel_response.status_code == 200
    assert cancel_response.json()["run_id"] == run_id
    assert _wait_for_agent_run_status(client, run_id, "cancelled").json()[
        "status"
    ] == "cancelled"
    with session_factory() as session:
        persisted_run = session.get(AgentRun, run_id)
        assert persisted_run is not None
        assert persisted_run.status == "cancelled"


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


def test_repeated_cancel_preserves_cancelled_run(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id = _seed_workspace(session_factory, name="重复取消生命周期")
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
            assert workspace_id > 0
            assert request_text == "重复取消请求"
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
            "request_text": "重复取消请求",
        },
    )

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert started.wait(timeout=2)

    first_cancel_response = client.post(
        f"/api/v1/agent-runs/{run_id}/cancel"
    )
    assert first_cancel_response.status_code == 200
    assert _wait_for_agent_run_status(client, run_id, "cancelled").json()[
        "status"
    ] == "cancelled"

    second_cancel_response = client.post(
        f"/api/v1/agent-runs/{run_id}/cancel"
    )

    assert second_cancel_response.status_code == 200
    assert second_cancel_response.json() == {
        "run_id": run_id,
        "status": "cancelled",
        "model_turns": 0,
        "error_code": None,
    }


def test_repeated_cancel_rejects_unknown_run_each_time(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, _ = agent_client

    for _ in range(2):
        response = client.post("/api/v1/agent-runs/999/cancel")

        assert response.status_code == 404
        assert response.json() == {
            "detail": {
                "code": "agent_run_not_found",
                "message": "Agent 运行记录不存在。",
            }
        }


def test_get_cancelled_agent_run_status(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    with session_factory() as session:
        agent_run = AgentRun(
            status="cancelled",
            model_turns=0,
            error_code=None,
        )
        session.add(agent_run)
        session.commit()
        run_id = agent_run.id

    response = client.get(f"/api/v1/agent-runs/{run_id}")

    assert response.status_code == 200
    assert response.json() == {
        "run_id": run_id,
        "status": "cancelled",
        "model_turns": 0,
        "error_code": None,
    }


def test_get_cancelled_agent_run_status_rejects_unknown_run(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, _ = agent_client

    response = client.get("/api/v1/agent-runs/999")

    assert response.status_code == 404
    assert response.json() == {
        "detail": {
            "code": "agent_run_not_found",
            "message": "Agent 运行记录不存在。",
        }
    }


def test_resume_agent_run_restarts_cancelled_run(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id = _seed_workspace(session_factory, name="恢复生命周期")
    request_text = "恢复请求"
    initial_messages = (
        ModelMessage(role="system", content="系统上下文"),
        ModelMessage(role="user", content=request_text),
    )
    with session_factory() as session:
        recorder = SqlAlchemyAgentRunRecorder(session)
        run_id = recorder.start_pending_run(
            workspace_id=workspace_id,
            request_text=request_text,
            messages=initial_messages,
        )
        recorder.finish_run(
            agent_run_id=run_id,
            status="cancelled",
            model_turns=0,
            error_code=None,
        )

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
            assert request_text == "恢复请求"
            assert run_id is not None
            assert cancel_event is not None
            started.set()
            SqlAlchemyAgentRunRecorder(session).finish_run(
                agent_run_id=run_id,
                status="completed",
                model_turns=1,
                error_code=None,
            )
            return AgentRunResponse(
                run_id=run_id,
                status="completed",
            )

    app.dependency_overrides[get_agent_run_executor] = ResumeExecutor
    response = client.post(f"/api/v1/agent-runs/{run_id}/resume")

    assert response.status_code == 202
    assert response.json() == {"run_id": run_id}
    assert started.wait(timeout=2)
    assert _wait_for_agent_run_status(client, run_id, "completed").json()[
        "status"
    ] == "completed"


def test_resume_agent_run_rejects_cancelled_run_without_context(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id = _seed_workspace(session_factory, name="缺少恢复上下文")
    with session_factory() as session:
        agent_run = AgentRun(
            status="cancelled",
            workspace_id=workspace_id,
            request_text="无法恢复",
            model_turns=0,
            error_code=None,
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


def test_resume_completed_agent_run_is_not_allowed(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    with session_factory() as session:
        agent_run = AgentRun(
            status="completed",
            model_turns=1,
            error_code=None,
        )
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


def test_resume_agent_run_rejects_unknown_run(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, _ = agent_client

    response = client.post("/api/v1/agent-runs/999/resume")

    assert response.status_code == 404
    assert response.json() == {
        "detail": {
            "code": "agent_run_not_found",
            "message": "Agent 运行记录不存在。",
        }
    }


def test_restart_marks_interrupted_run_failed_and_allows_resume(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id = _seed_workspace(session_factory, name="重启恢复生命周期")
    request_text = "重启后恢复请求"
    initial_messages = (
        ModelMessage(role="system", content="系统上下文"),
        ModelMessage(role="user", content=request_text),
    )
    with session_factory() as session:
        recorder = SqlAlchemyAgentRunRecorder(session)
        run_id = recorder.start_pending_run(
            workspace_id=workspace_id,
            request_text=request_text,
            messages=initial_messages,
        )
        recorder.start_existing_run(run_id)

    with session_factory() as session:
        assert recover_unfinished_agent_runs(session) == (run_id,)
    with session_factory() as session:
        interrupted_run = session.get(AgentRun, run_id)
        assert interrupted_run is not None
        assert interrupted_run.status == "failed"
        assert interrupted_run.error_code == "worker_interrupted"

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
            assert request_text == "重启后恢复请求"
            assert run_id is not None
            assert cancel_event is not None
            started.set()
            SqlAlchemyAgentRunRecorder(session).finish_run(
                agent_run_id=run_id,
                status="completed",
                model_turns=1,
                error_code=None,
            )
            return AgentRunResponse(
                run_id=run_id,
                status="completed",
            )

    app.dependency_overrides[get_agent_run_executor] = ResumeExecutor
    response = client.post(f"/api/v1/agent-runs/{run_id}/resume")

    assert response.status_code == 202
    assert response.json() == {"run_id": run_id}
    assert started.wait(timeout=2)
    assert _wait_for_agent_run_status(client, run_id, "completed").json()[
        "status"
    ] == "completed"


def test_restart_resume_rejects_run_without_persisted_context(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id = _seed_workspace(session_factory, name="重启缺少上下文")
    with session_factory() as session:
        agent_run = AgentRun(
            status="running",
            workspace_id=workspace_id,
            request_text="重启后无法恢复",
            model_turns=0,
            error_code=None,
        )
        session.add(agent_run)
        session.commit()
        run_id = agent_run.id

    with session_factory() as session:
        assert recover_unfinished_agent_runs(session) == (run_id,)

    response = client.post(f"/api/v1/agent-runs/{run_id}/resume")

    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "code": "agent_run_resume_unavailable",
            "message": "Agent 运行缺少可恢复的持久状态。",
        }
    }


def test_agent_run_records_failed_background_execution(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    workspace_id = _seed_workspace(session_factory, name="失败生命周期")
    started = Event()

    class FailingExecutor:
        def run(self, *args: object, **kwargs: object) -> AgentRunResponse:
            started.set()
            raise RuntimeError("controlled provider failure")

    app.dependency_overrides[get_agent_run_executor] = FailingExecutor
    response = client.post(
        "/api/v1/agent-runs",
        json={
            "workspace_id": workspace_id,
            "request_text": "触发失败",
        },
    )

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert started.wait(timeout=2)
    failed_response = _wait_for_agent_run_status(client, run_id, "failed")
    assert failed_response.json()["error_code"] == "model_provider_error"
    with session_factory() as session:
        persisted_run = session.get(AgentRun, run_id)
        assert persisted_run is not None
        assert persisted_run.status == "failed"
        assert persisted_run.error_code == "model_provider_error"


def test_get_failed_agent_run_status_preserves_error_evidence(
    agent_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, session_factory = agent_client
    with session_factory() as session:
        agent_run = AgentRun(
            status="failed",
            model_turns=0,
            error_code="model_provider_error",
        )
        session.add(agent_run)
        session.commit()
        run_id = agent_run.id

    response = client.get(f"/api/v1/agent-runs/{run_id}")

    assert response.status_code == 200
    assert response.json() == {
        "run_id": run_id,
        "status": "failed",
        "model_turns": 0,
        "error_code": "model_provider_error",
    }
