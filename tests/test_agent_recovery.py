import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.agent_recovery import (
    inspect_agent_run_recovery,
    recover_unfinished_agent_runs,
    scan_unfinished_agent_runs,
)
from backend.app.agent_observability import SqlAlchemyAgentRunRecorder
from backend.app.database import Base
from backend.app.model_client import ModelMessage
from backend.app.models import AgentMessage, AgentRun, AgentSession


def test_startup_scan_identifies_unfinished_agent_runs_and_marks_interrupted(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'agent-recovery.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine, expire_on_commit=False) as session:
            runs = [
                AgentRun(status="running", model_turns=2),
                AgentRun(status="waiting_approval", model_turns=3),
                AgentRun(
                    status="failed",
                    model_turns=4,
                    error_code="model_connection_error",
                ),
                AgentRun(status="completed", model_turns=5),
                AgentRun(status="pending", model_turns=0),
            ]
            session.add_all(runs)
            session.commit()

            scanned = scan_unfinished_agent_runs(session)
            scanned_ids = tuple(agent_run.id for agent_run in scanned)
            recovered_ids = recover_unfinished_agent_runs(session)

            assert scanned_ids == recovered_ids
            assert [agent_run.status for agent_run in scanned] == [
                "failed",
                "waiting_approval",
                "failed",
            ]

            session.expire_all()
            interrupted = session.get(AgentRun, runs[0].id)
            completed = session.get(AgentRun, runs[3].id)
            pending = session.get(AgentRun, runs[4].id)

            assert interrupted is not None
            assert interrupted.status == "failed"
            assert interrupted.error_code == "worker_interrupted"
            assert interrupted.model_turns == 2
            assert interrupted.finished_at is not None
            assert completed is not None
            assert completed.status == "completed"
            assert pending is not None
            assert pending.status == "pending"
    finally:
        engine.dispose()


def test_recovery_snapshot_loads_completed_and_incomplete_lifecycle_steps(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'agent-recovery-snapshot.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)
    initial_messages = (
        ModelMessage(role="user", content="safe request"),
    )
    response_messages = (
        *initial_messages,
        ModelMessage(role="assistant", content="safe response"),
    )

    try:
        with Session(engine, expire_on_commit=False) as session:
            recorder = SqlAlchemyAgentRunRecorder(session, workspace_id=7)
            run_id = recorder.start_pending_run(
                workspace_id=7,
                request_text="safe request",
                messages=initial_messages,
            )
            recorder.start_existing_run(run_id)
            first_step_id = recorder.start_step(
                agent_run_id=run_id,
                step_index=0,
                step_type="model_turn",
                messages=initial_messages,
            )
            recorder.record_model_message(
                agent_run_id=run_id,
                agent_step_id=first_step_id,
                sequence_no=0,
                message=initial_messages[0],
            )
            recorder.record_model_message(
                agent_run_id=run_id,
                agent_step_id=first_step_id,
                sequence_no=1,
                message=response_messages[-1],
            )
            recorder.record_model_run(
                agent_run_id=run_id,
                agent_step_id=first_step_id,
                model="deterministic-model",
                model_provider="fake",
                prompt_version="agent-system-v1",
                metrics=None,
            )
            recorder.finish_step(
                agent_run_id=run_id,
                agent_step_id=first_step_id,
                status="completed",
                messages=response_messages,
                error_code=None,
            )
            recorder.checkpoint_run(
                agent_run_id=run_id,
                messages=response_messages,
                model_turns=1,
            )
            second_step_id = recorder.start_step(
                agent_run_id=run_id,
                step_index=1,
                step_type="model_turn",
                messages=response_messages,
            )

            snapshot = inspect_agent_run_recovery(session, run_id)

            assert snapshot.can_resume is True
            assert snapshot.recovery_code is None
            assert snapshot.agent_session_id is not None
            assert snapshot.context_message_count == 2
            assert snapshot.last_completed_step_index == 0
            assert snapshot.incomplete_step_index == 1
            assert snapshot.model_run_ids
            assert second_step_id != first_step_id
    finally:
        engine.dispose()


def test_recovery_snapshot_rejects_workspace_mismatch_without_file_access(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'agent-recovery-mismatch.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)
    sentinel = tmp_path / "must-not-be-touched.txt"
    sentinel.write_text("unchanged", encoding="utf-8")

    try:
        with Session(engine, expire_on_commit=False) as session:
            recorder = SqlAlchemyAgentRunRecorder(session, workspace_id=7)
            run_id = recorder.start_pending_run(
                workspace_id=7,
                request_text="safe request",
                messages=(ModelMessage(role="user", content="safe request"),),
            )
            linked_session = session.scalar(
                session.query(AgentSession).where(
                    AgentSession.workspace_id == 7
                ).statement
            )
            assert linked_session is not None
            linked_session.workspace_id = 8
            session.commit()

            snapshot = inspect_agent_run_recovery(session, run_id)

            assert snapshot.can_resume is False
            assert snapshot.recovery_code == "agent_workspace_mismatch"
            assert sentinel.read_text(encoding="utf-8") == "unchanged"
            assert session.query(AgentMessage).count() == 0
    finally:
        engine.dispose()


def test_agent_run_recovers_after_real_fastapi_process_restart(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "agent-process-restart.db"
    started_path = tmp_path / "agent-started.txt"
    repository_root = Path(__file__).resolve().parents[1]
    database_url = f"sqlite:///{database_path.as_posix()}"
    environment = os.environ.copy()
    environment["FILENEST_DATABASE_URL"] = database_url
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(repository_root), environment.get("PYTHONPATH", ""))
        if part
    )

    first_process_script = r'''
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.agent_api import get_agent_run_executor
from backend.app.database import Base, SessionFactory, get_session, engine
from backend.app.main import app
from backend.app.models import Workspace


database_path = Path(sys.argv[1])
started_path = Path(sys.argv[2])
Base.metadata.create_all(bind=engine)

with SessionFactory() as session:
    workspace = Workspace(name="process-restart", root_path=str(database_path.parent))
    session.add(workspace)
    session.commit()
    workspace_id = workspace.id


class CrashExecutor:
    def run(self, session, *, workspace_id, request_text, run_id=None, cancel_event=None):
        assert run_id is not None
        started_path.write_text(str(run_id), encoding="utf-8")
        while True:
            time.sleep(0.05)


def override_get_session():
    with SessionFactory() as session:
        yield session


app.dependency_overrides[get_session] = override_get_session
app.dependency_overrides[get_agent_run_executor] = lambda: CrashExecutor()

with TestClient(app) as client:
    response = client.post(
        "/api/v1/agent-runs",
        json={"workspace_id": workspace_id, "request_text": "进程重启恢复"},
    )
    assert response.status_code == 202
    while True:
        time.sleep(1)
'''
    first_process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            first_process_script,
            str(database_path),
            str(started_path),
        ],
        cwd=repository_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    run_id: int | None = None
    try:
        deadline = monotonic() + 15
        while monotonic() < deadline:
            if started_path.is_file():
                try:
                    run_id = int(started_path.read_text(encoding="utf-8"))
                    break
                except ValueError:
                    pass
            if first_process.poll() is not None:
                stdout, stderr = first_process.communicate()
                raise AssertionError(
                    "initial FastAPI process exited before Agent started:\n"
                    f"stdout={stdout}\nstderr={stderr}"
                )
            sleep(0.05)
        assert run_id is not None, "initial FastAPI process did not start Agent"
    finally:
        if first_process.poll() is None:
            first_process.terminate()
        first_process.wait(timeout=10)
        if first_process.stdout is not None:
            first_process.stdout.close()
        if first_process.stderr is not None:
            first_process.stderr.close()

    engine = create_engine(database_url)
    try:
        with Session(engine) as session:
            interrupted = session.get(AgentRun, run_id)
            assert interrupted is not None
            assert interrupted.status == "running"
            assert interrupted.context_json is not None
    finally:
        engine.dispose()

    second_process_script = r'''
import sys
import time

from fastapi.testclient import TestClient

from backend.app.agent_api import AgentRunResponse, get_agent_run_executor
from backend.app.agent_observability import SqlAlchemyAgentRunRecorder
from backend.app.database import SessionFactory, get_session
from backend.app.main import app


run_id = int(sys.argv[1])


class ResumeExecutor:
    def run(self, session, *, workspace_id, request_text, run_id=None, cancel_event=None):
        assert run_id == %d
        assert run_id is not None
        recorder = SqlAlchemyAgentRunRecorder(session)
        messages, model_turns = recorder.load_context(run_id)
        assert messages
        recorder.finish_run(
            agent_run_id=run_id,
            status="completed",
            model_turns=model_turns + 1,
            error_code=None,
        )
        return AgentRunResponse(
            run_id=run_id,
            status="completed",
            final_answer="进程重启后恢复完成",
        )


def override_get_session():
    with SessionFactory() as session:
        yield session


app.dependency_overrides[get_session] = override_get_session
app.dependency_overrides[get_agent_run_executor] = lambda: ResumeExecutor()

with TestClient(app) as client:
    assert run_id in app.state.unfinished_agent_run_ids
    state = client.get(f"/api/v1/agent-runs/{run_id}")
    assert state.status_code == 200
    assert state.json()["status"] == "failed"
    assert state.json()["error_code"] == "worker_interrupted"

    response = client.post(f"/api/v1/agent-runs/{run_id}/resume")
    assert response.status_code == 202

    deadline = time.monotonic() + 10
    while True:
        state = client.get(f"/api/v1/agent-runs/{run_id}")
        if state.json()["status"] == "completed":
            break
        assert time.monotonic() < deadline
        time.sleep(0.05)
'''
    second_process = subprocess.run(
        [
            sys.executable,
            "-c",
            second_process_script % run_id,
            str(run_id),
        ],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert second_process.returncode == 0, (
        "restarted FastAPI process failed:\n"
        f"stdout={second_process.stdout}\nstderr={second_process.stderr}"
    )
