from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database import Base
from backend.app.job_runner import (
    JobHandlerRegistry,
    JobHandlerUnavailableError,
    JobIdentityConflictError,
    SingleProcessJobRunner,
)
from backend.app.job_store import SqlAlchemyJobStore
from backend.app.job_system import JobState, JobTaskPayload
from backend.app.models import Workspace


def _wait_for_status(
    runner: SingleProcessJobRunner,
    job_id: UUID,
    expected_status: str,
) -> JobState:
    deadline = monotonic() + 2
    while monotonic() < deadline:
        state = runner.get(job_id)
        if state.status == expected_status:
            return state
        sleep(0.005)
    pytest.fail(f"job did not reach status {expected_status!r}")


def test_job_payload_accepts_only_rebuildable_workspace_scope() -> None:
    payload = JobTaskPayload.model_validate(
        {"workspace_id": 7, "index_scope": "workspace"}
    )
    assert payload.workspace_id == 7

    with pytest.raises(ValidationError):
        JobTaskPayload.model_validate(
            {
                "workspace_id": 7,
                "absolute_tmp_path": "C:\\temp\\job",
            }
        )

    with pytest.raises(ValidationError):
        JobState(
            job_id=uuid4(),
            kind="workspace_scan",
            workspace_id=7,
            payload={"workspace_id": 8},
            idempotency_key="payload-mismatch",
            max_attempts=1,
            created_at=datetime.now(timezone.utc),
        )


def test_handler_registry_requires_exact_kind_and_version() -> None:
    called = []

    def handler(_context, payload) -> None:
        called.append(payload.workspace_id)

    registry = JobHandlerRegistry({("workspace_scan", "v1"): handler})
    assert registry.resolve(kind="workspace_scan", task_version="v1") is handler
    with pytest.raises(JobHandlerUnavailableError):
        registry.resolve(kind="workspace_scan", task_version="v2")


def test_unknown_task_version_fails_without_guessing_a_handler() -> None:
    called = []

    def handler(_context, _payload) -> None:
        called.append(True)

    runner = SingleProcessJobRunner(
        handler_registry=JobHandlerRegistry(
            {("workspace_scan", "v1"): handler}
        )
    )
    try:
        submitted = runner.submit(
            kind="workspace_scan",
            workspace_id=7,
            idempotency_key="unknown-task-version",
            task_version="v2",
            max_attempts=1,
        )
        failed = _wait_for_status(runner, submitted.job_id, "failed")
        assert failed.error_code == "recovery_required"
        assert called == []
    finally:
        runner.shutdown()


def test_same_idempotency_key_is_reused_after_runner_restart(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'restart.db').as_posix()}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    with Session(engine) as session:
        workspace = Workspace(
            name="幂等重启测试工作区",
            root_path=str(tmp_path / "workspace"),
        )
        session.add(workspace)
        session.commit()
        workspace_id = workspace.id

    calls = []

    def handler(_context, payload) -> None:
        calls.append(payload.workspace_id)

    registry = JobHandlerRegistry({("workspace_scan", "v1"): handler})
    store = SqlAlchemyJobStore(session_factory)
    first_runner = SingleProcessJobRunner(
        store=store,
        handler_registry=registry,
    )
    try:
        first = first_runner.submit(
            kind="workspace_scan",
            workspace_id=workspace_id,
            idempotency_key="restart-stable-key",
        )
        _wait_for_status(first_runner, first.job_id, "succeeded")
    finally:
        first_runner.shutdown()

    second_runner = SingleProcessJobRunner(
        store=store,
        handler_registry=registry,
    )
    try:
        second = second_runner.submit(
            kind="workspace_scan",
            workspace_id=workspace_id,
            idempotency_key="restart-stable-key",
        )
        assert second.job_id == first.job_id
        assert second.status == "succeeded"
        assert calls == [workspace_id]
    finally:
        second_runner.shutdown()
        engine.dispose()


def test_same_key_with_a_different_payload_is_rejected() -> None:
    runner = SingleProcessJobRunner()
    try:
        runner.submit(
            kind="workspace_scan",
            workspace_id=7,
            idempotency_key="payload-conflict-key",
            task=lambda _context: None,
            payload={"workspace_id": 7},
        )
        with pytest.raises(JobIdentityConflictError):
            runner.submit(
                kind="workspace_scan",
                workspace_id=7,
                idempotency_key="payload-conflict-key",
                task=lambda _context: None,
                payload={
                    "workspace_id": 7,
                    "index_scope": "workspace",
                },
            )
    finally:
        runner.shutdown()
