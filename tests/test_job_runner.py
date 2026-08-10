from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base
from backend.app.job_runner import (
    JobContext,
    JobIdentityConflictError,
    SingleProcessJobRunner,
)
from backend.app.job_store import SqlAlchemyJobStore
from backend.app.job_system import JobEvent, JobState, transition_job
from backend.app.models import Workspace


def _wait_for_status(
    runner: SingleProcessJobRunner,
    job_id,
    expected: str,
) -> object:
    deadline = monotonic() + 2
    while monotonic() < deadline:
        state = runner.get(job_id)
        if state.status == expected:
            return state
        sleep(0.005)
    pytest.fail(f"job did not reach status {expected!r}")


def test_submit_runs_in_background_and_reports_progress() -> None:
    started = Event()
    release = Event()
    runner = SingleProcessJobRunner()

    def task(context: JobContext) -> None:
        started.set()
        context.report_progress(
            1,
            total_units=2,
            phase_code="scanning",
        )
        release.wait(timeout=2)
        context.report_progress(
            2,
            total_units=None,
            phase_code="persisting",
        )

    try:
        submitted = runner.submit(
            kind="workspace_scan",
            workspace_id=7,
            idempotency_key="scan:7:version-1",
            task=task,
        )
        assert started.wait(timeout=1)
        running = runner.get(submitted.job_id)
        assert running.status == "running"
        assert running.attempts[0].progress.completed_units == 1

        release.set()
        completed = _wait_for_status(runner, submitted.job_id, "succeeded")
        assert completed.attempts[0].progress.completed_units == 2
        assert completed.attempts[0].progress.phase_code == "persisting"
    finally:
        release.set()
        runner.shutdown()


def test_running_task_observes_cooperative_cancellation() -> None:
    started = Event()
    cancellation_seen = Event()
    runner = SingleProcessJobRunner()

    def task(context: JobContext) -> None:
        started.set()
        while True:
            try:
                context.raise_if_cancelled()
            except Exception:
                cancellation_seen.set()
                raise
            sleep(0.005)

    try:
        submitted = runner.submit(
            kind="workspace_scan",
            workspace_id=7,
            idempotency_key="scan:7:cancel-1",
            task=task,
        )
        assert started.wait(timeout=1)
        requested = runner.cancel(submitted.job_id)
        assert requested.status == "cancel_requested"
        cancelled = _wait_for_status(runner, submitted.job_id, "cancelled")
        assert cancellation_seen.is_set()
        assert cancelled.attempts[0].status == "cancelled"
    finally:
        runner.shutdown()


def test_duplicate_submission_returns_original_job_without_second_execution() -> None:
    executions = 0
    completed = Event()
    runner = SingleProcessJobRunner()

    def task(_: JobContext) -> None:
        nonlocal executions
        executions += 1
        completed.set()

    def conflicting_task(_: JobContext) -> None:
        raise AssertionError("conflicting task must not run")

    try:
        first = runner.submit(
            kind="document_index",
            workspace_id=9,
            idempotency_key="index:9:version-1",
            task=task,
        )
        duplicate = runner.submit(
            kind="document_index",
            workspace_id=9,
            idempotency_key="index:9:version-1",
            task=conflicting_task,
        )

        assert duplicate.job_id == first.job_id
        assert completed.wait(timeout=1)
        _wait_for_status(runner, first.job_id, "succeeded")
        assert executions == 1

        with pytest.raises(JobIdentityConflictError):
            runner.submit(
                kind="workspace_scan",
                workspace_id=9,
                idempotency_key="index:9:version-1",
                task=conflicting_task,
            )
    finally:
        runner.shutdown()


def test_restart_recovers_interrupted_attempt_and_deduplicates_delivery(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'runner-restart.db').as_posix()}"
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    with session_factory() as session:
        workspace = Workspace(
            name="Runner 重启测试工作区",
            root_path=str(tmp_path / "workspace"),
        )
        session.add(workspace)
        session.commit()
        workspace_id = workspace.id

    store = SqlAlchemyJobStore(session_factory)
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    initial = JobState(
        job_id=UUID("7ed862b4-cf85-441d-9903-73599aa655a7"),
        kind="workspace_scan",
        workspace_id=workspace_id,
        idempotency_key="scan:workspace:restart",
        max_attempts=2,
        created_at=created_at,
    )
    persisted = store.create_or_get(initial)
    running = transition_job(
        persisted,
        JobEvent(
            job_id=persisted.job_id,
            sequence_no=1,
            kind="attempt_started",
            attempt_id=UUID("3ea544f8-7302-4bf6-ae49-676d2eaf4523"),
            occurred_at=created_at + timedelta(seconds=1),
        ),
    )
    # 留下 running 记录，代表 worker 在开始落库后、完成落库前退出。
    store.save_transition(expected_revision=0, state=running)

    executions = 0
    completed = Event()

    def resumed_task(_: JobContext) -> None:
        nonlocal executions
        executions += 1
        completed.set()

    def duplicate_task(_: JobContext) -> None:
        raise AssertionError("duplicate delivery must not execute")

    runner = SingleProcessJobRunner(store=store)
    try:
        resumed = runner.submit(
            kind="workspace_scan",
            workspace_id=workspace_id,
            idempotency_key="scan:workspace:restart",
            task=resumed_task,
            max_attempts=2,
        )
        duplicate = runner.submit(
            kind="workspace_scan",
            workspace_id=workspace_id,
            idempotency_key="scan:workspace:restart",
            task=duplicate_task,
            max_attempts=2,
        )

        assert duplicate.job_id == resumed.job_id
        assert completed.wait(timeout=1)
        succeeded = _wait_for_status(runner, resumed.job_id, "succeeded")
        assert executions == 1
        assert [attempt.status for attempt in succeeded.attempts] == [
            "interrupted",
            "succeeded",
        ]
        assert succeeded.attempts[0].error_code == "worker_interrupted"
        assert succeeded.attempts[0].retryable is True

        reopened = SqlAlchemyJobStore(session_factory).get(resumed.job_id)
        assert reopened == succeeded
    finally:
        runner.shutdown()
        engine.dispose()
