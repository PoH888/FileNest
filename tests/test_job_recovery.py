from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from uuid import UUID, uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database import Base
from backend.app.job_runner import (
    JobContext,
    JobHandlerRegistry,
    SingleProcessJobRunner,
)
from backend.app.job_store import SqlAlchemyJobStore
from backend.app.job_system import JobEvent, JobState, JobTaskPayload, transition_job
from backend.app.models import Workspace


def _store(tmp_path: Path) -> tuple[object, object, int]:
    engine = create_engine(f"sqlite:///{(tmp_path / 'recovery.db').as_posix()}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    with Session(engine) as session:
        workspace = Workspace(
            name="Job 恢复测试工作区",
            root_path=str(tmp_path / "workspace"),
        )
        session.add(workspace)
        session.commit()
        workspace_id = workspace.id
    return engine, session_factory, workspace_id


def _persisted_job(
    store: SqlAlchemyJobStore,
    workspace_id: int,
    *,
    key: str,
    max_attempts: int = 1,
    created_at: datetime | None = None,
) -> JobState:
    return store.create_or_get(
        JobState(
            job_id=uuid4(),
            kind="workspace_scan",
            workspace_id=workspace_id,
            idempotency_key=key,
            max_attempts=max_attempts,
            created_at=created_at or datetime.now(timezone.utc),
        )
    )


def _wait_for_store_status(
    store: SqlAlchemyJobStore,
    job_id: UUID,
    expected: str,
) -> JobState:
    deadline = monotonic() + 2
    while monotonic() < deadline:
        state = store.get(job_id)
        assert state is not None
        if state.status == expected:
            return state
        sleep(0.005)
    raise AssertionError(f"job did not reach status {expected!r}")


def test_recovery_requeues_pending_and_never_reruns_cancel_requested(
    tmp_path: Path,
) -> None:
    engine, session_factory, workspace_id = _store(tmp_path)
    store = SqlAlchemyJobStore(session_factory)
    called: list[int] = []

    def handler(_context: JobContext, payload: JobTaskPayload) -> None:
        called.append(payload.workspace_id)

    registry = JobHandlerRegistry({("workspace_scan", "v1"): handler})
    pending = _persisted_job(store, workspace_id, key="pending-recovery")
    pending_runner = SingleProcessJobRunner(
        store=store,
        handler_registry=registry,
    )
    try:
        pending_runner.recover_persisted_jobs(can_run=lambda _state: True)
        completed = _wait_for_store_status(store, pending.job_id, "succeeded")
        assert completed.attempts[0].status == "succeeded"
    finally:
        pending_runner.shutdown()

    cancelled = _persisted_job(
        store,
        workspace_id,
        key="cancelled-recovery",
        max_attempts=2,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    created_at = cancelled.created_at
    running = transition_job(
        cancelled,
        JobEvent(
            job_id=cancelled.job_id,
            sequence_no=1,
            kind="attempt_started",
            attempt_id=UUID("5a59d9d8-a5e0-4a34-89b1-5b6a84dbdb11"),
            occurred_at=created_at + timedelta(seconds=1),
        ),
    )
    requested = transition_job(
        running,
        JobEvent(
            job_id=cancelled.job_id,
            sequence_no=2,
            kind="cancellation_requested",
            occurred_at=created_at + timedelta(seconds=2),
        ),
    )
    store.save_transition(expected_revision=0, state=running)
    store.save_transition(expected_revision=1, state=requested)

    cancelled_runner = SingleProcessJobRunner(
        store=store,
        handler_registry=registry,
    )
    try:
        recovered = cancelled_runner.recover_persisted_jobs(
            can_run=lambda _state: True
        )
        assert recovered[0].status == "cancelled"
        assert store.get(cancelled.job_id).status == "cancelled"
        assert called == [workspace_id]
    finally:
        cancelled_runner.shutdown()
        engine.dispose()


def test_policy_invalid_recovery_fails_closed_without_calling_handler(
    tmp_path: Path,
) -> None:
    engine, session_factory, workspace_id = _store(tmp_path)
    store = SqlAlchemyJobStore(session_factory)
    called = Event()

    def handler(_context: JobContext, _payload: JobTaskPayload) -> None:
        called.set()

    runner = SingleProcessJobRunner(
        store=store,
        handler_registry=JobHandlerRegistry(
            {("workspace_scan", "v1"): handler}
        ),
    )
    try:
        pending = _persisted_job(
            store,
            workspace_id,
            key="invalid-policy-recovery",
        )
        recovered = runner.recover_persisted_jobs(
            can_run=lambda _state: False
        )
        assert recovered[0].status == "failed"
        assert recovered[0].error_code == "recovery_required"
        persisted = store.get(pending.job_id)
        assert persisted is not None
        assert persisted.status == "failed"
        assert persisted.attempts[0].status == "failed"
        assert not called.is_set()
    finally:
        runner.shutdown()
        engine.dispose()


def test_recovery_interrupts_old_running_attempt_then_retries(
    tmp_path: Path,
) -> None:
    engine, session_factory, workspace_id = _store(tmp_path)
    store = SqlAlchemyJobStore(session_factory)
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    persisted = _persisted_job(
        store,
        workspace_id,
        key="running-recovery",
        max_attempts=2,
        created_at=created_at,
    )
    running = transition_job(
        persisted,
        JobEvent(
            job_id=persisted.job_id,
            sequence_no=1,
            kind="attempt_started",
            attempt_id=UUID("42b1e7f5-465c-4a8a-b9d5-0e2d39998e7f"),
            occurred_at=created_at + timedelta(seconds=1),
        ),
    )
    store.save_transition(expected_revision=0, state=running)

    calls = 0

    def handler(_context: JobContext, _payload: JobTaskPayload) -> None:
        nonlocal calls
        calls += 1

    runner = SingleProcessJobRunner(
        store=store,
        handler_registry=JobHandlerRegistry({("workspace_scan", "v1"): handler}),
    )
    try:
        recovered = runner.recover_persisted_jobs(can_run=lambda _state: True)
        assert recovered[0].status == "pending"
        completed = _wait_for_store_status(store, persisted.job_id, "succeeded")
        assert [attempt.status for attempt in completed.attempts] == [
            "interrupted",
            "succeeded",
        ]
        assert calls == 1
    finally:
        runner.shutdown()
        engine.dispose()


def test_two_recovery_runners_cas_claim_one_execution(
    tmp_path: Path,
) -> None:
    engine, session_factory, workspace_id = _store(tmp_path)
    store = SqlAlchemyJobStore(session_factory)
    started = Event()
    release = Event()
    calls = 0

    def handler(_context: JobContext, _payload: JobTaskPayload) -> None:
        nonlocal calls
        calls += 1
        started.set()
        release.wait(timeout=2)

    registry = JobHandlerRegistry({("workspace_scan", "v1"): handler})
    pending = _persisted_job(store, workspace_id, key="cas-recovery")
    first = SingleProcessJobRunner(store=store, handler_registry=registry)
    second = SingleProcessJobRunner(store=store, handler_registry=registry)
    try:
        first.recover_persisted_jobs(can_run=lambda _state: True)
        second.recover_persisted_jobs(can_run=lambda _state: True)
        assert started.wait(timeout=1)
        release.set()
        completed = _wait_for_store_status(store, pending.job_id, "succeeded")
        assert completed.status == "succeeded"
        assert calls == 1
    finally:
        release.set()
        first.shutdown()
        second.shutdown()
        engine.dispose()
