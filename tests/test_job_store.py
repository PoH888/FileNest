from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base
from backend.app.job_store import (
    JobStoreIdentityConflictError,
    JobStoreRevisionConflictError,
    SqlAlchemyJobStore,
)
from backend.app.job_system import JobEvent, JobState, transition_job
from backend.app.models import Workspace


def _build_store(tmp_path: Path):
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'job-store.db').as_posix()}"
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    with session_factory() as session:
        workspace = Workspace(
            name="Job Store 测试工作区",
            root_path=str(tmp_path / "workspace"),
        )
        session.add(workspace)
        session.commit()
        workspace_id = workspace.id
    return engine, session_factory, workspace_id


def test_store_round_trips_attempt_and_progress_across_store_instances(
    tmp_path: Path,
) -> None:
    engine, session_factory, workspace_id = _build_store(tmp_path)
    store = SqlAlchemyJobStore(session_factory)
    created_at = datetime(
        2026,
        9,
        1,
        10,
        0,
        tzinfo=timezone(timedelta(hours=8)),
    )
    initial = JobState(
        job_id=UUID("9cc83203-9166-43f0-a382-cbab91654132"),
        kind="workspace_scan",
        workspace_id=workspace_id,
        idempotency_key="scan:workspace:store-roundtrip",
        max_attempts=2,
        created_at=created_at,
    )

    try:
        persisted = store.create_or_get(initial)
        running = transition_job(
            persisted,
            JobEvent(
                job_id=initial.job_id,
                sequence_no=1,
                kind="attempt_started",
                attempt_id=UUID("c1fe7992-2da5-408b-b55a-f82d3582d94c"),
                occurred_at=created_at,
            ),
        )
        store.save_transition(expected_revision=0, state=running)
        progressed = transition_job(
            running,
            JobEvent(
                job_id=initial.job_id,
                sequence_no=2,
                kind="progress_reported",
                attempt_id=running.attempts[0].attempt_id,
                completed_units=4,
                total_units=10,
                phase_code="scanning",
                occurred_at=created_at,
            ),
        )
        store.save_transition(expected_revision=1, state=progressed)

        reopened_store = SqlAlchemyJobStore(session_factory)
        assert reopened_store.get(initial.job_id) == progressed
    finally:
        engine.dispose()


def test_store_returns_idempotent_job_and_rejects_stale_revision(
    tmp_path: Path,
) -> None:
    engine, session_factory, workspace_id = _build_store(tmp_path)
    store = SqlAlchemyJobStore(session_factory)
    created_at = datetime(2026, 9, 1, 11, 0, tzinfo=timezone.utc)
    initial = JobState(
        job_id=UUID("ebf1628c-2048-47e0-96d1-dff59b1b1068"),
        kind="document_index",
        workspace_id=workspace_id,
        idempotency_key="index:workspace:idempotent",
        max_attempts=1,
        created_at=created_at,
    )

    try:
        persisted = store.create_or_get(initial)
        duplicate = initial.model_copy(
            update={
                "job_id": UUID("9d48079b-395a-4ea5-95f7-eae22f4c9056")
            }
        )
        assert store.create_or_get(duplicate).job_id == persisted.job_id

        conflicting = duplicate.model_copy(
            update={"kind": "workspace_scan"}
        )
        with pytest.raises(JobStoreIdentityConflictError):
            store.create_or_get(conflicting)

        running = transition_job(
            persisted,
            JobEvent(
                job_id=persisted.job_id,
                sequence_no=1,
                kind="attempt_started",
                attempt_id=UUID("ac718ff7-24fd-447e-b0d8-aae6e4f262df"),
                occurred_at=created_at,
            ),
        )
        store.save_transition(expected_revision=0, state=running)

        with pytest.raises(JobStoreRevisionConflictError):
            store.save_transition(expected_revision=0, state=running)
    finally:
        engine.dispose()
