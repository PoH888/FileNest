from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from backend.app.job_system import (
    JobEvent,
    JobState,
    JobTransitionError,
    JobTransitionErrorCode,
    transition_job,
)


JOB_ID = UUID("5ea58eb4-80ba-4e7e-9f20-103bde0430b1")
FIRST_ATTEMPT_ID = UUID("fd3641f4-a195-4903-85d7-1997ced8cd51")
SECOND_ATTEMPT_ID = UUID("b33d9ae2-7904-43d0-aaf4-e28117cb133b")
CREATED_AT = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)


def _job(*, max_attempts: int = 2) -> JobState:
    return JobState(
        job_id=JOB_ID,
        kind="workspace_scan",
        workspace_id=7,
        idempotency_key="workspace-scan:7:version-1",
        max_attempts=max_attempts,
        created_at=CREATED_AT,
    )


def _event(sequence_no: int, kind: str, **details: object) -> JobEvent:
    return JobEvent(
        job_id=JOB_ID,
        sequence_no=sequence_no,
        kind=kind,
        occurred_at=CREATED_AT + timedelta(seconds=sequence_no),
        **details,
    )


def test_job_records_monotonic_attempt_progress_and_success() -> None:
    state = transition_job(
        _job(),
        _event(1, "attempt_started", attempt_id=FIRST_ATTEMPT_ID),
    )
    state = transition_job(
        state,
        _event(
            2,
            "progress_reported",
            attempt_id=FIRST_ATTEMPT_ID,
            completed_units=40,
            total_units=100,
            phase_code="scanning",
        ),
    )

    with pytest.raises(JobTransitionError) as raised:
        transition_job(
            state,
            _event(
                3,
                "progress_reported",
                attempt_id=FIRST_ATTEMPT_ID,
                completed_units=39,
                total_units=100,
                phase_code="scanning",
            ),
        )

    assert raised.value.code == JobTransitionErrorCode.PROGRESS_REGRESSION

    state = transition_job(
        state,
        _event(
            3,
            "progress_reported",
            attempt_id=FIRST_ATTEMPT_ID,
            completed_units=100,
            phase_code="persisting",
        ),
    )
    state = transition_job(
        state,
        _event(4, "attempt_succeeded", attempt_id=FIRST_ATTEMPT_ID),
    )

    assert state.status == "succeeded"
    assert state.revision == 4
    assert state.finished_at == CREATED_AT + timedelta(seconds=4)
    assert len(state.attempts) == 1
    assert state.attempts[0].status == "succeeded"
    assert state.attempts[0].progress.completed_units == 100
    assert state.attempts[0].progress.total_units == 100
    assert state.attempts[0].progress.phase_code == "persisting"


def test_running_job_cancellation_is_cooperative() -> None:
    state = transition_job(
        _job(),
        _event(1, "attempt_started", attempt_id=FIRST_ATTEMPT_ID),
    )
    state = transition_job(state, _event(2, "cancellation_requested"))

    assert state.status == "cancel_requested"
    assert state.attempts[0].status == "running"
    assert state.finished_at is None

    state = transition_job(
        state,
        _event(3, "attempt_cancelled", attempt_id=FIRST_ATTEMPT_ID),
    )

    assert state.status == "cancelled"
    assert state.cancel_requested_at == CREATED_AT + timedelta(seconds=2)
    assert state.finished_at == CREATED_AT + timedelta(seconds=3)
    assert state.attempts[0].status == "cancelled"
    assert state.attempts[0].retryable is False


def test_retry_preserves_attempt_history_and_respects_limit() -> None:
    state = transition_job(
        _job(max_attempts=2),
        _event(1, "attempt_started", attempt_id=FIRST_ATTEMPT_ID),
    )
    state = transition_job(
        state,
        _event(
            2,
            "attempt_failed",
            attempt_id=FIRST_ATTEMPT_ID,
            error_code="workspace_unavailable",
            retryable=True,
        ),
    )
    state = transition_job(state, _event(3, "retry_requested"))
    state = transition_job(
        state,
        _event(4, "attempt_started", attempt_id=SECOND_ATTEMPT_ID),
    )
    state = transition_job(
        state,
        _event(
            5,
            "attempt_interrupted",
            attempt_id=SECOND_ATTEMPT_ID,
            error_code="worker_interrupted",
            retryable=True,
        ),
    )

    assert state.status == "failed"
    assert [attempt.attempt_no for attempt in state.attempts] == [1, 2]
    assert [attempt.status for attempt in state.attempts] == [
        "failed",
        "interrupted",
    ]
    assert state.attempts[0].error_code == "workspace_unavailable"

    with pytest.raises(JobTransitionError) as raised:
        transition_job(state, _event(6, "retry_requested"))

    assert raised.value.code == JobTransitionErrorCode.RETRY_NOT_ALLOWED
