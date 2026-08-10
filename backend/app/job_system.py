"""长任务的框架无关 Job、Attempt、进度、取消和重试语义。"""

from datetime import datetime
from enum import StrEnum
from typing import Literal, Never
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


JobKind = Literal["workspace_scan", "document_index"]
JobStatus = Literal[
    "pending",
    "running",
    "cancel_requested",
    "succeeded",
    "failed",
    "cancelled",
]
AttemptStatus = Literal[
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
]
JobEventKind = Literal[
    "attempt_started",
    "progress_reported",
    "cancellation_requested",
    "attempt_succeeded",
    "attempt_failed",
    "attempt_cancelled",
    "attempt_interrupted",
    "retry_requested",
]


class JobTransitionErrorCode(StrEnum):
    """Job 状态转换失败时供程序稳定判断的错误码。"""

    JOB_MISMATCH = "job_mismatch"
    EVENT_SEQUENCE_MISMATCH = "event_sequence_mismatch"
    ATTEMPT_MISMATCH = "attempt_mismatch"
    INVALID_TRANSITION = "invalid_transition"
    PROGRESS_REGRESSION = "progress_regression"
    RETRY_NOT_ALLOWED = "retry_not_allowed"


class JobTransitionError(ValueError):
    """拒绝不属于当前 Job 或不符合生命周期规则的事件。"""

    def __init__(
        self,
        code: JobTransitionErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


class JobProgress(BaseModel):
    """一次 Attempt 内单调递增的结构化进度。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    completed_units: int = Field(default=0, ge=0)
    total_units: int | None = Field(default=None, ge=0)
    phase_code: str = Field(
        default="starting",
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )

    @model_validator(mode="after")
    def validate_units(self) -> "JobProgress":
        if (
            self.total_units is not None
            and self.completed_units > self.total_units
        ):
            raise ValueError("completed_units must not exceed total_units")
        return self


class JobAttempt(BaseModel):
    """一个 Job 的单次执行记录；重试必须创建新的 Attempt。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    attempt_id: UUID
    job_id: UUID
    attempt_no: int = Field(ge=1)
    status: AttemptStatus = "running"
    progress: JobProgress = Field(default_factory=JobProgress)
    started_at: datetime
    finished_at: datetime | None = None
    error_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,99}$",
    )
    retryable: bool = False

    @field_validator("started_at", "finished_at")
    @classmethod
    def validate_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and (
            value.tzinfo is None or value.utcoffset() is None
        ):
            raise ValueError("attempt timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_outcome(self) -> "JobAttempt":
        if self.status == "running":
            if self.finished_at is not None or self.error_code is not None:
                raise ValueError("running attempt must not contain an outcome")
            if self.retryable:
                raise ValueError("running attempt must not be retryable")
            return self

        if self.finished_at is None:
            raise ValueError("terminal attempt requires finished_at")
        if self.finished_at < self.started_at:
            raise ValueError("attempt must not finish before it starts")

        failed = self.status in {"failed", "interrupted"}
        if failed and self.error_code is None:
            raise ValueError("failed attempt requires error_code")
        if not failed and self.error_code is not None:
            raise ValueError("successful or cancelled attempt cannot contain error_code")
        if self.status == "interrupted" and not self.retryable:
            raise ValueError("interrupted attempt must remain retryable")
        if self.status in {"succeeded", "cancelled"} and self.retryable:
            raise ValueError("successful or cancelled attempt cannot be retryable")
        return self


class JobState(BaseModel):
    """一个逻辑长任务及其不可覆盖的 Attempt 历史。

    idempotency_key 标识同一次逻辑提交；唯一绑定由后续持久化边界负责。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    job_id: UUID
    kind: JobKind
    workspace_id: int = Field(ge=1)
    idempotency_key: str = Field(min_length=1, max_length=128)
    status: JobStatus = "pending"
    max_attempts: int = Field(ge=1, le=10)
    revision: int = Field(default=0, ge=0)
    created_at: datetime
    cancel_requested_at: datetime | None = None
    finished_at: datetime | None = None
    error_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,99}$",
    )
    attempts: tuple[JobAttempt, ...] = ()

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("idempotency_key must not contain surrounding whitespace")
        return value

    @field_validator("created_at", "cancel_requested_at", "finished_at")
    @classmethod
    def validate_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and (
            value.tzinfo is None or value.utcoffset() is None
        ):
            raise ValueError("job timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "JobState":
        if len(self.attempts) > self.max_attempts:
            raise ValueError("attempt count must not exceed max_attempts")

        seen_attempt_ids: set[UUID] = set()
        for expected_no, attempt in enumerate(self.attempts, start=1):
            if attempt.job_id != self.job_id:
                raise ValueError("attempt must belong to this job")
            if attempt.attempt_no != expected_no:
                raise ValueError("attempt numbers must be contiguous")
            if attempt.attempt_id in seen_attempt_ids:
                raise ValueError("attempt ids must be unique within a job")
            if attempt.started_at < self.created_at:
                raise ValueError("attempt must not start before its job")
            seen_attempt_ids.add(attempt.attempt_id)

        terminal = self.status in {"succeeded", "failed", "cancelled"}
        if terminal != (self.finished_at is not None):
            raise ValueError("only terminal job requires finished_at")
        if self.finished_at is not None and self.finished_at < self.created_at:
            raise ValueError("job must not finish before it is created")

        if self.status == "failed" and self.error_code is None:
            raise ValueError("failed job requires error_code")
        if self.status != "failed" and self.error_code is not None:
            raise ValueError("non-failed job cannot contain error_code")
        if self.cancel_requested_at is not None and (
            self.cancel_requested_at < self.created_at
        ):
            raise ValueError("cancellation must not predate the job")
        if self.status in {"cancel_requested", "cancelled"} and (
            self.cancel_requested_at is None
        ):
            raise ValueError("cancelled job requires a cancellation request")

        latest_attempt = self.attempts[-1] if self.attempts else None
        if self.status in {"running", "cancel_requested"}:
            if latest_attempt is None or latest_attempt.status != "running":
                raise ValueError("active job requires a running attempt")
        elif self.status == "succeeded":
            if latest_attempt is None or latest_attempt.status != "succeeded":
                raise ValueError("succeeded job requires a succeeded attempt")
        elif self.status == "failed":
            if latest_attempt is None or latest_attempt.status not in {
                "failed",
                "interrupted",
            }:
                raise ValueError("failed job requires a failed attempt")
            if latest_attempt.error_code != self.error_code:
                raise ValueError("job error_code must match its latest attempt")
        elif self.status == "cancelled" and latest_attempt is not None:
            if latest_attempt.status != "cancelled":
                raise ValueError("cancelled job requires a cancelled attempt")
        elif self.status == "pending" and latest_attempt is not None:
            if latest_attempt.status not in {"failed", "interrupted"}:
                raise ValueError("pending retry requires a failed attempt")
        return self


class JobEvent(BaseModel):
    """驱动一次纯 Job 状态转换的有序事件。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    job_id: UUID
    sequence_no: int = Field(ge=1)
    kind: JobEventKind
    occurred_at: datetime
    attempt_id: UUID | None = None
    completed_units: int | None = Field(default=None, ge=0)
    total_units: int | None = Field(default=None, ge=0)
    phase_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )
    error_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,99}$",
    )
    retryable: bool | None = None

    @field_validator("occurred_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_details(self) -> "JobEvent":
        attempt_events = {
            "attempt_started",
            "progress_reported",
            "attempt_succeeded",
            "attempt_failed",
            "attempt_cancelled",
            "attempt_interrupted",
        }
        if (self.kind in attempt_events) != (self.attempt_id is not None):
            raise ValueError("attempt event must identify exactly one attempt")

        progress_fields = (
            self.completed_units,
            self.total_units,
            self.phase_code,
        )
        if self.kind == "progress_reported":
            if self.completed_units is None or self.phase_code is None:
                raise ValueError("progress event requires completed units and phase")
        elif any(value is not None for value in progress_fields):
            raise ValueError("non-progress event cannot contain progress details")

        failure_event = self.kind in {"attempt_failed", "attempt_interrupted"}
        if failure_event:
            if self.error_code is None or self.retryable is None:
                raise ValueError("failed attempt event requires retry metadata")
            if self.kind == "attempt_interrupted" and not self.retryable:
                raise ValueError("interrupted attempt must be retryable")
        elif self.error_code is not None or self.retryable is not None:
            raise ValueError("non-failure event cannot contain retry metadata")
        return self


def transition_job(state: JobState, event: JobEvent) -> JobState:
    """应用一个有序事件；不执行持久化、线程调度或业务操作。"""

    if event.job_id != state.job_id:
        raise JobTransitionError(
            JobTransitionErrorCode.JOB_MISMATCH,
            "Job 事件不属于当前 Job",
        )
    if event.sequence_no != state.revision + 1:
        raise JobTransitionError(
            JobTransitionErrorCode.EVENT_SEQUENCE_MISMATCH,
            "Job 事件序号不连续",
        )
    if event.occurred_at < state.created_at:
        raise JobTransitionError(
            JobTransitionErrorCode.INVALID_TRANSITION,
            "Job 事件不能早于 Job 创建时间",
        )

    if event.kind == "attempt_started":
        return _start_attempt(state, event)
    if event.kind == "progress_reported":
        return _report_progress(state, event)
    if event.kind == "cancellation_requested":
        return _request_cancellation(state, event)
    if event.kind in {
        "attempt_succeeded",
        "attempt_failed",
        "attempt_cancelled",
        "attempt_interrupted",
    }:
        return _finish_attempt(state, event)
    if event.kind == "retry_requested":
        return _request_retry(state, event)

    raise JobTransitionError(
        JobTransitionErrorCode.INVALID_TRANSITION,
        f"未知 Job 事件：{event.kind}",
    )


def _start_attempt(state: JobState, event: JobEvent) -> JobState:
    if state.status != "pending":
        _invalid_transition(state, event)
    if len(state.attempts) >= state.max_attempts:
        raise JobTransitionError(
            JobTransitionErrorCode.RETRY_NOT_ALLOWED,
            "Job 已达到最大 Attempt 数量",
        )
    attempt_id = event.attempt_id
    if attempt_id is None or any(
        attempt.attempt_id == attempt_id for attempt in state.attempts
    ):
        raise JobTransitionError(
            JobTransitionErrorCode.ATTEMPT_MISMATCH,
            "Attempt 标识缺失或重复",
        )

    attempt = JobAttempt(
        attempt_id=attempt_id,
        job_id=state.job_id,
        attempt_no=len(state.attempts) + 1,
        started_at=event.occurred_at,
    )
    return _replace_job(
        state,
        status="running",
        revision=event.sequence_no,
        attempts=(*state.attempts, attempt),
    )


def _report_progress(state: JobState, event: JobEvent) -> JobState:
    attempt = _active_attempt(state, event)
    completed_units = event.completed_units
    phase_code = event.phase_code
    if completed_units is None or phase_code is None:
        _invalid_transition(state, event)
    if completed_units < attempt.progress.completed_units:
        raise JobTransitionError(
            JobTransitionErrorCode.PROGRESS_REGRESSION,
            "同一 Attempt 的进度不能倒退",
        )

    current_total = attempt.progress.total_units
    if current_total is not None and (
        event.total_units is not None and event.total_units != current_total
    ):
        raise JobTransitionError(
            JobTransitionErrorCode.PROGRESS_REGRESSION,
            "已确认的总工作量不能改变",
        )
    total_units = current_total if current_total is not None else event.total_units
    progress = JobProgress(
        completed_units=completed_units,
        total_units=total_units,
        phase_code=phase_code,
    )
    updated_attempt = _replace_attempt(attempt, progress=progress)
    return _replace_latest_attempt(state, event.sequence_no, updated_attempt)


def _request_cancellation(state: JobState, event: JobEvent) -> JobState:
    if state.status == "pending":
        return _replace_job(
            state,
            status="cancelled",
            revision=event.sequence_no,
            cancel_requested_at=event.occurred_at,
            finished_at=event.occurred_at,
        )
    if state.status == "running":
        return _replace_job(
            state,
            status="cancel_requested",
            revision=event.sequence_no,
            cancel_requested_at=event.occurred_at,
        )
    if state.status == "cancel_requested":
        return _replace_job(state, revision=event.sequence_no)
    _invalid_transition(state, event)


def _finish_attempt(state: JobState, event: JobEvent) -> JobState:
    attempt = _active_attempt(state, event)
    if event.occurred_at < attempt.started_at:
        _invalid_transition(state, event)
    if event.kind == "attempt_cancelled" and state.status != "cancel_requested":
        _invalid_transition(state, event)

    outcomes: dict[JobEventKind, tuple[AttemptStatus, JobStatus]] = {
        "attempt_succeeded": ("succeeded", "succeeded"),
        "attempt_failed": ("failed", "failed"),
        "attempt_cancelled": ("cancelled", "cancelled"),
        "attempt_interrupted": ("interrupted", "failed"),
    }
    attempt_status, job_status = outcomes[event.kind]
    updated_attempt = _replace_attempt(
        attempt,
        status=attempt_status,
        finished_at=event.occurred_at,
        error_code=event.error_code,
        retryable=event.retryable or False,
    )
    return _replace_job(
        state,
        status=job_status,
        revision=event.sequence_no,
        attempts=(*state.attempts[:-1], updated_attempt),
        finished_at=event.occurred_at,
        error_code=event.error_code,
    )


def _request_retry(state: JobState, event: JobEvent) -> JobState:
    latest_attempt = state.attempts[-1] if state.attempts else None
    if (
        state.status != "failed"
        or latest_attempt is None
        or not latest_attempt.retryable
        or len(state.attempts) >= state.max_attempts
    ):
        raise JobTransitionError(
            JobTransitionErrorCode.RETRY_NOT_ALLOWED,
            "当前 Job 不允许重试",
        )
    return _replace_job(
        state,
        status="pending",
        revision=event.sequence_no,
        cancel_requested_at=None,
        finished_at=None,
        error_code=None,
    )


def _active_attempt(state: JobState, event: JobEvent) -> JobAttempt:
    if state.status not in {"running", "cancel_requested"}:
        _invalid_transition(state, event)
    attempt = state.attempts[-1]
    if attempt.status != "running" or attempt.attempt_id != event.attempt_id:
        raise JobTransitionError(
            JobTransitionErrorCode.ATTEMPT_MISMATCH,
            "事件不属于当前运行中的 Attempt",
        )
    return attempt


def _replace_latest_attempt(
    state: JobState,
    revision: int,
    attempt: JobAttempt,
) -> JobState:
    return _replace_job(
        state,
        revision=revision,
        attempts=(*state.attempts[:-1], attempt),
    )


def _replace_job(state: JobState, **changes: object) -> JobState:
    return JobState.model_validate({**state.model_dump(), **changes})


def _replace_attempt(attempt: JobAttempt, **changes: object) -> JobAttempt:
    return JobAttempt.model_validate({**attempt.model_dump(), **changes})


def _invalid_transition(state: JobState, event: JobEvent) -> Never:
    raise JobTransitionError(
        JobTransitionErrorCode.INVALID_TRANSITION,
        f"状态 {state.status} 不接受事件 {event.kind}",
    )
