"""单进程 Job Runner，可选持久化但不负责进程间调度。"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from threading import Event, RLock
from uuid import UUID, uuid4

from .job_store import (
    JobStoreIdentityConflictError,
    SqlAlchemyJobStore,
)
from .job_system import (
    JobEvent,
    JobEventKind,
    JobKind,
    JobState,
    transition_job,
)


JobTask = Callable[["JobContext"], None]
JobClock = Callable[[], datetime]
_ERROR_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,99}$")


class JobRunnerClosedError(RuntimeError):
    """拒绝向已经关闭的单进程 Runner 提交新 Job。"""


class JobNotFoundError(KeyError):
    """请求的 Job 不存在于当前进程。"""


class JobIdentityConflictError(ValueError):
    """同一幂等键被用于不同的 Job 身份。"""


class JobCancelledError(Exception):
    """任务通过上下文观察到取消请求后主动停止。"""


class JobTaskError(Exception):
    """任务以稳定错误码报告一次可观察失败。"""

    def __init__(self, error_code: str, *, retryable: bool = False) -> None:
        if not _ERROR_CODE_PATTERN.fullmatch(error_code):
            raise ValueError("error_code must be a lowercase stable code")
        super().__init__(error_code)
        self.error_code = error_code
        self.retryable = retryable


@dataclass
class _JobRecord:
    state: JobState
    task: JobTask
    cancel_event: Event = field(default_factory=Event)


class JobContext:
    """传给任务函数的最小控制面，不暴露 Session 或文件系统对象。"""

    def __init__(
        self,
        runner: "SingleProcessJobRunner",
        job_id: UUID,
        attempt_id: UUID,
    ) -> None:
        self._runner = runner
        self.job_id = job_id
        self.attempt_id = attempt_id

    def report_progress(
        self,
        completed_units: int,
        *,
        total_units: int | None,
        phase_code: str,
    ) -> JobState:
        """报告当前 Attempt 进度；倒退会被 Job 状态机拒绝。"""

        return self._runner._report_progress(
            self.job_id,
            self.attempt_id,
            completed_units=completed_units,
            total_units=total_units,
            phase_code=phase_code,
        )

    def raise_if_cancelled(self) -> None:
        """在任务自己的安全边界检查取消请求，不强杀执行线程。"""

        self._runner._raise_if_cancelled(self.job_id)


class SingleProcessJobRunner:
    """只有一个 worker；接入 Store 时要求当前进程独占执行权。"""

    def __init__(
        self,
        *,
        clock: JobClock | None = None,
        store: SqlAlchemyJobStore | None = None,
    ) -> None:
        self._clock = clock or _utc_now
        self._store = store
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="filenest-job",
        )
        self._lock = RLock()
        self._jobs: dict[UUID, _JobRecord] = {}
        self._idempotency_index: dict[str, UUID] = {}
        self._closed = False

    def submit(
        self,
        *,
        kind: JobKind,
        workspace_id: int,
        idempotency_key: str,
        task: JobTask,
        max_attempts: int = 1,
    ) -> JobState:
        """登记并异步提交一个 Job；重复逻辑提交返回原 Job。"""

        if not callable(task):
            raise TypeError("task must be callable")

        with self._lock:
            self._ensure_open()
            candidate = JobState(
                job_id=uuid4(),
                kind=kind,
                workspace_id=workspace_id,
                idempotency_key=idempotency_key,
                max_attempts=max_attempts,
                created_at=self._clock(),
            )
            existing_id = self._idempotency_index.get(candidate.idempotency_key)
            if existing_id is not None:
                existing = self._jobs[existing_id].state
                if (
                    existing.kind != candidate.kind
                    or existing.workspace_id != candidate.workspace_id
                ):
                    raise JobIdentityConflictError(
                        "idempotency_key is bound to another job identity"
                )
                return existing

            state = candidate
            if self._store is not None:
                try:
                    state = self._store.create_or_get(candidate)
                except JobStoreIdentityConflictError as error:
                    raise JobIdentityConflictError(
                        "idempotency_key is bound to another job identity"
                    ) from error
                state = self._recover_persisted_state(state)
                if state.status != "pending":
                    return state

            record = _JobRecord(state=state, task=task)
            self._jobs[state.job_id] = record
            self._idempotency_index[state.idempotency_key] = state.job_id
            try:
                self._executor.submit(self._run, state.job_id)
            except BaseException:
                del self._jobs[state.job_id]
                del self._idempotency_index[state.idempotency_key]
                raise
            return state

    def get(self, job_id: UUID) -> JobState:
        """读取当前进程内 Job 的不可变快照。"""

        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                return record.state
            if self._store is not None:
                state = self._store.get(job_id)
                if state is not None:
                    return state
            raise JobNotFoundError(job_id)

    def cancel(self, job_id: UUID) -> JobState:
        """发出取消请求；运行中的任务必须自行观察并停止。"""

        with self._lock:
            record = self._get_record(job_id)
            state = self._apply_event(
                record,
                "cancellation_requested",
            )
            record.cancel_event.set()
            return state

    def shutdown(self, *, wait: bool = True) -> None:
        """关闭当前 Runner；执行线程不会转移到其他进程。"""

        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def _run(self, job_id: UUID) -> None:
        with self._lock:
            record = self._get_record(job_id)
            if record.state.status != "pending":
                return
            attempt_id = uuid4()
            self._apply_event(
                record,
                "attempt_started",
                attempt_id=attempt_id,
            )

        context = JobContext(self, job_id, attempt_id)
        try:
            record.task(context)
        except JobCancelledError:
            self._finish_attempt(
                job_id,
                attempt_id,
                "attempt_cancelled",
            )
        except JobTaskError as error:
            self._finish_attempt(
                job_id,
                attempt_id,
                "attempt_failed",
                error_code=error.error_code,
                retryable=error.retryable,
            )
        except Exception:
            # 不把异常文本或堆栈写入 Job 状态，外部只依赖稳定错误码。
            self._finish_attempt(
                job_id,
                attempt_id,
                "attempt_failed",
                error_code="job_task_failed",
                retryable=False,
            )
        else:
            # 取消是协作式的；任务在观察取消前正常返回时，完成结果获胜。
            self._finish_attempt(
                job_id,
                attempt_id,
                "attempt_succeeded",
            )

    def _report_progress(
        self,
        job_id: UUID,
        attempt_id: UUID,
        *,
        completed_units: int,
        total_units: int | None,
        phase_code: str,
    ) -> JobState:
        with self._lock:
            record = self._get_record(job_id)
            return self._apply_event(
                record,
                "progress_reported",
                attempt_id=attempt_id,
                completed_units=completed_units,
                total_units=total_units,
                phase_code=phase_code,
            )

    def _raise_if_cancelled(self, job_id: UUID) -> None:
        with self._lock:
            record = self._get_record(job_id)
            if (
                record.cancel_event.is_set()
                or record.state.status == "cancel_requested"
            ):
                raise JobCancelledError()

    def _finish_attempt(
        self,
        job_id: UUID,
        attempt_id: UUID,
        kind: JobEventKind,
        *,
        error_code: str | None = None,
        retryable: bool | None = None,
    ) -> JobState:
        with self._lock:
            record = self._get_record(job_id)
            return self._apply_event(
                record,
                kind,
                attempt_id=attempt_id,
                error_code=error_code,
                retryable=retryable,
            )

    def _apply_event(
        self,
        record: _JobRecord,
        kind: JobEventKind,
        **details: object,
    ) -> JobState:
        record.state = self._apply_event_to_state(
            record.state,
            kind,
            **details,
        )
        return record.state

    def _apply_event_to_state(
        self,
        state: JobState,
        kind: JobEventKind,
        **details: object,
    ) -> JobState:
        event = JobEvent(
            job_id=state.job_id,
            sequence_no=state.revision + 1,
            kind=kind,
            occurred_at=self._clock(),
            **details,
        )
        next_state = transition_job(state, event)
        if self._store is None:
            return next_state
        return self._store.save_transition(
            expected_revision=state.revision,
            state=next_state,
        )

    def _recover_persisted_state(self, state: JobState) -> JobState:
        """重新登记任务时收敛上一个独占 worker 留下的活动状态。"""

        if state.status == "running":
            state = self._apply_event_to_state(
                state,
                "attempt_interrupted",
                attempt_id=state.attempts[-1].attempt_id,
                error_code="worker_interrupted",
                retryable=True,
            )
        elif state.status == "cancel_requested":
            return self._apply_event_to_state(
                state,
                "attempt_cancelled",
                attempt_id=state.attempts[-1].attempt_id,
            )

        latest_attempt = state.attempts[-1] if state.attempts else None
        if (
            state.status == "failed"
            and latest_attempt is not None
            and latest_attempt.status == "interrupted"
            and latest_attempt.retryable
            and len(state.attempts) < state.max_attempts
        ):
            return self._apply_event_to_state(state, "retry_requested")
        return state

    def _get_record(self, job_id: UUID) -> _JobRecord:
        try:
            return self._jobs[job_id]
        except KeyError as error:
            raise JobNotFoundError(job_id) from error

    def _ensure_open(self) -> None:
        if self._closed:
            raise JobRunnerClosedError("job runner is closed")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
