"""单进程 Job Runner，可选持久化但不负责进程间调度。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from threading import Event, RLock
from uuid import UUID, uuid4

from .job_store import (
    JobStoreAttemptConflictError,
    JobStoreIdentityConflictError,
    JobStoreRevisionConflictError,
    SqlAlchemyJobStore,
)
from .job_system import (
    JobEvent,
    JobEventKind,
    JobKind,
    JobState,
    JobTaskPayload,
    JobTaskVersion,
    TASK_VERSION_PATTERN,
    JobTransitionError,
    transition_job,
)


JobTask = Callable[["JobContext"], None]
JobHandler = Callable[["JobContext", JobTaskPayload], None]
JobClock = Callable[[], datetime]
JobRecoveryValidator = Callable[[JobState], bool]
_ERROR_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,99}$")
_TASK_VERSION_PATTERN = re.compile(TASK_VERSION_PATTERN)


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


class JobActionConflictError(ValueError):
    """取消或重试与当前 Job 终态/能力不兼容。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class JobHandlerUnavailableError(LookupError):
    """任务描述没有对应的显式版本化 Handler。"""


class JobHandlerRegistry:
    """有限的 kind/version 到可重建 Handler 映射。"""

    def __init__(
        self,
        handlers: Mapping[tuple[JobKind, JobTaskVersion], JobHandler]
        | None = None,
    ) -> None:
        self._handlers: dict[tuple[JobKind, JobTaskVersion], JobHandler] = {}
        for (kind, task_version), handler in (handlers or {}).items():
            self.register(
                kind=kind,
                task_version=task_version,
                handler=handler,
            )

    def register(
        self,
        *,
        kind: JobKind,
        task_version: JobTaskVersion,
        handler: JobHandler,
    ) -> None:
        """注册一个明确的任务版本；不允许模糊 fallback。"""

        if kind not in {"workspace_scan", "document_index"}:
            raise ValueError("unsupported Job kind")
        if not _TASK_VERSION_PATTERN.fullmatch(task_version):
            raise ValueError("invalid task version")
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._handlers[(kind, task_version)] = handler

    def resolve(
        self,
        *,
        kind: JobKind,
        task_version: JobTaskVersion,
    ) -> JobHandler:
        try:
            return self._handlers[(kind, task_version)]
        except KeyError as error:
            raise JobHandlerUnavailableError(
                f"no handler for {kind}:{task_version}"
            ) from error


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
        handler_registry: JobHandlerRegistry | None = None,
    ) -> None:
        self._clock = clock or _utc_now
        self._store = store
        self._handler_registry = handler_registry
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
        task: JobTask | None = None,
        payload: JobTaskPayload | Mapping[str, object] | None = None,
        task_version: JobTaskVersion = "v1",
        max_attempts: int = 1,
    ) -> JobState:
        """登记并异步提交一个 Job；重复逻辑提交返回原 Job。"""

        if task is not None and not callable(task):
            raise TypeError("task must be callable")
        task_payload = (
            payload
            if isinstance(payload, JobTaskPayload)
            else JobTaskPayload.model_validate(
                payload
                if payload is not None
                else {"workspace_id": workspace_id}
            )
        )

        with self._lock:
            self._ensure_open()
            candidate = JobState(
                job_id=uuid4(),
                kind=kind,
                task_version=task_version,
                workspace_id=workspace_id,
                payload=task_payload,
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
                    or existing.task_version != candidate.task_version
                    or existing.payload != candidate.payload
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

            runtime_task = self._resolve_task(task, state)
            record = _JobRecord(state=state, task=runtime_task)
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

    def list_jobs(
        self,
        *,
        workspace_id: int,
        kind: JobKind | None = None,
        status: str | None = None,
    ) -> list[JobState]:
        """按 workspace 读取 Job；持久化 Runner 以 Store 为准。"""

        with self._lock:
            if self._store is not None:
                return self._store.list_jobs(
                    workspace_id=workspace_id,
                    kind=kind,
                    status=status,  # type: ignore[arg-type]
                )
            return sorted(
                (
                    record.state
                    for record in self._jobs.values()
                    if record.state.workspace_id == workspace_id
                    and (kind is None or record.state.kind == kind)
                    and (status is None or record.state.status == status)
                ),
                key=lambda state: (state.created_at, str(state.job_id)),
                reverse=True,
            )

    def cancel(self, job_id: UUID) -> JobState:
        """发出取消请求；运行中的任务必须自行观察并停止。"""

        with self._lock:
            record = self._get_action_record(job_id)
            if record.state.status in {"succeeded", "failed", "cancelled"}:
                raise JobActionConflictError("job_cancel_not_allowed")
            if record.state.status == "cancel_requested":
                return record.state
            try:
                state = self._apply_event(
                    record,
                    "cancellation_requested",
                )
            except (
                JobStoreAttemptConflictError,
                JobStoreRevisionConflictError,
                JobTransitionError,
            ) as error:
                raise JobActionConflictError("job_state_changed") from error
            record.cancel_event.set()
            return state

    def retry(self, job_id: UUID) -> JobState:
        """为可重试失败 Job 创建新的 Attempt，不覆盖旧历史。"""

        with self._lock:
            record = self._get_action_record(job_id)
            latest_attempt = record.state.attempts[-1] if record.state.attempts else None
            if (
                record.state.status != "failed"
                or latest_attempt is None
                or not latest_attempt.retryable
                or len(record.state.attempts) >= record.state.max_attempts
            ):
                raise JobActionConflictError("job_retry_not_allowed")
            try:
                state = self._apply_event(record, "retry_requested")
            except (
                JobStoreAttemptConflictError,
                JobStoreRevisionConflictError,
            ) as error:
                raise JobActionConflictError("job_state_changed") from error
            self._schedule_record(record)
            return state

    def shutdown(self, *, wait: bool = True) -> None:
        """关闭当前 Runner；执行线程不会转移到其他进程。"""

        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def recover_persisted_jobs(
        self,
        *,
        can_run: JobRecoveryValidator | None = None,
    ) -> tuple[JobState, ...]:
        """扫描并以 CAS 收敛或重新排队上次进程留下的 Job。"""

        if self._store is None:
            return ()

        with self._lock:
            self._ensure_open()
            recovered: list[JobState] = []
            for initial_state in self._store.list_unfinished():
                if initial_state.job_id in self._jobs:
                    continue

                state = initial_state
                if state.status == "cancel_requested":
                    try:
                        state = self._apply_event_to_state(
                            state,
                            "attempt_cancelled",
                            attempt_id=state.attempts[-1].attempt_id,
                        )
                    except (
                        JobStoreAttemptConflictError,
                        JobStoreRevisionConflictError,
                    ):
                        latest = self._store.get(state.job_id)
                        state = latest if latest is not None else state
                    if state is not None:
                        recovered.append(state)
                    continue

                can_requeue = True
                if can_run is not None:
                    try:
                        can_requeue = can_run(state)
                    except Exception:
                        can_requeue = False
                if not can_requeue:
                    try:
                        state = self._fail_recovery(state)
                    except (
                        JobStoreAttemptConflictError,
                        JobStoreRevisionConflictError,
                    ):
                        latest = self._store.get(state.job_id)
                        state = latest if latest is not None else state
                    if state is not None:
                        recovered.append(state)
                    continue

                try:
                    state = self._recover_persisted_state(state)
                except (
                    JobStoreAttemptConflictError,
                    JobStoreRevisionConflictError,
                ):
                    latest = self._store.get(state.job_id)
                    if latest is None or latest.status == "running":
                        if latest is not None:
                            recovered.append(latest)
                        continue
                    state = latest

                if state.status == "pending":
                    self._enqueue_persisted_state(state)
                recovered.append(state)
            return tuple(recovered)

    def _run(self, job_id: UUID) -> None:
        with self._lock:
            record = self._get_record(job_id)
            if record.state.status != "pending":
                return
            attempt_id = uuid4()
            try:
                self._apply_event(
                    record,
                    "attempt_started",
                    attempt_id=attempt_id,
                )
            except (
                JobStoreAttemptConflictError,
                JobStoreRevisionConflictError,
            ):
                if self._store is not None:
                    latest = self._store.get(job_id)
                    if latest is not None:
                        record.state = latest
                return

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

    def _fail_recovery(self, state: JobState) -> JobState:
        """在 workspace/Policy 失效时留下失败证据，不触发任务 Handler。"""

        if state.status == "running":
            return self._apply_event_to_state(
                state,
                "attempt_failed",
                attempt_id=state.attempts[-1].attempt_id,
                error_code="recovery_required",
                retryable=False,
            )

        if state.status == "pending":
            attempt_id = uuid4()
            state = self._apply_event_to_state(
                state,
                "attempt_started",
                attempt_id=attempt_id,
            )
            return self._apply_event_to_state(
                state,
                "attempt_failed",
                attempt_id=attempt_id,
                error_code="recovery_required",
                retryable=False,
            )

        raise ValueError("only active Jobs can require recovery failure")

    def _enqueue_persisted_state(self, state: JobState) -> None:
        if state.status != "pending" or state.job_id in self._jobs:
            return

        record = _JobRecord(
            state=state,
            task=self._resolve_task(None, state),
        )
        self._jobs[state.job_id] = record
        self._idempotency_index[state.idempotency_key] = state.job_id
        self._schedule_record(record)

    def _schedule_record(self, record: _JobRecord) -> None:
        if record.state.status != "pending":
            return
        try:
            self._executor.submit(self._run, record.state.job_id)
        except BaseException:
            if self._jobs.get(record.state.job_id) is record:
                del self._jobs[record.state.job_id]
                del self._idempotency_index[record.state.idempotency_key]
            raise

    def _get_action_record(self, job_id: UUID) -> _JobRecord:
        record = self._jobs.get(job_id)
        if record is not None:
            return record
        if self._store is None:
            raise JobNotFoundError(job_id)
        state = self._store.get(job_id)
        if state is None:
            raise JobNotFoundError(job_id)
        record = _JobRecord(
            state=state,
            task=self._resolve_task(None, state),
        )
        self._jobs[state.job_id] = record
        self._idempotency_index[state.idempotency_key] = state.job_id
        return record

    def _resolve_task(
        self,
        task: JobTask | None,
        state: JobState,
    ) -> JobTask:
        """优先兼容旧直接调用；可重建任务必须经过注册表解析。"""

        if task is not None:
            return task
        if self._handler_registry is None:
            return _handler_unavailable_task
        try:
            handler = self._handler_registry.resolve(
                kind=state.kind,
                task_version=state.task_version,
            )
        except JobHandlerUnavailableError:
            return _handler_unavailable_task
        return _bound_handler_task(handler, state.payload)

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


def _bound_handler_task(
    handler: JobHandler,
    payload: JobTaskPayload,
) -> JobTask:
    def run(context: JobContext) -> None:
        handler(context, payload)

    return run


def _handler_unavailable_task(_context: JobContext) -> None:
    raise JobTaskError("recovery_required", retryable=False)
