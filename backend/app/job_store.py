"""基于 SQLAlchemy 的最小 Job/Attempt 持久化边界。"""

from collections.abc import Callable
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from pydantic import ValidationError

from .job_system import (
    JobAttempt,
    JobKind,
    JobProgress,
    JobState,
    JobStatus,
    JobTaskPayload,
    hash_job_payload,
    serialize_job_payload,
)
from .models import JobAttemptRecord, JobRecord


class JobStoreError(RuntimeError):
    """Job 持久化失败。"""


class JobStoreIdentityConflictError(JobStoreError):
    """幂等键已经绑定到另一种 Job 身份。"""


class JobStoreRevisionConflictError(JobStoreError):
    """调用方尝试用陈旧 Job 快照覆盖较新状态。"""


class JobStoreAttemptConflictError(JobStoreError):
    """调用方尝试替换已经持久化的 Attempt 身份。"""


class JobStoreNotFoundError(JobStoreError):
    """指定 Job 不存在。"""


class JobStorePayloadError(JobStoreError):
    """数据库中的任务描述无法安全还原。"""


class SqlAlchemyJobStore:
    """持久化 Job 快照，并用 revision 保护并发状态转换。"""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def create_or_get(self, candidate: JobState) -> JobState:
        """创建初始 Job；重复幂等提交返回原 Job。"""

        if (
            candidate.status != "pending"
            or candidate.revision != 0
            or candidate.attempts
        ):
            raise ValueError("only a new pending Job can be created")

        with self._session_factory() as session:
            existing = self._get_by_idempotency_key(
                session,
                candidate.idempotency_key,
            )
            if existing is not None:
                self._require_same_identity(existing, candidate)
                return self._load_state(session, UUID(existing.job_id))

            session.add(self._new_job_record(candidate))
            try:
                session.commit()
            except IntegrityError as error:
                session.rollback()
                existing = self._get_by_idempotency_key(
                    session,
                    candidate.idempotency_key,
                )
                if existing is None:
                    raise JobStoreError("failed to create Job") from error
                self._require_same_identity(existing, candidate)
                return self._load_state(session, UUID(existing.job_id))

            session.expire_all()
            return self._load_state(session, candidate.job_id)

    def get(self, job_id: UUID) -> JobState | None:
        """按 Job ID 读取完整 Job 与有序 Attempt 历史。"""

        with self._session_factory() as session:
            record = session.get(JobRecord, str(job_id))
            if record is None:
                return None
            return self._state_from_records(
                record,
                self._get_attempts(session, job_id),
            )

    def list_unfinished(self) -> list[JobState]:
        """读取启动恢复所需的全部未终态 Job 快照。"""

        with self._session_factory() as session:
            records = list(
                session.scalars(
                    select(JobRecord)
                    .where(
                        JobRecord.status.in_(
                            ("pending", "running", "cancel_requested")
                        )
                    )
                    .order_by(JobRecord.created_at.asc(), JobRecord.job_id.asc())
                ).all()
            )
            return [
                self._state_from_records(
                    record,
                    self._get_attempts(session, UUID(record.job_id)),
                )
                for record in records
            ]

    def list_jobs(
        self,
        *,
        workspace_id: int,
        kind: JobKind | None = None,
        status: JobStatus | None = None,
    ) -> list[JobState]:
        """按工作区读取 Job；可选条件只在数据库查询边界生效。"""

        with self._session_factory() as session:
            statement = select(JobRecord).where(
                JobRecord.workspace_id == workspace_id
            )
            if kind is not None:
                statement = statement.where(JobRecord.kind == kind)
            if status is not None:
                statement = statement.where(JobRecord.status == status)
            records = list(
                session.scalars(
                    statement.order_by(
                        JobRecord.created_at.desc(),
                        JobRecord.job_id.desc(),
                    )
                ).all()
            )
            return [
                self._state_from_records(
                    record,
                    self._get_attempts(session, UUID(record.job_id)),
                )
                for record in records
            ]

    def save_transition(
        self,
        *,
        expected_revision: int,
        state: JobState,
    ) -> JobState:
        """原子保存恰好一次状态转换及其 Attempt 变化。"""

        if expected_revision < 0 or state.revision != expected_revision + 1:
            raise ValueError("state revision must advance exactly once")

        with self._session_factory() as session:
            try:
                record = session.get(JobRecord, str(state.job_id))
                if record is None:
                    raise JobStoreNotFoundError(str(state.job_id))
                self._require_same_definition(record, state)

                persisted_attempts = self._get_attempts(session, state.job_id)
                self._require_compatible_attempts(
                    persisted_attempts,
                    state.attempts,
                )

                result = session.execute(
                    update(JobRecord)
                    .where(
                        JobRecord.job_id == str(state.job_id),
                        JobRecord.revision == expected_revision,
                    )
                    .values(
                        status=state.status,
                        revision=state.revision,
                        cancel_requested_at=_as_optional_utc(
                            state.cancel_requested_at
                        ),
                        finished_at=_as_optional_utc(state.finished_at),
                        error_code=state.error_code,
                    ),
                    execution_options={"synchronize_session": False},
                )
                if result.rowcount != 1:
                    raise JobStoreRevisionConflictError(str(state.job_id))

                self._save_attempts(
                    session,
                    persisted_attempts,
                    state.attempts,
                )
                session.commit()
                session.expire_all()
                return self._load_state(session, state.job_id)
            except Exception:
                session.rollback()
                raise

    @staticmethod
    def _new_job_record(state: JobState) -> JobRecord:
        return JobRecord(
            job_id=str(state.job_id),
            schema_version=state.schema_version,
            kind=state.kind,
            task_version=state.task_version,
            workspace_id=state.workspace_id,
            payload_json=serialize_job_payload(state.payload),
            payload_hash=state.payload_hash or hash_job_payload(state.payload),
            idempotency_key=state.idempotency_key,
            status=state.status,
            max_attempts=state.max_attempts,
            revision=state.revision,
            created_at=_as_utc(state.created_at),
            cancel_requested_at=_as_optional_utc(
                state.cancel_requested_at
            ),
            finished_at=_as_optional_utc(state.finished_at),
            error_code=state.error_code,
        )

    @staticmethod
    def _get_by_idempotency_key(
        session: Session,
        idempotency_key: str,
    ) -> JobRecord | None:
        return session.scalar(
            select(JobRecord).where(
                JobRecord.idempotency_key == idempotency_key
            )
        )

    @staticmethod
    def _get_attempts(
        session: Session,
        job_id: UUID,
    ) -> list[JobAttemptRecord]:
        statement = (
            select(JobAttemptRecord)
            .where(JobAttemptRecord.job_id == str(job_id))
            .order_by(JobAttemptRecord.attempt_no.asc())
        )
        return list(session.scalars(statement).all())

    def _load_state(self, session: Session, job_id: UUID) -> JobState:
        record = session.get(JobRecord, str(job_id))
        if record is None:
            raise JobStoreNotFoundError(str(job_id))
        return self._state_from_records(
            record,
            self._get_attempts(session, job_id),
        )

    @staticmethod
    def _require_same_identity(
        record: JobRecord,
        state: JobState,
    ) -> None:
        persisted_payload = _parse_job_payload(record.payload_json)
        if (
            record.kind != state.kind
            or record.task_version != state.task_version
            or record.workspace_id != state.workspace_id
            or record.payload_hash != state.payload_hash
            or serialize_job_payload(persisted_payload)
            != serialize_job_payload(state.payload)
            or record.idempotency_key != state.idempotency_key
        ):
            raise JobStoreIdentityConflictError(state.idempotency_key)

    @classmethod
    def _require_same_definition(
        cls,
        record: JobRecord,
        state: JobState,
    ) -> None:
        cls._require_same_identity(record, state)
        if (
            record.schema_version != state.schema_version
            or record.max_attempts != state.max_attempts
            or _as_utc(record.created_at) != _as_utc(state.created_at)
        ):
            raise JobStoreIdentityConflictError(str(state.job_id))

    @staticmethod
    def _require_compatible_attempts(
        persisted: list[JobAttemptRecord],
        incoming: tuple[JobAttempt, ...],
    ) -> None:
        if not len(persisted) <= len(incoming) <= len(persisted) + 1:
            raise JobStoreAttemptConflictError(
                "Attempt history must append at most one record"
            )
        for record, attempt in zip(persisted, incoming, strict=False):
            if (
                record.attempt_no != attempt.attempt_no
                or record.attempt_id != str(attempt.attempt_id)
                or record.schema_version != attempt.schema_version
                or record.job_id != str(attempt.job_id)
                or _as_utc(record.started_at)
                != _as_utc(attempt.started_at)
            ):
                raise JobStoreAttemptConflictError(str(attempt.attempt_id))

    @staticmethod
    def _save_attempts(
        session: Session,
        persisted: list[JobAttemptRecord],
        incoming: tuple[JobAttempt, ...],
    ) -> None:
        for index, attempt in enumerate(incoming):
            if index < len(persisted):
                record = persisted[index]
                record.status = attempt.status
                record.completed_units = attempt.progress.completed_units
                record.total_units = attempt.progress.total_units
                record.phase_code = attempt.progress.phase_code
                record.finished_at = _as_optional_utc(
                    attempt.finished_at
                )
                record.error_code = attempt.error_code
                record.retryable = attempt.retryable
                continue

            session.add(
                JobAttemptRecord(
                    attempt_id=str(attempt.attempt_id),
                    schema_version=attempt.schema_version,
                    job_id=str(attempt.job_id),
                    attempt_no=attempt.attempt_no,
                    status=attempt.status,
                    completed_units=attempt.progress.completed_units,
                    total_units=attempt.progress.total_units,
                    phase_code=attempt.progress.phase_code,
                    started_at=_as_utc(attempt.started_at),
                    finished_at=_as_optional_utc(attempt.finished_at),
                    error_code=attempt.error_code,
                    retryable=attempt.retryable,
                )
            )

    @staticmethod
    def _state_from_records(
        record: JobRecord,
        attempts: list[JobAttemptRecord],
    ) -> JobState:
        payload = _parse_job_payload(record.payload_json)
        try:
            return JobState(
                schema_version=record.schema_version,
                job_id=UUID(record.job_id),
                kind=record.kind,
                task_version=record.task_version,
                workspace_id=record.workspace_id,
                payload=payload,
                payload_hash=record.payload_hash,
                idempotency_key=record.idempotency_key,
                status=record.status,
                max_attempts=record.max_attempts,
                revision=record.revision,
                created_at=_as_utc(record.created_at),
                cancel_requested_at=_as_optional_utc(
                    record.cancel_requested_at
                ),
                finished_at=_as_optional_utc(record.finished_at),
                error_code=record.error_code,
                attempts=tuple(
                    JobAttempt(
                        schema_version=attempt.schema_version,
                        attempt_id=UUID(attempt.attempt_id),
                        job_id=UUID(attempt.job_id),
                        attempt_no=attempt.attempt_no,
                        status=attempt.status,
                        progress=JobProgress(
                            completed_units=attempt.completed_units,
                            total_units=attempt.total_units,
                            phase_code=attempt.phase_code,
                        ),
                        started_at=_as_utc(attempt.started_at),
                        finished_at=_as_optional_utc(attempt.finished_at),
                        error_code=attempt.error_code,
                        retryable=attempt.retryable,
                    )
                    for attempt in attempts
                ),
            )
        except ValidationError as error:
            raise JobStorePayloadError(
                f"invalid persisted Job definition: {record.job_id}"
            ) from error


def _parse_job_payload(payload_json: str) -> JobTaskPayload:
    try:
        return JobTaskPayload.model_validate_json(payload_json)
    except (TypeError, ValueError, ValidationError) as error:
        raise JobStorePayloadError("invalid persisted Job payload") from error


def _as_utc(value: datetime) -> datetime:
    """SQLite 会丢失时区标记；存储边界统一恢复为 UTC。"""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _as_optional_utc(value: datetime | None) -> datetime | None:
    return None if value is None else _as_utc(value)
