from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import FileEntry, Workspace
from backend.app.operation_plan import (
    ContentHash,
    FilePrecondition,
    OperationPlan,
    OperationPlanItem,
    OperationReason,
)
from backend.app.path_policy import PathPolicyError
from backend.app.services import (
    FileEntryNotFoundError,
    OperationPlanExpiredError,
    OperationPlanSourceMismatchError,
    OperationPlanSourceChangedError,
    OperationPlanTargetConflictError,
    OperationPlanTargetUnavailableError,
    WorkspaceNotFoundError,
    validate_operation_plan,
)


PLAN_CREATED_AT = datetime(2026, 8, 30, 8, 0, tzinfo=timezone.utc)


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'operation-plan.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _seed_workspace(
    session: Session,
    workspace_root: Path,
) -> tuple[int, int, int, FilePrecondition]:
    source_path = workspace_root / "inbox" / "report.pdf"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"operation plan source")
    (workspace_root / "documents" / "reports").mkdir(parents=True)

    workspace = Workspace(name="计划工作区", root_path=str(workspace_root))
    other_workspace = Workspace(
        name="其他工作区",
        root_path=str(workspace_root.parent / "other-workspace"),
    )
    session.add_all([workspace, other_workspace])
    session.flush()

    source_stat = source_path.stat()
    source_entry = FileEntry(
        workspace_id=workspace.id,
        relative_path="inbox/report.pdf",
        name="report.pdf",
        extension=".pdf",
        size_bytes=source_stat.st_size,
        mtime_ns=source_stat.st_mtime_ns,
    )
    other_entry = FileEntry(
        workspace_id=other_workspace.id,
        relative_path="private.txt",
        name="private.txt",
        extension=".txt",
        size_bytes=1,
        mtime_ns=1,
    )
    session.add_all([source_entry, other_entry])
    session.commit()

    source_precondition = FilePrecondition(
        size_bytes=source_stat.st_size,
        mtime_ns=source_stat.st_mtime_ns,
    )
    return workspace.id, source_entry.id, other_entry.id, source_precondition


def _plan(
    workspace_id: int,
    source_file_id: int,
    *,
    created_at: datetime = PLAN_CREATED_AT,
    **operation_overrides: object,
) -> OperationPlan:
    operation_values: dict[str, object] = {
        "source_file_id": source_file_id,
        "source_relative_path": "inbox/report.pdf",
        "target_relative_path": "documents/reports/report.pdf",
        "source_precondition": FilePrecondition(size_bytes=21, mtime_ns=1),
        "reason": OperationReason(
            kind="manual_selection",
            description="由用户确认目标目录",
        ),
    }
    operation_values.update(operation_overrides)
    return OperationPlan(
        plan_id=UUID("2d053752-d3c4-45cb-b696-bd043e78ed92"),
        workspace_id=workspace_id,
        created_at=created_at,
        operations=[OperationPlanItem(**operation_values)],
    )


def test_validate_operation_plan_accepts_available_target_without_mutation(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    with Session(engine) as session:
        workspace_id, source_file_id, _, precondition = _seed_workspace(
            session,
            workspace_root,
        )
        plan = _plan(
            workspace_id,
            source_file_id,
            source_precondition=precondition,
        )

        validate_operation_plan(session, plan, now=PLAN_CREATED_AT)

        assert not (workspace_root / "documents" / "reports" / "report.pdf").exists()
        assert not session.new
        assert not session.dirty
        assert not session.deleted


def test_validate_operation_plan_rejects_missing_workspace(engine: Engine) -> None:
    plan = _plan(404, 1)

    with Session(engine) as session, pytest.raises(WorkspaceNotFoundError):
        validate_operation_plan(session, plan, now=PLAN_CREATED_AT)


def test_validate_operation_plan_hides_cross_workspace_source(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    with Session(engine) as session:
        workspace_id, _, other_file_id, _ = _seed_workspace(
            session,
            workspace_root,
        )
        plan = _plan(workspace_id, other_file_id)

        with pytest.raises(FileEntryNotFoundError):
            validate_operation_plan(session, plan, now=PLAN_CREATED_AT)


def test_validate_operation_plan_rejects_source_path_mismatch(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    with Session(engine) as session:
        workspace_id, source_file_id, _, _ = _seed_workspace(
            session,
            workspace_root,
        )
        plan = _plan(
            workspace_id,
            source_file_id,
            source_relative_path="inbox/renamed-report.pdf",
        )

        with pytest.raises(OperationPlanSourceMismatchError):
            validate_operation_plan(session, plan, now=PLAN_CREATED_AT)


def test_validate_operation_plan_rejects_existing_target(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    with Session(engine) as session:
        workspace_id, source_file_id, _, precondition = _seed_workspace(
            session,
            workspace_root,
        )
        target = workspace_root / "documents" / "reports" / "report.pdf"
        target.write_bytes(b"existing target")
        plan = _plan(
            workspace_id,
            source_file_id,
            source_precondition=precondition,
        )

        with pytest.raises(OperationPlanTargetConflictError):
            validate_operation_plan(session, plan, now=PLAN_CREATED_AT)

    assert target.read_bytes() == b"existing target"


def test_validate_operation_plan_rejects_missing_target_parent(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    with Session(engine) as session:
        workspace_id, source_file_id, _, precondition = _seed_workspace(
            session,
            workspace_root,
        )
        plan = _plan(
            workspace_id,
            source_file_id,
            target_relative_path="missing/report.pdf",
            source_precondition=precondition,
        )

        with pytest.raises(OperationPlanTargetUnavailableError):
            validate_operation_plan(session, plan, now=PLAN_CREATED_AT)


def test_validate_operation_plan_preserves_path_policy_errors(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    with Session(engine) as session:
        workspace_id, source_file_id, _, precondition = _seed_workspace(
            session,
            workspace_root,
        )
        plan = _plan(
            workspace_id,
            source_file_id,
            target_relative_path=".git/report.pdf",
            source_precondition=precondition,
        )

        with pytest.raises(PathPolicyError) as error:
            validate_operation_plan(session, plan, now=PLAN_CREATED_AT)

    assert error.value.code.value == "sensitive_path"


@pytest.mark.parametrize(
    "created_at",
    [
        PLAN_CREATED_AT - timedelta(minutes=15, microseconds=1),
        PLAN_CREATED_AT + timedelta(microseconds=1),
    ],
)
def test_validate_operation_plan_rejects_expired_or_future_plan(
    engine: Engine,
    created_at: datetime,
) -> None:
    plan = _plan(1, 1, created_at=created_at)

    with Session(engine) as session, pytest.raises(OperationPlanExpiredError):
        validate_operation_plan(session, plan, now=PLAN_CREATED_AT)


@pytest.mark.parametrize("changed_field", ["size_bytes", "mtime_ns"])
def test_validate_operation_plan_rejects_changed_source_metadata(
    engine: Engine,
    tmp_path: Path,
    changed_field: str,
) -> None:
    workspace_root = tmp_path / "workspace"
    with Session(engine) as session:
        workspace_id, source_file_id, _, precondition = _seed_workspace(
            session,
            workspace_root,
        )
        values = precondition.model_dump()
        values[changed_field] += 1
        plan = _plan(
            workspace_id,
            source_file_id,
            source_precondition=FilePrecondition(**values),
        )

        with pytest.raises(OperationPlanSourceChangedError):
            validate_operation_plan(session, plan, now=PLAN_CREATED_AT)


def test_validate_operation_plan_accepts_matching_optional_hash(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    with Session(engine) as session:
        workspace_id, source_file_id, _, precondition = _seed_workspace(
            session,
            workspace_root,
        )
        expected_digest = sha256(b"operation plan source").hexdigest()
        hashed_precondition = FilePrecondition(
            size_bytes=precondition.size_bytes,
            mtime_ns=precondition.mtime_ns,
            content_hash=ContentHash(digest=expected_digest),
        )
        plan = _plan(
            workspace_id,
            source_file_id,
            source_precondition=hashed_precondition,
        )

        validate_operation_plan(session, plan, now=PLAN_CREATED_AT)


def test_validate_operation_plan_rejects_changed_source_hash(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    with Session(engine) as session:
        workspace_id, source_file_id, _, precondition = _seed_workspace(
            session,
            workspace_root,
        )
        hashed_precondition = FilePrecondition(
            size_bytes=precondition.size_bytes,
            mtime_ns=precondition.mtime_ns,
            content_hash=ContentHash(digest="0" * 64),
        )
        plan = _plan(
            workspace_id,
            source_file_id,
            source_precondition=hashed_precondition,
        )

        with pytest.raises(OperationPlanSourceChangedError):
            validate_operation_plan(session, plan, now=PLAN_CREATED_AT)


def test_validate_operation_plan_rejects_missing_source_file(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    with Session(engine) as session:
        workspace_id, source_file_id, _, precondition = _seed_workspace(
            session,
            workspace_root,
        )
        plan = _plan(
            workspace_id,
            source_file_id,
            source_precondition=precondition,
        )
        (workspace_root / "inbox" / "report.pdf").unlink()

        with pytest.raises(OperationPlanSourceChangedError):
            validate_operation_plan(session, plan, now=PLAN_CREATED_AT)
