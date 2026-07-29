from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import (
    FileEntry,
    OperationExecution,
    OperationExecutionItem,
    Workspace,
)
from backend.app.repositories import (
    add_operation_execution,
    add_operation_execution_item,
    compare_and_set_file_entry_location,
    compare_and_set_operation_execution_item_status,
    compare_and_set_operation_execution_status,
    find_operation_execution_items,
    get_operation_execution_by_id,
    get_operation_execution_by_plan_id,
    get_operation_execution_by_workflow_id,
)


WORKFLOW_ID = "66c8d4ba-a042-4491-a5d2-ad28cb47b8d9"
PLAN_ID = "2d053752-d3c4-45cb-b696-bd043e78ed92"
COMPLETED_AT = datetime(2026, 8, 31, 9, 0, tzinfo=timezone.utc)
FAILED_AT = datetime(2026, 8, 31, 9, 2, tzinfo=timezone.utc)
UNDONE_AT = datetime(2026, 8, 31, 9, 5, tzinfo=timezone.utc)


def _workspace(session: Session, root_path: str) -> Workspace:
    workspace = Workspace(name="执行历史测试", root_path=root_path)
    session.add(workspace)
    session.commit()
    return workspace


def _execution(workspace_id: int) -> OperationExecution:
    return OperationExecution(
        workflow_id=WORKFLOW_ID,
        plan_id=PLAN_ID,
        workspace_id=workspace_id,
    )


def _execution_item(
    execution_id: int,
    sequence_no: int,
) -> OperationExecutionItem:
    return OperationExecutionItem(
        execution_id=execution_id,
        sequence_no=sequence_no,
        operation_type="move",
        source_file_id=sequence_no,
        before_location="workspace",
        before_relative_path=f"inbox/report-{sequence_no}.txt",
        before_size_bytes=10,
        before_mtime_ns=100,
        after_location="workspace",
        after_relative_path=f"archive/report-{sequence_no}.txt",
        undo_source_relative_path=f"archive/report-{sequence_no}.txt",
        undo_target_relative_path=f"inbox/report-{sequence_no}.txt",
    )


def test_repository_persists_and_reads_execution_history_in_plan_order(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'execution-history.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            workspace = _workspace(session, str(tmp_path / "workspace"))
            execution = _execution(workspace.id)
            add_operation_execution(session, execution)
            session.flush()
            add_operation_execution_item(
                session,
                _execution_item(execution.id, 2),
            )
            add_operation_execution_item(
                session,
                _execution_item(execution.id, 1),
            )
            session.commit()

            assert get_operation_execution_by_id(
                session,
                execution.id,
            ) is execution
            assert get_operation_execution_by_workflow_id(
                session,
                WORKFLOW_ID,
            ).id == execution.id
            assert get_operation_execution_by_plan_id(
                session,
                PLAN_ID,
            ).id == execution.id
            assert [
                item.sequence_no
                for item in find_operation_execution_items(
                    session,
                    execution.id,
                )
            ] == [1, 2]
    finally:
        engine.dispose()


def test_repository_leaves_history_transaction_to_caller(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'execution-rollback.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            workspace = _workspace(session, str(tmp_path / "workspace"))
            execution = _execution(workspace.id)
            add_operation_execution(session, execution)
            session.flush()
            add_operation_execution_item(
                session,
                _execution_item(execution.id, 1),
            )

            session.rollback()

            assert get_operation_execution_by_workflow_id(
                session,
                WORKFLOW_ID,
            ) is None
    finally:
        engine.dispose()


def test_status_updates_are_atomic_and_preserve_recorded_evidence(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'execution-status.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            workspace = _workspace(session, str(tmp_path / "workspace"))
            execution = _execution(workspace.id)
            add_operation_execution(session, execution)
            session.flush()
            execution_item = _execution_item(execution.id, 1)
            add_operation_execution_item(session, execution_item)
            session.commit()

            assert compare_and_set_operation_execution_status(
                session,
                execution.id,
                "EXECUTING",
                next_status="COMPLETED",
                completed_at=COMPLETED_AT,
            )
            assert compare_and_set_operation_execution_item_status(
                session,
                execution_item.id,
                "PENDING",
                next_status="COMPLETED",
                after_size_bytes=10,
                after_mtime_ns=101,
                after_sha256="a" * 64,
                completed_at=COMPLETED_AT,
            )
            assert not compare_and_set_operation_execution_status(
                session,
                execution.id,
                "EXECUTING",
                next_status="FAILED",
            )
            assert not compare_and_set_operation_execution_item_status(
                session,
                execution_item.id,
                "PENDING",
                next_status="FAILED",
                error_code="safe_move_source_unavailable",
                failed_at=FAILED_AT,
            )

            assert compare_and_set_operation_execution_status(
                session,
                execution.id,
                "COMPLETED",
                next_status="UNDOING",
            )
            assert compare_and_set_operation_execution_item_status(
                session,
                execution_item.id,
                "COMPLETED",
                next_status="UNDOING",
            )
            assert compare_and_set_operation_execution_item_status(
                session,
                execution_item.id,
                "UNDOING",
                next_status="UNDONE",
                undone_at=UNDONE_AT,
            )
            assert compare_and_set_operation_execution_status(
                session,
                execution.id,
                "UNDOING",
                next_status="UNDONE",
                undone_at=UNDONE_AT,
            )
            session.commit()
            session.expire_all()

            restored_execution = get_operation_execution_by_id(
                session,
                execution.id,
            )
            restored_item = find_operation_execution_items(
                session,
                execution.id,
            )[0]

            assert restored_execution.status == "UNDONE"
            assert restored_execution.completed_at == COMPLETED_AT.replace(
                tzinfo=None
            )
            assert restored_execution.undone_at == UNDONE_AT.replace(
                tzinfo=None
            )
            assert restored_item.status == "UNDONE"
            assert restored_item.before_relative_path == "inbox/report-1.txt"
            assert restored_item.after_size_bytes == 10
            assert restored_item.after_mtime_ns == 101
            assert restored_item.after_sha256 == "a" * 64
            assert restored_item.completed_at == COMPLETED_AT.replace(
                tzinfo=None
            )
            assert restored_item.undone_at == UNDONE_AT.replace(tzinfo=None)
    finally:
        engine.dispose()


def test_item_failure_evidence_is_atomic_and_cannot_be_overwritten(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'execution-item-failure.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            workspace = _workspace(session, str(tmp_path / "workspace"))
            execution = _execution(workspace.id)
            add_operation_execution(session, execution)
            session.flush()
            execution_item = _execution_item(execution.id, 1)
            add_operation_execution_item(session, execution_item)
            session.commit()

            with pytest.raises(
                ValueError,
                match="requires error_code and failed_at",
            ):
                compare_and_set_operation_execution_item_status(
                    session,
                    execution_item.id,
                    "PENDING",
                    next_status="FAILED",
                )

            assert compare_and_set_operation_execution_item_status(
                session,
                execution_item.id,
                "PENDING",
                next_status="FAILED",
                error_code="safe_move_target_conflict",
                failed_at=FAILED_AT,
            )
            assert not compare_and_set_operation_execution_item_status(
                session,
                execution_item.id,
                "PENDING",
                next_status="COMPLETED",
                after_size_bytes=10,
                after_mtime_ns=101,
                completed_at=COMPLETED_AT,
            )

            with pytest.raises(
                ValueError,
                match="FAILED -> COMPLETED",
            ):
                compare_and_set_operation_execution_item_status(
                    session,
                    execution_item.id,
                    "FAILED",
                    next_status="COMPLETED",
                    after_size_bytes=10,
                    after_mtime_ns=101,
                    completed_at=COMPLETED_AT,
                )

            session.commit()
            session.expire_all()

            restored_item = find_operation_execution_items(
                session,
                execution.id,
            )[0]
            assert restored_item.status == "FAILED"
            assert restored_item.error_code == "safe_move_target_conflict"
            assert restored_item.failed_at == FAILED_AT.replace(tzinfo=None)
            assert restored_item.after_size_bytes is None
            assert restored_item.after_mtime_ns is None
            assert restored_item.completed_at is None
    finally:
        engine.dispose()


def test_execution_status_graph_rejects_illegal_transitions(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'execution-state-graph.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            workspace = _workspace(session, str(tmp_path / "workspace"))
            execution = _execution(workspace.id)
            add_operation_execution(session, execution)
            session.commit()

            with pytest.raises(
                ValueError,
                match="EXECUTING -> UNDONE",
            ):
                compare_and_set_operation_execution_status(
                    session,
                    execution.id,
                    "EXECUTING",
                    next_status="UNDONE",
                )

            assert compare_and_set_operation_execution_status(
                session,
                execution.id,
                "EXECUTING",
                next_status="PARTIALLY_COMPLETED",
            )
            assert compare_and_set_operation_execution_status(
                session,
                execution.id,
                "PARTIALLY_COMPLETED",
                next_status="EXECUTING",
            )
            assert compare_and_set_operation_execution_status(
                session,
                execution.id,
                "EXECUTING",
                next_status="FAILED",
            )
            assert compare_and_set_operation_execution_status(
                session,
                execution.id,
                "FAILED",
                next_status="EXECUTING",
            )
            assert compare_and_set_operation_execution_status(
                session,
                execution.id,
                "EXECUTING",
                next_status="COMPLETED",
                completed_at=COMPLETED_AT,
            )
            assert compare_and_set_operation_execution_status(
                session,
                execution.id,
                "COMPLETED",
                next_status="UNDOING",
            )
            assert compare_and_set_operation_execution_status(
                session,
                execution.id,
                "UNDOING",
                next_status="UNDONE",
                undone_at=UNDONE_AT,
            )

            with pytest.raises(
                ValueError,
                match="UNDONE -> EXECUTING",
            ):
                compare_and_set_operation_execution_status(
                    session,
                    execution.id,
                    "UNDONE",
                    next_status="EXECUTING",
                )

            session.commit()
            session.expire_all()

            restored = get_operation_execution_by_id(session, execution.id)
            assert restored.status == "UNDONE"
            assert restored.attempt == 3
    finally:
        engine.dispose()


def test_retry_attempt_increments_once_when_expected_state_is_stale(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'execution-retry-cas.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as setup_session:
            workspace = _workspace(
                setup_session,
                str(tmp_path / "workspace"),
            )
            execution = _execution(workspace.id)
            execution.status = "FAILED"
            add_operation_execution(setup_session, execution)
            setup_session.commit()
            execution_id = execution.id

        with Session(engine) as first_retry:
            assert compare_and_set_operation_execution_status(
                first_retry,
                execution_id,
                "FAILED",
                next_status="EXECUTING",
            )
            first_retry.commit()

        with Session(engine) as stale_retry:
            assert not compare_and_set_operation_execution_status(
                stale_retry,
                execution_id,
                "FAILED",
                next_status="EXECUTING",
            )
            stale_retry.commit()

        with Session(engine) as verification_session:
            restored = get_operation_execution_by_id(
                verification_session,
                execution_id,
            )
            assert restored.status == "EXECUTING"
            assert restored.attempt == 2
    finally:
        engine.dispose()


def test_file_entry_location_update_is_guarded_and_caller_controlled(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'file-entry-location.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            workspace = _workspace(session, str(tmp_path / "workspace"))
            file_entry = FileEntry(
                workspace_id=workspace.id,
                relative_path="inbox/report.txt",
                name="report.txt",
                extension=".txt",
                size_bytes=10,
                mtime_ns=100,
            )
            session.add(file_entry)
            session.commit()

            assert not compare_and_set_file_entry_location(
                session,
                workspace.id,
                file_entry.id,
                "inbox/stale.txt",
                next_relative_path="archive/report.md",
                size_bytes=11,
                mtime_ns=101,
            )
            assert compare_and_set_file_entry_location(
                session,
                workspace.id,
                file_entry.id,
                "inbox/report.txt",
                next_relative_path="archive/report.md",
                size_bytes=11,
                mtime_ns=101,
            )
            session.rollback()
            session.expire_all()

            restored = session.get(FileEntry, file_entry.id)
            assert restored.relative_path == "inbox/report.txt"

            assert compare_and_set_file_entry_location(
                session,
                workspace.id,
                file_entry.id,
                "inbox/report.txt",
                next_relative_path="archive/report.md",
                size_bytes=11,
                mtime_ns=101,
            )
            session.commit()
            session.expire_all()

            updated = session.get(FileEntry, file_entry.id)
            assert updated.relative_path == "archive/report.md"
            assert updated.name == "report.md"
            assert updated.extension == ".md"
            assert updated.size_bytes == 11
            assert updated.mtime_ns == 101

            with pytest.raises(ValueError):
                compare_and_set_file_entry_location(
                    session,
                    workspace.id,
                    file_entry.id,
                    "archive/report.md",
                    next_relative_path="../outside.txt",
                    size_bytes=11,
                    mtime_ns=101,
                )
    finally:
        engine.dispose()
