from datetime import datetime, timedelta, timezone
from hashlib import sha256
import os
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

import backend.app.safe_execution as safe_execution_module
from backend.app.database import Base
from backend.app.models import ApprovalRequest, FileEntry, Workspace
from backend.app.operation_plan import (
    ContentHash,
    FilePrecondition,
    OperationPlan,
    OperationPlanItem,
    OperationReason,
)
from backend.app.repositories import (
    find_operation_execution_items,
    get_file_entry_by_id,
    get_operation_execution_by_workflow_id,
)
from backend.app.safe_execution import (
    SafeExecutionError,
    SafeExecutionErrorCode,
    SafeExecutionRequest,
    execute_safe_operation_plan,
    undo_safe_operation_execution,
)
from backend.app.safe_file_mover import (
    SafeFileMoveError,
    SafeFileMoveErrorCode,
)


WORKFLOW_ID = UUID("66c8d4ba-a042-4491-a5d2-ad28cb47b8d9")
PLAN_ID = UUID("2d053752-d3c4-45cb-b696-bd043e78ed92")
NOW = datetime(2026, 8, 31, 10, 0, tzinfo=timezone.utc)


def _approved_request(
    tmp_path: Path,
    *,
    operation_count: int = 1,
    include_hash: bool = False,
) -> tuple[Engine, Path, SafeExecutionRequest]:
    workspace_root = tmp_path / "execution-workspace"
    target_directory = workspace_root / "archive"
    target_directory.mkdir(parents=True)
    operations: list[OperationPlanItem] = []

    engine = create_engine(
        f"sqlite:///{(tmp_path / 'safe-execution.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    with Session(engine) as session:
        workspace = Workspace(
            id=3,
            name="正式执行测试工作区",
            root_path=str(workspace_root),
        )
        session.add(workspace)

        for offset in range(operation_count):
            source_file_id = 7 + offset
            file_name = f"report-{source_file_id}.txt"
            source_path = workspace_root / "inbox" / file_name
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(
                f"approved content {source_file_id}",
                encoding="utf-8",
            )
            metadata = source_path.stat()
            relative_source = f"inbox/{file_name}"
            relative_target = f"archive/final-{source_file_id}.md"
            session.add(
                FileEntry(
                    id=source_file_id,
                    workspace_id=workspace.id,
                    relative_path=relative_source,
                    name=file_name,
                    extension=".txt",
                    size_bytes=metadata.st_size,
                    mtime_ns=metadata.st_mtime_ns,
                )
            )
            operations.append(
                OperationPlanItem(
                    source_file_id=source_file_id,
                    source_relative_path=relative_source,
                    target_relative_path=relative_target,
                    source_precondition=FilePrecondition(
                        size_bytes=metadata.st_size,
                        mtime_ns=metadata.st_mtime_ns,
                        content_hash=(
                            ContentHash(
                                digest=sha256(
                                    source_path.read_bytes()
                                ).hexdigest()
                            )
                            if include_hash
                            else None
                        ),
                    ),
                    reason=OperationReason(
                        kind="manual_selection",
                        description="由用户确认目标路径",
                    ),
                )
            )

        session.add(
            ApprovalRequest(
                workflow_id=str(WORKFLOW_ID),
                plan_id=str(PLAN_ID),
                status="APPROVED",
            )
        )
        session.commit()

    plan = OperationPlan(
        plan_id=PLAN_ID,
        workspace_id=3,
        created_at=NOW,
        operations=operations,
    )
    return (
        engine,
        workspace_root,
        SafeExecutionRequest(workflow_id=WORKFLOW_ID, plan=plan),
    )


def test_execution_moves_file_and_persists_history_and_index(
    tmp_path: Path,
) -> None:
    engine, workspace_root, request = _approved_request(
        tmp_path,
        include_hash=True,
    )
    source_path = workspace_root / "inbox" / "report-7.txt"
    target_path = workspace_root / "archive" / "final-7.md"

    try:
        with Session(engine) as session:
            result = execute_safe_operation_plan(session, request, now=NOW)

            execution = get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            )
            execution_item = find_operation_execution_items(
                session,
                execution.id,
            )[0]
            file_entry = get_file_entry_by_id(session, 3, 7)

            assert result.status == "COMPLETED"
            assert execution.status == "COMPLETED"
            assert execution_item.status == "COMPLETED"
            assert execution_item.before_relative_path == (
                "inbox/report-7.txt"
            )
            assert execution_item.after_relative_path == "archive/final-7.md"
            assert execution_item.undo_source_relative_path == (
                "archive/final-7.md"
            )
            assert execution_item.undo_target_relative_path == (
                "inbox/report-7.txt"
            )
            assert execution_item.after_size_bytes == target_path.stat().st_size
            assert execution_item.after_mtime_ns == target_path.stat().st_mtime_ns
            assert execution_item.before_sha256 == sha256(
                b"approved content 7"
            ).hexdigest()
            assert execution_item.after_sha256 == execution_item.before_sha256
            assert file_entry.relative_path == "archive/final-7.md"
            assert file_entry.name == "final-7.md"
            assert file_entry.extension == ".md"

        assert not source_path.exists()
        assert target_path.read_text(encoding="utf-8") == "approved content 7"
    finally:
        engine.dispose()


def test_undo_restores_file_index_and_history(
    tmp_path: Path,
) -> None:
    engine, workspace_root, request = _approved_request(tmp_path)
    source_path = workspace_root / "inbox" / "report-7.txt"
    target_path = workspace_root / "archive" / "final-7.md"

    try:
        with Session(engine) as session:
            execute_safe_operation_plan(session, request, now=NOW)
            result = undo_safe_operation_execution(
                session,
                WORKFLOW_ID,
                now=NOW + timedelta(minutes=1),
            )

            execution = get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            )
            execution_item = find_operation_execution_items(
                session,
                execution.id,
            )[0]
            file_entry = get_file_entry_by_id(session, 3, 7)

            assert result.status == "UNDONE"
            assert execution.status == "UNDONE"
            assert execution_item.status == "UNDONE"
            assert execution.undone_at is not None
            assert execution_item.undone_at is not None
            assert file_entry.relative_path == "inbox/report-7.txt"
            assert file_entry.name == "report-7.txt"
            assert file_entry.extension == ".txt"

        assert source_path.read_text(encoding="utf-8") == "approved content 7"
        assert not target_path.exists()
    finally:
        engine.dispose()


def test_multi_operation_plan_is_rejected_before_history_or_disk_write(
    tmp_path: Path,
) -> None:
    engine, workspace_root, request = _approved_request(
        tmp_path,
        operation_count=2,
    )

    try:
        with Session(engine) as session:
            with pytest.raises(SafeExecutionError) as error:
                execute_safe_operation_plan(session, request, now=NOW)

            assert (
                error.value.code
                is SafeExecutionErrorCode.BATCH_NOT_SUPPORTED
            )
            assert get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            ) is None

        assert (workspace_root / "inbox" / "report-7.txt").exists()
        assert (workspace_root / "inbox" / "report-8.txt").exists()
        assert not (workspace_root / "archive" / "final-7.md").exists()
        assert not (workspace_root / "archive" / "final-8.md").exists()
    finally:
        engine.dispose()


def test_duplicate_execution_is_rejected_from_persisted_history(
    tmp_path: Path,
) -> None:
    engine, _, request = _approved_request(tmp_path)

    try:
        with Session(engine) as session:
            execute_safe_operation_plan(session, request, now=NOW)

            with pytest.raises(SafeExecutionError) as error:
                execute_safe_operation_plan(session, request, now=NOW)

            assert error.value.code is SafeExecutionErrorCode.HISTORY_EXISTS
    finally:
        engine.dispose()


def test_undo_rejects_changed_file_without_changing_history(
    tmp_path: Path,
) -> None:
    engine, workspace_root, request = _approved_request(tmp_path)
    target_path = workspace_root / "archive" / "final-7.md"

    try:
        with Session(engine) as session:
            execute_safe_operation_plan(session, request, now=NOW)
            target_path.write_text("changed after execution", encoding="utf-8")

            with pytest.raises(SafeExecutionError) as error:
                undo_safe_operation_execution(
                    session,
                    WORKFLOW_ID,
                    now=NOW + timedelta(minutes=1),
                )

            execution = get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            )
            assert error.value.code is SafeExecutionErrorCode.FILE_CHANGED
            assert execution.status == "COMPLETED"

        assert target_path.read_text(encoding="utf-8") == (
            "changed after execution"
        )
    finally:
        engine.dispose()


def test_undo_detects_hash_change_when_size_and_mtime_are_unchanged(
    tmp_path: Path,
) -> None:
    engine, workspace_root, request = _approved_request(
        tmp_path,
        include_hash=True,
    )
    target_path = workspace_root / "archive" / "final-7.md"

    try:
        with Session(engine) as session:
            execute_safe_operation_plan(session, request, now=NOW)
            original_stat = target_path.stat()
            target_path.write_text("tampered content 7", encoding="utf-8")
            os.utime(
                target_path,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )

            with pytest.raises(SafeExecutionError) as error:
                undo_safe_operation_execution(
                    session,
                    WORKFLOW_ID,
                    now=NOW + timedelta(minutes=1),
                )

            assert target_path.stat().st_size == original_stat.st_size
            assert target_path.stat().st_mtime_ns == original_stat.st_mtime_ns
            assert error.value.code is SafeExecutionErrorCode.FILE_CHANGED
    finally:
        engine.dispose()


def test_undo_rejects_occupied_original_path_and_repeated_undo(
    tmp_path: Path,
) -> None:
    engine, workspace_root, request = _approved_request(tmp_path)
    source_path = workspace_root / "inbox" / "report-7.txt"

    try:
        with Session(engine) as session:
            execute_safe_operation_plan(session, request, now=NOW)
            source_path.write_text("new occupant", encoding="utf-8")

            with pytest.raises(SafeExecutionError) as conflict:
                undo_safe_operation_execution(
                    session,
                    WORKFLOW_ID,
                    now=NOW + timedelta(minutes=1),
                )
            assert (
                conflict.value.code
                is SafeExecutionErrorCode.UNDO_TARGET_CONFLICT
            )
            source_path.unlink()

            undo_safe_operation_execution(
                session,
                WORKFLOW_ID,
                now=NOW + timedelta(minutes=2),
            )
            with pytest.raises(SafeExecutionError) as repeated:
                undo_safe_operation_execution(
                    session,
                    WORKFLOW_ID,
                    now=NOW + timedelta(minutes=3),
                )
            assert (
                repeated.value.code
                is SafeExecutionErrorCode.INVALID_HISTORY_STATE
            )
    finally:
        engine.dispose()


def test_execution_failure_keeps_executing_history_for_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, workspace_root, request = _approved_request(tmp_path)
    source_path = workspace_root / "inbox" / "report-7.txt"

    def fail_move(*args: object, **kwargs: object) -> Path:
        raise SafeFileMoveError(
            SafeFileMoveErrorCode.MOVE_FAILED,
            "simulated move failure",
        )

    monkeypatch.setattr(
        safe_execution_module.SafeFileMover,
        "move",
        fail_move,
    )

    try:
        with Session(engine) as session:
            with pytest.raises(SafeFileMoveError):
                execute_safe_operation_plan(session, request, now=NOW)

            execution = get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            )
            execution_item = find_operation_execution_items(
                session,
                execution.id,
            )[0]
            assert execution.status == "EXECUTING"
            assert execution_item.status == "PENDING"

        assert source_path.read_text(encoding="utf-8") == "approved content 7"
    finally:
        engine.dispose()


def test_undo_failure_keeps_undoing_history_for_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, workspace_root, request = _approved_request(tmp_path)
    target_path = workspace_root / "archive" / "final-7.md"

    try:
        with Session(engine) as session:
            execute_safe_operation_plan(session, request, now=NOW)

            def fail_move(*args: object, **kwargs: object) -> Path:
                raise SafeFileMoveError(
                    SafeFileMoveErrorCode.MOVE_FAILED,
                    "simulated undo failure",
                )

            monkeypatch.setattr(
                safe_execution_module.SafeFileMover,
                "move",
                fail_move,
            )
            with pytest.raises(SafeFileMoveError):
                undo_safe_operation_execution(
                    session,
                    WORKFLOW_ID,
                    now=NOW + timedelta(minutes=1),
                )

            execution = get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            )
            execution_item = find_operation_execution_items(
                session,
                execution.id,
            )[0]
            assert execution.status == "UNDOING"
            assert execution_item.status == "UNDOING"

        assert target_path.read_text(encoding="utf-8") == "approved content 7"
    finally:
        engine.dispose()
