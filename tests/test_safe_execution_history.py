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
from backend.app.models import (
    ApprovalRequest,
    FileEntry,
    OperationExecution,
    OperationExecutionItem,
    Workspace,
)
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
    compensate_partial_operation_execution,
    execute_safe_operation_plan,
    recover_interrupted_operation_execution,
    retry_failed_operation_execution,
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


@pytest.mark.parametrize(
    ("mutation", "include_hash", "expected_error_code"),
    [
        (
            "size",
            False,
            SafeExecutionErrorCode.FILE_CHANGED.value,
        ),
        (
            "mtime",
            False,
            SafeExecutionErrorCode.FILE_CHANGED.value,
        ),
        (
            "hash",
            True,
            SafeExecutionErrorCode.FILE_CHANGED.value,
        ),
        (
            "deleted",
            False,
            SafeExecutionErrorCode.FILE_CHANGED.value,
        ),
        (
            "target_conflict",
            False,
            SafeFileMoveErrorCode.TARGET_CONFLICT.value,
        ),
    ],
    ids=[
        "size-changed",
        "mtime-changed",
        "hash-changed",
        "source-deleted",
        "target-conflict",
    ],
)
def test_execution_revalidates_file_and_target_before_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    include_hash: bool,
    expected_error_code: str,
) -> None:
    engine, workspace_root, request = _approved_request(
        tmp_path,
        include_hash=include_hash,
    )
    source_path = workspace_root / "inbox" / "report-7.txt"
    target_path = workspace_root / "archive" / "final-7.md"
    original_validate = safe_execution_module.validate_operation_plan
    mutation_applied = False

    def validate_then_mutate(
        session: Session,
        plan: OperationPlan,
        *,
        now: datetime | None = None,
    ) -> None:
        nonlocal mutation_applied
        original_validate(session, plan, now=now)
        if mutation_applied:
            return
        mutation_applied = True
        expected = plan.operations[0].source_precondition
        if mutation == "size":
            source_path.write_bytes(b"x" * (expected.size_bytes + 1))
            os.utime(
                source_path,
                ns=(source_path.stat().st_atime_ns, expected.mtime_ns),
            )
        elif mutation == "mtime":
            os.utime(
                source_path,
                ns=(
                    source_path.stat().st_atime_ns,
                    expected.mtime_ns + 1_000_000,
                ),
            )
        elif mutation == "hash":
            source_path.write_bytes(b"x" * expected.size_bytes)
            os.utime(
                source_path,
                ns=(source_path.stat().st_atime_ns, expected.mtime_ns),
            )
        elif mutation == "deleted":
            source_path.unlink()
        elif mutation == "target_conflict":
            target_path.write_text("competing", encoding="utf-8")
        else:
            raise AssertionError(f"unknown mutation: {mutation}")

    monkeypatch.setattr(
        safe_execution_module,
        "validate_operation_plan",
        validate_then_mutate,
    )

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

            assert mutation_applied
            assert result.status == "FAILED"
            assert result.items[0].status == "FAILED"
            assert result.items[0].error_code == expected_error_code
            assert execution.status == "FAILED"
            assert execution_item.status == "FAILED"
            assert execution_item.error_code == expected_error_code
            assert execution_item.failed_at is not None
            assert file_entry.relative_path == "inbox/report-7.txt"

        if mutation == "deleted":
            assert not source_path.exists()
        else:
            assert source_path.exists()
        if mutation == "target_conflict":
            assert target_path.read_text(encoding="utf-8") == "competing"
        else:
            assert not target_path.exists()
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


def test_multi_operation_plan_moves_all_files_and_records_each_result(
    tmp_path: Path,
) -> None:
    engine, workspace_root, request = _approved_request(
        tmp_path,
        operation_count=2,
    )

    try:
        with Session(engine) as session:
            result = execute_safe_operation_plan(session, request, now=NOW)
            execution = get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            )
            execution_items = find_operation_execution_items(
                session,
                execution.id,
            )
            file_entries = [
                get_file_entry_by_id(session, 3, source_file_id)
                for source_file_id in (7, 8)
            ]

            assert result.status == "COMPLETED"
            assert [item.status for item in result.items] == [
                "COMPLETED",
                "COMPLETED",
            ]
            assert [item.sequence_no for item in result.items] == [1, 2]
            assert execution.status == "COMPLETED"
            assert [item.status for item in execution_items] == [
                "COMPLETED",
                "COMPLETED",
            ]
            assert [entry.relative_path for entry in file_entries] == [
                "archive/final-7.md",
                "archive/final-8.md",
            ]

            with pytest.raises(
                ValueError,
                match="batch execution result",
            ):
                _ = result.before_relative_path

        assert not (workspace_root / "inbox" / "report-7.txt").exists()
        assert not (workspace_root / "inbox" / "report-8.txt").exists()
        assert (workspace_root / "archive" / "final-7.md").read_text(
            encoding="utf-8"
        ) == "approved content 7"
        assert (workspace_root / "archive" / "final-8.md").read_text(
            encoding="utf-8"
        ) == "approved content 8"
    finally:
        engine.dispose()


def test_multi_operation_plan_records_partial_success_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, workspace_root, request = _approved_request(
        tmp_path,
        operation_count=3,
    )
    original_move = safe_execution_module.SafeFileMover.move

    def fail_second_move(
        mover: object,
        source_path: Path,
        target_path: Path,
    ) -> Path:
        if source_path.name == "report-8.txt":
            raise SafeFileMoveError(
                SafeFileMoveErrorCode.TARGET_CONFLICT,
                "simulated target conflict",
            )
        return original_move(mover, source_path, target_path)

    monkeypatch.setattr(
        safe_execution_module.SafeFileMover,
        "move",
        fail_second_move,
    )

    try:
        with Session(engine) as session:
            first_result = execute_safe_operation_plan(
                session,
                request,
                now=NOW,
            )
            execution = get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            )
            execution_items = find_operation_execution_items(
                session,
                execution.id,
            )
            file_entries = [
                get_file_entry_by_id(session, 3, source_file_id)
                for source_file_id in (7, 8, 9)
            ]

            assert first_result.status == "PARTIALLY_COMPLETED"
            assert [item.status for item in first_result.items] == [
                "COMPLETED",
                "FAILED",
                "COMPLETED",
            ]
            assert [item.error_code for item in first_result.items] == [
                None,
                "safe_move_target_conflict",
                None,
            ]
            assert execution.status == "PARTIALLY_COMPLETED"
            assert execution.attempt == 1
            assert execution_items[1].failed_at is not None
            assert [entry.relative_path for entry in file_entries] == [
                "archive/final-7.md",
                "inbox/report-8.txt",
                "archive/final-9.md",
            ]

        def fail_repeated_move(*args: object, **kwargs: object) -> Path:
            raise AssertionError("重复批量请求不应再次移动文件")

        monkeypatch.setattr(
            safe_execution_module.SafeFileMover,
            "move",
            fail_repeated_move,
        )
        with Session(engine) as session:
            repeated_result = execute_safe_operation_plan(
                session,
                request,
                now=NOW,
            )
            assert repeated_result == first_result

        assert not (workspace_root / "inbox" / "report-7.txt").exists()
        assert (workspace_root / "inbox" / "report-8.txt").exists()
        assert not (workspace_root / "inbox" / "report-9.txt").exists()
        assert (workspace_root / "archive" / "final-7.md").exists()
        assert not (workspace_root / "archive" / "final-8.md").exists()
        assert (workspace_root / "archive" / "final-9.md").exists()
    finally:
        engine.dispose()


def test_multi_operation_plan_records_all_recoverable_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, workspace_root, request = _approved_request(
        tmp_path,
        operation_count=2,
    )

    def fail_move(*args: object, **kwargs: object) -> Path:
        raise SafeFileMoveError(
            SafeFileMoveErrorCode.SOURCE_UNAVAILABLE,
            "simulated unavailable source",
        )

    monkeypatch.setattr(
        safe_execution_module.SafeFileMover,
        "move",
        fail_move,
    )

    try:
        with Session(engine) as session:
            result = execute_safe_operation_plan(session, request, now=NOW)
            execution = get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            )
            execution_items = find_operation_execution_items(
                session,
                execution.id,
            )

            assert result.status == "FAILED"
            assert [item.status for item in result.items] == [
                "FAILED",
                "FAILED",
            ]
            assert all(
                item.error_code == "safe_move_source_unavailable"
                for item in execution_items
            )
            assert all(item.failed_at is not None for item in execution_items)
            assert execution.status == "FAILED"

        assert (workspace_root / "inbox" / "report-7.txt").exists()
        assert (workspace_root / "inbox" / "report-8.txt").exists()
        assert not (workspace_root / "archive" / "final-7.md").exists()
        assert not (workspace_root / "archive" / "final-8.md").exists()
    finally:
        engine.dispose()


def test_retry_failed_items_only_retries_failed_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, workspace_root, request = _approved_request(
        tmp_path,
        operation_count=3,
    )
    original_move = safe_execution_module.SafeFileMover.move

    def fail_second_move(
        mover: object,
        source_path: Path,
        target_path: Path,
    ) -> Path:
        if source_path.name == "report-8.txt":
            raise SafeFileMoveError(
                SafeFileMoveErrorCode.TARGET_CONFLICT,
                "simulated target conflict",
            )
        return original_move(mover, source_path, target_path)

    monkeypatch.setattr(
        safe_execution_module.SafeFileMover,
        "move",
        fail_second_move,
    )

    try:
        with Session(engine) as session:
            first_result = execute_safe_operation_plan(
                session,
                request,
                now=NOW,
            )
            assert first_result.status == "PARTIALLY_COMPLETED"

        retried_sources: list[str] = []

        def record_retry_move(
            mover: object,
            source_path: Path,
            target_path: Path,
        ) -> Path:
            retried_sources.append(source_path.name)
            return original_move(mover, source_path, target_path)

        monkeypatch.setattr(
            safe_execution_module.SafeFileMover,
            "move",
            record_retry_move,
        )
        with Session(engine) as session:
            result = retry_failed_operation_execution(
                session,
                WORKFLOW_ID,
                now=NOW + timedelta(minutes=1),
            )
            execution = get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            )
            execution_items = find_operation_execution_items(
                session,
                execution.id,
            )
            file_entries = [
                get_file_entry_by_id(session, 3, source_file_id)
                for source_file_id in (7, 8, 9)
            ]

            assert result.status == "COMPLETED"
            assert [item.status for item in result.items] == [
                "COMPLETED",
                "COMPLETED",
                "COMPLETED",
            ]
            assert retried_sources == ["report-8.txt"]
            assert execution.attempt == 2
            assert execution_items[1].error_code is None
            assert execution_items[1].failed_at is None
            assert [entry.relative_path for entry in file_entries] == [
                "archive/final-7.md",
                "archive/final-8.md",
                "archive/final-9.md",
            ]

        assert not (workspace_root / "inbox" / "report-7.txt").exists()
        assert not (workspace_root / "inbox" / "report-8.txt").exists()
        assert not (workspace_root / "inbox" / "report-9.txt").exists()
    finally:
        engine.dispose()


def test_compensation_restores_only_completed_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, workspace_root, request = _approved_request(
        tmp_path,
        operation_count=3,
    )
    original_move = safe_execution_module.SafeFileMover.move

    def fail_second_move(
        mover: object,
        source_path: Path,
        target_path: Path,
    ) -> Path:
        if source_path.name == "report-8.txt":
            raise SafeFileMoveError(
                SafeFileMoveErrorCode.TARGET_CONFLICT,
                "simulated target conflict",
            )
        return original_move(mover, source_path, target_path)

    monkeypatch.setattr(
        safe_execution_module.SafeFileMover,
        "move",
        fail_second_move,
    )

    try:
        with Session(engine) as session:
            first_result = execute_safe_operation_plan(
                session,
                request,
                now=NOW,
            )
            assert first_result.status == "PARTIALLY_COMPLETED"

        monkeypatch.setattr(
            safe_execution_module.SafeFileMover,
            "move",
            original_move,
        )
        with Session(engine) as session:
            result = compensate_partial_operation_execution(
                session,
                WORKFLOW_ID,
                now=NOW + timedelta(minutes=1),
            )
            execution = get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            )
            file_entries = [
                get_file_entry_by_id(session, 3, source_file_id)
                for source_file_id in (7, 8, 9)
            ]

            assert result.status == "UNDONE"
            assert [item.status for item in result.items] == [
                "UNDONE",
                "FAILED",
                "UNDONE",
            ]
            assert execution.attempt == 1
            assert [entry.relative_path for entry in file_entries] == [
                "inbox/report-7.txt",
                "inbox/report-8.txt",
                "inbox/report-9.txt",
            ]

        for source_file_id in (7, 8, 9):
            assert (
                workspace_root / "inbox" / f"report-{source_file_id}.txt"
            ).exists()
            assert not (
                workspace_root / "archive" / f"final-{source_file_id}.md"
            ).exists()
    finally:
        engine.dispose()


def test_compensation_rejects_changed_file_before_any_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, workspace_root, request = _approved_request(
        tmp_path,
        operation_count=2,
    )
    original_move = safe_execution_module.SafeFileMover.move

    def fail_second_move(
        mover: object,
        source_path: Path,
        target_path: Path,
    ) -> Path:
        if source_path.name == "report-8.txt":
            raise SafeFileMoveError(
                SafeFileMoveErrorCode.TARGET_CONFLICT,
                "simulated target conflict",
            )
        return original_move(mover, source_path, target_path)

    monkeypatch.setattr(
        safe_execution_module.SafeFileMover,
        "move",
        fail_second_move,
    )

    try:
        with Session(engine) as session:
            execute_safe_operation_plan(session, request, now=NOW)

        changed_target = workspace_root / "archive" / "final-7.md"
        changed_target.write_text("changed content", encoding="utf-8")
        monkeypatch.setattr(
            safe_execution_module.SafeFileMover,
            "move",
            original_move,
        )
        with Session(engine) as session:
            with pytest.raises(SafeExecutionError) as error_info:
                compensate_partial_operation_execution(
                    session,
                    WORKFLOW_ID,
                    now=NOW + timedelta(minutes=1),
                )
            assert error_info.value.code is SafeExecutionErrorCode.FILE_CHANGED

            execution = get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            )
            execution_items = find_operation_execution_items(
                session,
                execution.id,
            )
            assert execution.status == "PARTIALLY_COMPLETED"
            assert [item.status for item in execution_items] == [
                "COMPLETED",
                "FAILED",
            ]

        assert changed_target.read_text(encoding="utf-8") == "changed content"
        assert not (workspace_root / "inbox" / "report-7.txt").exists()
        assert (workspace_root / "inbox" / "report-8.txt").exists()
    finally:
        engine.dispose()


def test_duplicate_execution_returns_persisted_result_without_disk_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, workspace_root, request = _approved_request(tmp_path)
    source_path = workspace_root / "inbox" / "report-7.txt"
    target_path = workspace_root / "archive" / "final-7.md"

    try:
        with Session(engine) as session:
            first_result = execute_safe_operation_plan(
                session,
                request,
                now=NOW,
            )

        def fail_repeated_move(*args: object, **kwargs: object) -> Path:
            raise AssertionError("重复请求不应再次移动文件")

        monkeypatch.setattr(
            safe_execution_module.SafeFileMover,
            "move",
            fail_repeated_move,
        )

        with Session(engine) as session:
            repeated_result = execute_safe_operation_plan(
                session,
                request,
                now=NOW,
            )
            execution = get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            )
            execution_items = find_operation_execution_items(
                session,
                execution.id,
            )

            assert repeated_result == first_result
            assert execution.attempt == 1
            assert len(execution_items) == 1

        assert not source_path.exists()
        assert target_path.read_text(encoding="utf-8") == "approved content 7"
    finally:
        engine.dispose()


def test_idempotency_key_cannot_bind_to_another_workflow(
    tmp_path: Path,
) -> None:
    engine, workspace_root, request = _approved_request(tmp_path)
    conflicting_workflow_id = UUID("8933c981-fe44-4d3f-a4e0-3d7ed66be0ca")

    try:
        with Session(engine) as session:
            operation = request.plan.operations[0]
            conflicting_execution = OperationExecution(
                workflow_id=str(conflicting_workflow_id),
                plan_id=str(request.plan.plan_id),
                workspace_id=request.plan.workspace_id,
            )
            session.add(conflicting_execution)
            session.flush()
            session.add(
                OperationExecutionItem(
                    execution_id=conflicting_execution.id,
                    sequence_no=1,
                    operation_type="move",
                    source_file_id=operation.source_file_id,
                    before_location="workspace",
                    before_relative_path=operation.source_relative_path,
                    before_size_bytes=operation.source_precondition.size_bytes,
                    before_mtime_ns=operation.source_precondition.mtime_ns,
                    after_location="workspace",
                    after_relative_path=operation.target_relative_path,
                    undo_source_relative_path=operation.target_relative_path,
                    undo_target_relative_path=operation.source_relative_path,
                )
            )
            session.commit()

            with pytest.raises(SafeExecutionError) as error:
                execute_safe_operation_plan(session, request, now=NOW)

            assert error.value.code is SafeExecutionErrorCode.HISTORY_EXISTS

        assert (workspace_root / "inbox" / "report-7.txt").exists()
        assert not (workspace_root / "archive" / "final-7.md").exists()
    finally:
        engine.dispose()


def test_concurrent_history_creation_returns_winning_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, workspace_root, request = _approved_request(tmp_path)
    original_add_execution = safe_execution_module.add_operation_execution

    def create_competing_history(
        current_session: Session,
        pending_execution: OperationExecution,
    ) -> None:
        current_session.rollback()
        operation = request.plan.operations[0]
        with Session(engine) as winning_session:
            winning_execution = OperationExecution(
                workflow_id=str(request.workflow_id),
                plan_id=str(request.plan.plan_id),
                workspace_id=request.plan.workspace_id,
            )
            winning_session.add(winning_execution)
            winning_session.flush()
            winning_session.add(
                OperationExecutionItem(
                    execution_id=winning_execution.id,
                    sequence_no=1,
                    operation_type="move",
                    source_file_id=operation.source_file_id,
                    before_location="workspace",
                    before_relative_path=operation.source_relative_path,
                    before_size_bytes=operation.source_precondition.size_bytes,
                    before_mtime_ns=operation.source_precondition.mtime_ns,
                    after_location="workspace",
                    after_relative_path=operation.target_relative_path,
                    undo_source_relative_path=operation.target_relative_path,
                    undo_target_relative_path=operation.source_relative_path,
                )
            )
            winning_session.commit()

        original_add_execution(current_session, pending_execution)

    monkeypatch.setattr(
        safe_execution_module,
        "add_operation_execution",
        create_competing_history,
    )

    try:
        with Session(engine) as session:
            result = execute_safe_operation_plan(session, request, now=NOW)
            execution = get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            )

            assert result.execution_id == execution.id
            assert result.status == "EXECUTING"
            assert execution.attempt == 1
            assert len(
                find_operation_execution_items(session, execution.id)
            ) == 1

        assert (workspace_root / "inbox" / "report-7.txt").exists()
        assert not (workspace_root / "archive" / "final-7.md").exists()
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
    target_path = workspace_root / "archive" / "final-7.md"
    original_move = safe_execution_module.SafeFileMover.move

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

            repeated_result = execute_safe_operation_plan(
                session,
                request,
                now=NOW,
            )
            assert repeated_result.execution_id == execution.id
            assert repeated_result.status == "EXECUTING"
            assert execution.attempt == 1

        monkeypatch.setattr(
            safe_execution_module.SafeFileMover,
            "move",
            original_move,
        )
        with Session(engine, expire_on_commit=False) as restarted_session:
            recovered_result = recover_interrupted_operation_execution(
                restarted_session,
                WORKFLOW_ID,
                now=NOW + timedelta(minutes=1),
            )
            recovered_execution = get_operation_execution_by_workflow_id(
                restarted_session,
                str(WORKFLOW_ID),
            )
            recovered_item = find_operation_execution_items(
                restarted_session,
                recovered_execution.id,
            )[0]

            assert recovered_result.status == "COMPLETED"
            assert recovered_execution.attempt == 1
            assert recovered_item.status == "COMPLETED"

        assert not source_path.exists()
        assert target_path.read_text(encoding="utf-8") == "approved content 7"
    finally:
        engine.dispose()


def test_restart_reconciles_move_completed_before_history_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, workspace_root, request = _approved_request(tmp_path)
    source_path = workspace_root / "inbox" / "report-7.txt"
    target_path = workspace_root / "archive" / "final-7.md"
    original_index_update = (
        safe_execution_module.compare_and_set_file_entry_location
    )

    def fail_index_update(*args: object, **kwargs: object) -> bool:
        return False

    monkeypatch.setattr(
        safe_execution_module,
        "compare_and_set_file_entry_location",
        fail_index_update,
    )

    try:
        with Session(engine) as session:
            with pytest.raises(SafeExecutionError) as error_info:
                execute_safe_operation_plan(session, request, now=NOW)
            assert error_info.value.code is SafeExecutionErrorCode.STATE_CHANGED

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

        assert not source_path.exists()
        assert target_path.read_text(encoding="utf-8") == "approved content 7"

        monkeypatch.setattr(
            safe_execution_module,
            "compare_and_set_file_entry_location",
            original_index_update,
        )

        def fail_repeated_move(*args: object, **kwargs: object) -> Path:
            raise AssertionError("恢复对账不应重复移动已经到达目标的文件")

        monkeypatch.setattr(
            safe_execution_module.SafeFileMover,
            "move",
            fail_repeated_move,
        )
        with Session(engine, expire_on_commit=False) as restarted_session:
            recovered_result = recover_interrupted_operation_execution(
                restarted_session,
                WORKFLOW_ID,
                now=NOW + timedelta(minutes=1),
            )
            file_entry = get_file_entry_by_id(restarted_session, 3, 7)

            assert recovered_result.status == "COMPLETED"
            assert recovered_result.items[0].status == "COMPLETED"
            assert file_entry.relative_path == "archive/final-7.md"
    finally:
        engine.dispose()


def test_undo_failure_keeps_undoing_history_for_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, workspace_root, request = _approved_request(tmp_path)
    source_path = workspace_root / "inbox" / "report-7.txt"
    target_path = workspace_root / "archive" / "final-7.md"
    original_move = safe_execution_module.SafeFileMover.move

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

        monkeypatch.setattr(
            safe_execution_module.SafeFileMover,
            "move",
            original_move,
        )
        with Session(engine, expire_on_commit=False) as restarted_session:
            recovered_result = recover_interrupted_operation_execution(
                restarted_session,
                WORKFLOW_ID,
                now=NOW + timedelta(minutes=2),
            )
            recovered_execution = get_operation_execution_by_workflow_id(
                restarted_session,
                str(WORKFLOW_ID),
            )
            recovered_item = find_operation_execution_items(
                restarted_session,
                recovered_execution.id,
            )[0]

            assert recovered_result.status == "UNDONE"
            assert recovered_execution.status == "UNDONE"
            assert recovered_item.status == "UNDONE"

        assert not target_path.exists()
        assert source_path.read_text(encoding="utf-8") == "approved content 7"
    finally:
        engine.dispose()
