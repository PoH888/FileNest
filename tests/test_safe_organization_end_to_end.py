from dataclasses import dataclass
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
from backend.app.operation_preview import OperationPreviewRequest
from backend.app.organization_planning import build_operation_plan_record
from backend.app.repositories import (
    find_operation_execution_items,
    get_file_entry_by_id,
    get_operation_execution_by_workflow_id,
)
from backend.app.safe_execution import (
    SafeExecutionRequest,
    compensate_partial_operation_execution,
    execute_safe_operation_plan,
    undo_safe_operation_execution,
)
from backend.app.services import (
    OperationPlanApprovalError,
    OperationPlanSourceChangedError,
    approve_operation_plan,
    edit_operation_plan,
    generate_operation_preview,
    search_files,
)


WORKFLOW_ID = UUID("f11af391-4c13-412e-9373-fb0a44279da3")
PLAN_ID = UUID("4f3d5ee8-cd0a-49e8-8f8e-99303839704e")
BATCH_PLAN_ID = UUID("2a8c8685-e154-4463-9a69-f70eb6d0aadd")


@dataclass(frozen=True, slots=True)
class _WaitingScenario:
    engine: Engine
    database_path: Path
    workspace_root: Path
    source_path: Path
    target_path: Path
    workspace_id: int
    source_file_id: int
    request: SafeExecutionRequest
    plan_created_at: datetime


def _build_waiting_scenario(tmp_path: Path) -> _WaitingScenario:
    workspace_root = tmp_path / "e26-safety-workspace"
    source_path = workspace_root / "inbox" / "Python_Project_Code.zip"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"FileNest E26 safety boundary")
    (workspace_root / "programming" / "Python").mkdir(parents=True)
    (workspace_root / "programming" / "Java").mkdir(parents=True)

    database_path = tmp_path / "e26-safety.db"
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    Base.metadata.create_all(bind=engine)
    plan_created_at = datetime.now(timezone.utc)

    with Session(engine) as session:
        workspace = Workspace(
            name="E26 安全边界工作区",
            root_path=str(workspace_root),
        )
        session.add(workspace)
        session.flush()

        source_metadata = source_path.stat()
        source_entry = FileEntry(
            workspace_id=workspace.id,
            relative_path="inbox/Python_Project_Code.zip",
            name="Python_Project_Code.zip",
            extension=".zip",
            size_bytes=source_metadata.st_size,
            mtime_ns=source_metadata.st_mtime_ns,
        )
        session.add(source_entry)
        session.commit()

        query_result = search_files(
            session,
            workspace.id,
            keyword="Python",
            extension="zip",
        )
        assert query_result.total == 1

        preview = generate_operation_preview(
            session,
            OperationPreviewRequest(
                workspace_id=workspace.id,
                source_file_ids=(query_result.items[0].id,),
                target_directories=(
                    "programming/Python",
                    "programming/Java",
                ),
            ),
        )
        selected_candidate = preview.items[0].candidates[0]
        assert selected_candidate.relative_directory == "programming/Python"

        target_relative_path = (
            f"{selected_candidate.relative_directory}/{source_entry.name}"
        )
        plan = OperationPlan(
            plan_id=PLAN_ID,
            workspace_id=workspace.id,
            created_at=plan_created_at,
            operations=(
                OperationPlanItem(
                    source_file_id=source_entry.id,
                    source_relative_path=source_entry.relative_path,
                    target_relative_path=target_relative_path,
                    source_precondition=FilePrecondition(
                        size_bytes=source_metadata.st_size,
                        mtime_ns=source_metadata.st_mtime_ns,
                        content_hash=ContentHash(
                            digest=sha256(source_path.read_bytes()).hexdigest()
                        ),
                    ),
                    reason=OperationReason(
                        kind="matched_candidate",
                        description="采用只读预览的最高分候选目录",
                        match_score=selected_candidate.score,
                    ),
                ),
            ),
        )
        session.add(
            build_operation_plan_record(
                plan,
                workflow_id=WORKFLOW_ID,
            )
        )
        session.add(
            ApprovalRequest(
                workflow_id=str(WORKFLOW_ID),
                plan_id=str(plan.plan_id),
            )
        )
        session.commit()

        workspace_id = workspace.id
        source_file_id = source_entry.id

    return _WaitingScenario(
        engine=engine,
        database_path=database_path,
        workspace_root=workspace_root,
        source_path=source_path,
        target_path=workspace_root / target_relative_path,
        workspace_id=workspace_id,
        source_file_id=source_file_id,
        request=SafeExecutionRequest(
            workflow_id=WORKFLOW_ID,
            plan=plan,
        ),
        plan_created_at=plan_created_at,
    )


def test_query_plan_approve_execute_and_undo_real_file_chain(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "safe-organization-workspace"
    source_path = workspace_root / "inbox" / "Python_Project_Code.zip"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"FileNest E26 safe organization")
    (workspace_root / "programming" / "Python").mkdir(parents=True)
    (workspace_root / "programming" / "Java").mkdir(parents=True)

    engine = create_engine(
        f"sqlite:///{(tmp_path / 'safe-organization-e2e.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)
    plan_created_at = datetime.now(timezone.utc)

    try:
        with Session(engine) as session:
            workspace = Workspace(
                name="安全整理综合评测工作区",
                root_path=str(workspace_root),
            )
            session.add(workspace)
            session.flush()

            source_metadata = source_path.stat()
            source_entry = FileEntry(
                workspace_id=workspace.id,
                relative_path="inbox/Python_Project_Code.zip",
                name="Python_Project_Code.zip",
                extension=".zip",
                size_bytes=source_metadata.st_size,
                mtime_ns=source_metadata.st_mtime_ns,
            )
            session.add(source_entry)
            session.commit()

            query_result = search_files(
                session,
                workspace.id,
                keyword="Python",
                extension="zip",
            )
            assert query_result.total == 1
            assert query_result.items == [source_entry]

            preview = generate_operation_preview(
                session,
                OperationPreviewRequest(
                    workspace_id=workspace.id,
                    source_file_ids=(query_result.items[0].id,),
                    target_directories=(
                        "programming/Python",
                        "programming/Java",
                    ),
                ),
            )
            selected_candidate = preview.items[0].candidates[0]
            assert selected_candidate.relative_directory == "programming/Python"

            target_relative_path = (
                f"{selected_candidate.relative_directory}/{source_entry.name}"
            )
            plan = OperationPlan(
                plan_id=PLAN_ID,
                workspace_id=workspace.id,
                created_at=plan_created_at,
                operations=(
                    OperationPlanItem(
                        source_file_id=source_entry.id,
                        source_relative_path=source_entry.relative_path,
                        target_relative_path=target_relative_path,
                        source_precondition=FilePrecondition(
                            size_bytes=source_metadata.st_size,
                            mtime_ns=source_metadata.st_mtime_ns,
                            content_hash=ContentHash(
                                digest=sha256(source_path.read_bytes()).hexdigest()
                            ),
                        ),
                        reason=OperationReason(
                            kind="matched_candidate",
                            description="采用只读预览的最高分候选目录",
                            match_score=selected_candidate.score,
                        ),
                    ),
                ),
            )
            session.add(
                build_operation_plan_record(
                    plan,
                    workflow_id=WORKFLOW_ID,
                )
            )
            approval = ApprovalRequest(
                workflow_id=str(WORKFLOW_ID),
                plan_id=str(plan.plan_id),
            )
            session.add(approval)
            session.commit()

            approved = approve_operation_plan(
                session,
                WORKFLOW_ID,
                plan.plan_id,
            )
            assert approved.status == "APPROVED"

            request = SafeExecutionRequest(
                workflow_id=WORKFLOW_ID,
                plan=plan,
            )
            executed = execute_safe_operation_plan(
                session,
                request,
                now=plan_created_at,
            )
            target_path = workspace_root / target_relative_path
            moved_entry = get_file_entry_by_id(
                session,
                workspace.id,
                source_entry.id,
            )

            assert executed.status == "COMPLETED"
            assert not source_path.exists()
            assert target_path.read_bytes() == b"FileNest E26 safe organization"
            assert moved_entry is not None
            assert moved_entry.relative_path == target_relative_path

            undone = undo_safe_operation_execution(
                session,
                WORKFLOW_ID,
                now=plan_created_at + timedelta(minutes=1),
            )
            restored_entry = get_file_entry_by_id(
                session,
                workspace.id,
                source_entry.id,
            )
            execution = get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            )

            assert undone.status == "UNDONE"
            assert source_path.read_bytes() == b"FileNest E26 safe organization"
            assert not target_path.exists()
            assert restored_entry is not None
            assert restored_entry.relative_path == "inbox/Python_Project_Code.zip"
            assert execution is not None
            assert execution.status == "UNDONE"
    finally:
        engine.dispose()


def test_approved_cross_workspace_plan_cannot_execute(
    tmp_path: Path,
) -> None:
    scenario = _build_waiting_scenario(tmp_path)
    other_workspace_root = tmp_path / "other-workspace"
    other_source_path = other_workspace_root / "private" / "secret.zip"
    other_source_path.parent.mkdir(parents=True)
    other_source_path.write_bytes(b"other workspace private content")

    try:
        with Session(scenario.engine) as session:
            other_workspace = Workspace(
                name="其他授权工作区",
                root_path=str(other_workspace_root),
            )
            session.add(other_workspace)
            session.flush()

            other_metadata = other_source_path.stat()
            other_entry = FileEntry(
                workspace_id=other_workspace.id,
                relative_path="private/secret.zip",
                name="secret.zip",
                extension=".zip",
                size_bytes=other_metadata.st_size,
                mtime_ns=other_metadata.st_mtime_ns,
            )
            session.add(other_entry)
            session.commit()

            cross_workspace_plan = OperationPlan(
                plan_id=scenario.request.plan.plan_id,
                workspace_id=scenario.workspace_id,
                created_at=scenario.plan_created_at,
                operations=(
                    OperationPlanItem(
                        source_file_id=other_entry.id,
                        source_relative_path=other_entry.relative_path,
                        target_relative_path=(
                            scenario.request.plan.operations[0].target_relative_path
                        ),
                        source_precondition=FilePrecondition(
                            size_bytes=other_metadata.st_size,
                            mtime_ns=other_metadata.st_mtime_ns,
                            content_hash=ContentHash(
                                digest=sha256(
                                    other_source_path.read_bytes()
                                ).hexdigest()
                            ),
                        ),
                        reason=OperationReason(
                            kind="manual_selection",
                            description="模拟跨工作区文件被塞入当前计划",
                        ),
                    ),
                ),
            )
            approve_operation_plan(
                session,
                WORKFLOW_ID,
                cross_workspace_plan.plan_id,
            )

            with pytest.raises(OperationPlanApprovalError):
                execute_safe_operation_plan(
                    session,
                    SafeExecutionRequest(
                        workflow_id=WORKFLOW_ID,
                        plan=cross_workspace_plan,
                    ),
                    now=scenario.plan_created_at,
                )

            execution = get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            )

            assert execution is None

        assert scenario.source_path.read_bytes() == (
            b"FileNest E26 safety boundary"
        )
        assert not scenario.target_path.exists()
        assert other_source_path.read_bytes() == b"other workspace private content"
    finally:
        scenario.engine.dispose()


def test_file_changed_after_approval_is_rejected_before_execution(
    tmp_path: Path,
) -> None:
    scenario = _build_waiting_scenario(tmp_path)

    try:
        with Session(scenario.engine) as session:
            approve_operation_plan(
                session,
                WORKFLOW_ID,
                scenario.request.plan.plan_id,
            )

        original_stat = scenario.source_path.stat()
        scenario.source_path.write_bytes(b"X" * original_stat.st_size)
        os.utime(
            scenario.source_path,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        changed_stat = scenario.source_path.stat()
        assert changed_stat.st_size == original_stat.st_size
        assert changed_stat.st_mtime_ns == original_stat.st_mtime_ns

        with Session(scenario.engine) as session:
            with pytest.raises(OperationPlanSourceChangedError):
                execute_safe_operation_plan(
                    session,
                    scenario.request,
                    now=scenario.plan_created_at,
                )

            execution = get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            )

            assert execution is None

        assert scenario.source_path.read_bytes() == b"X" * original_stat.st_size
        assert not scenario.target_path.exists()
    finally:
        scenario.engine.dispose()


def test_restart_replays_duplicate_without_move_and_can_undo(
    tmp_path: Path,
) -> None:
    scenario = _build_waiting_scenario(tmp_path)
    restarted_engine: Engine | None = None

    try:
        with Session(scenario.engine) as session:
            approve_operation_plan(
                session,
                WORKFLOW_ID,
                scenario.request.plan.plan_id,
            )
            first_result = execute_safe_operation_plan(
                session,
                scenario.request,
                now=scenario.plan_created_at,
            )

        target_stat = scenario.target_path.stat()
        assert not scenario.source_path.exists()

        scenario.engine.dispose()
        restarted_engine = create_engine(
            f"sqlite:///{scenario.database_path.as_posix()}"
        )
        with Session(restarted_engine) as restarted_session:
            repeated_result = execute_safe_operation_plan(
                restarted_session,
                scenario.request,
                now=scenario.plan_created_at + timedelta(minutes=1),
            )
            execution = get_operation_execution_by_workflow_id(
                restarted_session,
                str(WORKFLOW_ID),
            )
            execution_items = find_operation_execution_items(
                restarted_session,
                execution.id,
            )

            assert repeated_result == first_result
            assert execution.attempt == 1
            assert len(execution_items) == 1
            assert scenario.target_path.stat().st_size == target_stat.st_size
            assert (
                scenario.target_path.stat().st_mtime_ns
                == target_stat.st_mtime_ns
            )

            undone = undo_safe_operation_execution(
                restarted_session,
                WORKFLOW_ID,
                now=scenario.plan_created_at + timedelta(minutes=2),
            )

            assert undone.status == "UNDONE"

        assert scenario.source_path.read_bytes() == (
            b"FileNest E26 safety boundary"
        )
        assert not scenario.target_path.exists()
    finally:
        scenario.engine.dispose()
        if restarted_engine is not None:
            restarted_engine.dispose()


def test_batch_partial_failure_compensates_only_completed_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _build_waiting_scenario(tmp_path)
    second_source_path = (
        scenario.workspace_root / "inbox" / "Python_Project_Notes.zip"
    )
    second_source_path.write_bytes(b"FileNest E26 second batch source")
    conflict_content = b"external target must be preserved"
    original_move = safe_execution_module.SafeFileMover.move

    try:
        with Session(scenario.engine) as session:
            second_metadata = second_source_path.stat()
            second_entry = FileEntry(
                workspace_id=scenario.workspace_id,
                relative_path="inbox/Python_Project_Notes.zip",
                name="Python_Project_Notes.zip",
                extension=".zip",
                size_bytes=second_metadata.st_size,
                mtime_ns=second_metadata.st_mtime_ns,
            )
            session.add(second_entry)
            session.commit()

            query_result = search_files(
                session,
                scenario.workspace_id,
                keyword="Python",
                extension="zip",
            )
            assert query_result.total == 2

            preview = generate_operation_preview(
                session,
                OperationPreviewRequest(
                    workspace_id=scenario.workspace_id,
                    source_file_ids=tuple(
                        entry.id for entry in query_result.items
                    ),
                    target_directories=(
                        "programming/Python",
                        "programming/Java",
                    ),
                ),
            )
            entries_by_id = {
                entry.id: entry for entry in query_result.items
            }
            operations: list[OperationPlanItem] = []
            target_paths: dict[int, Path] = {}
            for preview_item in preview.items:
                entry = entries_by_id[preview_item.source_file_id]
                source_path = scenario.workspace_root / entry.relative_path
                source_metadata = source_path.stat()
                candidate = preview_item.candidates[0]
                target_relative_path = (
                    f"{candidate.relative_directory}/{entry.name}"
                )
                target_paths[entry.id] = (
                    scenario.workspace_root / target_relative_path
                )
                operations.append(
                    OperationPlanItem(
                        source_file_id=entry.id,
                        source_relative_path=entry.relative_path,
                        target_relative_path=target_relative_path,
                        source_precondition=FilePrecondition(
                            size_bytes=source_metadata.st_size,
                            mtime_ns=source_metadata.st_mtime_ns,
                            content_hash=ContentHash(
                                digest=sha256(source_path.read_bytes()).hexdigest()
                            ),
                        ),
                        reason=OperationReason(
                            kind="matched_candidate",
                            description="采用批量预览的最高分候选目录",
                            match_score=candidate.score,
                        ),
                    )
                )

            batch_plan = OperationPlan(
                plan_id=BATCH_PLAN_ID,
                workspace_id=scenario.workspace_id,
                created_at=scenario.plan_created_at,
                operations=tuple(operations),
            )
            session.add(
                build_operation_plan_record(
                    batch_plan,
                    workflow_id=WORKFLOW_ID,
                    parent_plan_id=PLAN_ID,
                )
            )
            session.flush()
            edited = edit_operation_plan(
                session,
                WORKFLOW_ID,
                PLAN_ID,
                batch_plan.plan_id,
            )
            assert edited.status == "WAITING_APPROVAL"
            approved = approve_operation_plan(
                session,
                WORKFLOW_ID,
                batch_plan.plan_id,
            )
            assert approved.status == "APPROVED"

            def create_conflict_before_second_move(
                mover: object,
                source_path: Path,
                target_path: Path,
            ) -> Path:
                if source_path.name == second_entry.name:
                    (scenario.workspace_root / target_path).write_bytes(
                        conflict_content
                    )
                return original_move(mover, source_path, target_path)

            monkeypatch.setattr(
                safe_execution_module.SafeFileMover,
                "move",
                create_conflict_before_second_move,
            )
            result = execute_safe_operation_plan(
                session,
                SafeExecutionRequest(
                    workflow_id=WORKFLOW_ID,
                    plan=batch_plan,
                ),
                now=scenario.plan_created_at,
            )

            assert result.status == "PARTIALLY_COMPLETED"
            assert [item.status for item in result.items] == [
                "COMPLETED",
                "FAILED",
            ]
            assert result.items[1].error_code == "safe_move_target_conflict"

            monkeypatch.setattr(
                safe_execution_module.SafeFileMover,
                "move",
                original_move,
            )
            compensated = compensate_partial_operation_execution(
                session,
                WORKFLOW_ID,
                now=scenario.plan_created_at + timedelta(minutes=1),
            )
            execution = get_operation_execution_by_workflow_id(
                session,
                str(WORKFLOW_ID),
            )
            execution_items = find_operation_execution_items(
                session,
                execution.id,
            )
            restored_entries = [
                get_file_entry_by_id(
                    session,
                    scenario.workspace_id,
                    operation.source_file_id,
                )
                for operation in batch_plan.operations
            ]

            assert compensated.status == "UNDONE"
            assert [item.status for item in execution_items] == [
                "UNDONE",
                "FAILED",
            ]
            assert [entry.relative_path for entry in restored_entries] == [
                operation.source_relative_path
                for operation in batch_plan.operations
            ]

        first_target_path = target_paths[scenario.source_file_id]
        second_target_path = target_paths[second_entry.id]
        assert scenario.source_path.read_bytes() == (
            b"FileNest E26 safety boundary"
        )
        assert not first_target_path.exists()
        assert second_source_path.read_bytes() == (
            b"FileNest E26 second batch source"
        )
        assert second_target_path.read_bytes() == conflict_content
    finally:
        scenario.engine.dispose()
