from collections.abc import Iterator
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

import backend.app.safe_execution as safe_execution_module
from backend.app.database import Base
from backend.app.models import (
    ApprovalRequest,
    FileEntry,
    Workspace,
)
from backend.app.operation_plan import (
    ContentHash,
    FilePrecondition,
    OperationPlan,
    OperationPlanItem,
    OperationReason,
)
from backend.app.organization_decisions import apply_organization_decision
from backend.app.organization_planning import build_operation_plan_record
from backend.app.proposal_tools import build_propose_quarantine_tool
from backend.app.repositories import (
    find_operation_execution_items,
    get_operation_execution_by_workflow_id,
)
from backend.app.safe_execution import (
    SafeExecutionError,
    SafeExecutionErrorCode,
    SafeExecutionRequest,
    execute_safe_operation_plan,
    undo_safe_operation_execution,
)
from backend.app.services import get_operation_plan, validate_operation_plan
from backend.app.workflow_graph import open_checkpointed_workflow_graph


WORKFLOW_ID = UUID("66c8d4ba-a042-4491-a5d2-ad28cb47b8d9")
PLAN_ID = UUID("2d053752-d3c4-45cb-b696-bd043e78ed92")
NOW = datetime(2026, 9, 2, 8, 0, tzinfo=timezone.utc)


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'quarantine-execution.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _approved_request(
    engine: Engine,
    tmp_path: Path,
) -> tuple[Path, Path, SafeExecutionRequest]:
    workspace_root = tmp_path / "workspace"
    source_path = workspace_root / "inbox" / "report.pdf"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"quarantine execution")
    quarantine_root = tmp_path / "application-quarantine"

    metadata = source_path.stat()
    plan = OperationPlan(
        plan_id=PLAN_ID,
        workspace_id=3,
        created_at=NOW,
        operations=[
            OperationPlanItem(
                operation_type="quarantine",
                source_file_id=7,
                source_relative_path="inbox/report.pdf",
                target_relative_path=(
                    f"workspace-3/{PLAN_ID}/7/report.pdf"
                ),
                source_precondition=FilePrecondition(
                    size_bytes=metadata.st_size,
                    mtime_ns=metadata.st_mtime_ns,
                    content_hash=ContentHash(
                        digest=sha256(source_path.read_bytes()).hexdigest()
                    ),
                ),
                reason=OperationReason(
                    kind="manual_selection",
                    description="由用户确认进入隔离区",
                ),
            )
        ],
    )

    with Session(engine) as session:
        session.add(
            Workspace(
                id=3,
                name="Quarantine 执行测试工作区",
                root_path=str(workspace_root),
            )
        )
        session.add(
            FileEntry(
                id=7,
                workspace_id=3,
                relative_path="inbox/report.pdf",
                name="report.pdf",
                extension=".pdf",
                size_bytes=metadata.st_size,
                mtime_ns=metadata.st_mtime_ns,
            )
        )
        plan_record = build_operation_plan_record(
            plan,
            workflow_id=WORKFLOW_ID,
        )
        plan_record.status = "APPROVED"
        session.add(plan_record)
        session.add(
            ApprovalRequest(
                workflow_id=str(WORKFLOW_ID),
                plan_id=str(PLAN_ID),
                status="APPROVED",
            )
        )
        session.commit()

    return (
        workspace_root,
        quarantine_root,
        SafeExecutionRequest(
            workflow_id=WORKFLOW_ID,
            plan=plan,
            quarantine_root=quarantine_root,
        ),
    )


def _quarantine_target(
    quarantine_root: Path,
    plan_id: UUID = PLAN_ID,
) -> Path:
    return (
        quarantine_root
        / "workspace-3"
        / str(plan_id)
        / "7"
        / "report.pdf"
    )


def test_quarantine_execution_moves_file_and_records_location_history(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root, quarantine_root, request = _approved_request(
        engine,
        tmp_path,
    )

    with Session(engine) as session:
        result = execute_safe_operation_plan(session, request, now=NOW)
        execution = get_operation_execution_by_workflow_id(
            session,
            str(WORKFLOW_ID),
        )
        assert execution is not None
        execution_item = find_operation_execution_items(
            session,
            execution.id,
        )[0]

        target_path = _quarantine_target(quarantine_root)
        assert result.status == "COMPLETED"
        assert execution_item.status == "COMPLETED"
        assert execution_item.before_location == "workspace"
        assert execution_item.after_location == "quarantine"
        assert execution_item.after_relative_path == (
            f"workspace-3/{PLAN_ID}/7/report.pdf"
        )
        assert target_path.read_bytes() == b"quarantine execution"
        assert not (workspace_root / "inbox" / "report.pdf").exists()


def test_quarantine_destination_conflict_is_recorded_without_overwrite(
    engine: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root, quarantine_root, request = _approved_request(
        engine,
        tmp_path,
    )
    target_path = _quarantine_target(quarantine_root)
    original_validate = safe_execution_module.validate_operation_plan

    def validate_then_occupy_target(
        session: Session,
        plan: OperationPlan,
        *,
        now: datetime | None = None,
        quarantine_root: Path | None = None,
    ) -> None:
        original_validate(
            session,
            plan,
            now=now,
            quarantine_root=quarantine_root,
        )
        target_path.parent.mkdir(parents=True)
        target_path.write_bytes(b"competing content")

    monkeypatch.setattr(
        safe_execution_module,
        "validate_operation_plan",
        validate_then_occupy_target,
    )

    with Session(engine) as session:
        result = execute_safe_operation_plan(session, request, now=NOW)
        execution = get_operation_execution_by_workflow_id(
            session,
            str(WORKFLOW_ID),
        )
        assert execution is not None
        execution_item = find_operation_execution_items(
            session,
            execution.id,
        )[0]

        assert result.status == "FAILED"
        assert execution_item.status == "FAILED"
        assert execution_item.error_code == "quarantine_target_conflict"
        assert target_path.read_bytes() == b"competing content"
        assert (workspace_root / "inbox" / "report.pdf").read_bytes() == (
            b"quarantine execution"
        )


def test_quarantine_undo_restores_file_and_history(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root, quarantine_root, request = _approved_request(
        engine,
        tmp_path,
    )

    with Session(engine) as session:
        execute_safe_operation_plan(session, request, now=NOW)
        result = undo_safe_operation_execution(
            session,
            WORKFLOW_ID,
            now=NOW,
            quarantine_root=quarantine_root,
        )
        execution = get_operation_execution_by_workflow_id(
            session,
            str(WORKFLOW_ID),
        )
        assert execution is not None
        execution_item = find_operation_execution_items(
            session,
            execution.id,
        )[0]

        assert result.status == "UNDONE"
        assert execution_item.status == "UNDONE"
        assert (workspace_root / "inbox" / "report.pdf").read_bytes() == (
            b"quarantine execution"
        )
        assert not _quarantine_target(quarantine_root).exists()


def test_quarantine_undo_rejects_occupied_original_path(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root, quarantine_root, request = _approved_request(
        engine,
        tmp_path,
    )
    original_path = workspace_root / "inbox" / "report.pdf"

    with Session(engine) as session:
        execute_safe_operation_plan(session, request, now=NOW)
        original_path.write_bytes(b"replacement content")

        with pytest.raises(SafeExecutionError) as error:
            undo_safe_operation_execution(
                session,
                WORKFLOW_ID,
                now=NOW,
                quarantine_root=quarantine_root,
            )

        execution = get_operation_execution_by_workflow_id(
            session,
            str(WORKFLOW_ID),
        )
        assert execution is not None
        execution_item = find_operation_execution_items(
            session,
            execution.id,
        )[0]
        assert error.value.code is SafeExecutionErrorCode.UNDO_TARGET_CONFLICT
        assert execution.status == "COMPLETED"
        assert execution_item.status == "COMPLETED"
        assert _quarantine_target(quarantine_root).read_bytes() == (
            b"quarantine execution"
        )


def test_quarantine_end_to_end_proposal_approval_execution_history_and_undo(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "e2e-workspace"
    source_path = workspace_root / "inbox" / "report.pdf"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"quarantine e2e")
    quarantine_root = tmp_path / "application-quarantine"
    checkpoint_path = tmp_path / "workflow-checkpoints.sqlite"
    workflow_id = UUID("44444444-4444-4444-8444-444444444444")
    plan_id = UUID("55555555-5555-4555-8555-555555555555")

    with Session(engine) as session:
        metadata = source_path.stat()
        workspace = Workspace(
            id=3,
            name="Quarantine E2E 工作区",
            root_path=str(workspace_root),
        )
        session.add(workspace)
        session.add(
            FileEntry(
                id=7,
                workspace_id=workspace.id,
                relative_path="inbox/report.pdf",
                name="report.pdf",
                extension=".pdf",
                size_bytes=metadata.st_size,
                mtime_ns=metadata.st_mtime_ns,
            )
        )
        session.commit()

        with open_checkpointed_workflow_graph(
            checkpoint_path,
            operation_plan_validator=partial(validate_operation_plan, session),
        ) as graph:
            proposal = build_propose_quarantine_tool(
                session,
                graph,
                quarantine_root=quarantine_root,
                workflow_id_factory=lambda: workflow_id,
                plan_id_factory=lambda: plan_id,
            ).invoke(
                {
                    "workspace_id": workspace.id,
                    "source_file_id": 7,
                }
            )
            assert proposal.ok is True
            proposed_plan = get_operation_plan(
                session,
                plan_id,
                workflow_id=workflow_id,
            )
            assert proposed_plan is not None
            assert proposed_plan.operations[0].operation_type == "quarantine"
            assert not quarantine_root.exists()

            applied = apply_organization_decision(
                session,
                graph,
                workflow_id,
                plan_id,
                "approve",
            )
            assert applied.approval_status == "APPROVED"

            execution_now = datetime.now(timezone.utc)
            executed = execute_safe_operation_plan(
                session,
                SafeExecutionRequest(
                    workflow_id=workflow_id,
                    plan=proposed_plan,
                    quarantine_root=quarantine_root,
                ),
                now=execution_now,
            )
            assert executed.status == "COMPLETED"
            assert executed.items[0].after_relative_path == (
                f"workspace-3/{plan_id}/7/report.pdf"
            )
            assert _quarantine_target(quarantine_root, plan_id).read_bytes() == (
                b"quarantine e2e"
            )

            execution = get_operation_execution_by_workflow_id(
                session,
                str(workflow_id),
            )
            assert execution is not None
            history_item = find_operation_execution_items(
                session,
                execution.id,
            )[0]
            assert history_item.before_location == "workspace"
            assert history_item.after_location == "quarantine"

            undone = undo_safe_operation_execution(
                session,
                workflow_id,
                now=execution_now,
                quarantine_root=quarantine_root,
            )
            assert undone.status == "UNDONE"
            assert source_path.read_bytes() == b"quarantine e2e"
            assert not _quarantine_target(quarantine_root, plan_id).exists()
