from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import ApprovalRequest, Workspace
from backend.app.operation_plan import (
    FilePrecondition,
    OperationPlan,
    OperationPlanItem,
    OperationReason,
)
from backend.app.organization_planning import build_operation_plan_record
from backend.app.services import (
    OperationPlanApprovalError,
    OperationPlanApprovalErrorCode,
    require_approved_operation_plan,
)


DiskEntry = tuple[str, int, int, int, int, str | None]
WORKFLOW_ID = UUID("66c8d4ba-a042-4491-a5d2-ad28cb47b8d9")
PLAN_ID = UUID("2d053752-d3c4-45cb-b696-bd043e78ed92")
OTHER_PLAN_ID = UUID("37cb1621-44db-49cd-9251-31c7e871e34d")


def _snapshot_workspace(workspace_root: Path) -> dict[str, DiskEntry]:
    paths = [
        workspace_root,
        *sorted(workspace_root.rglob("*"), key=lambda path: path.as_posix()),
    ]
    snapshot: dict[str, DiskEntry] = {}
    for path in paths:
        relative_path = "."
        if path != workspace_root:
            relative_path = path.relative_to(workspace_root).as_posix()
        if path.is_file():
            kind = "file"
            content_digest = sha256(path.read_bytes()).hexdigest()
        elif path.is_dir():
            kind = "directory"
            content_digest = None
        else:
            kind = "other"
            content_digest = None

        metadata = path.stat()
        snapshot[relative_path] = (
            kind,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
            content_digest,
        )
    return snapshot


def _workspace_and_plan(tmp_path: Path) -> tuple[Path, OperationPlan]:
    workspace_root = tmp_path / "approval-workspace"
    source_path = workspace_root / "inbox" / "report.pdf"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"must not move without approval")
    (workspace_root / "documents" / "reports").mkdir(parents=True)

    metadata = source_path.stat()
    plan = OperationPlan(
        plan_id=PLAN_ID,
        workspace_id=3,
        created_at=datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc),
        operations=[
            OperationPlanItem(
                source_file_id=7,
                source_relative_path="inbox/report.pdf",
                target_relative_path="documents/reports/report.pdf",
                source_precondition=FilePrecondition(
                    size_bytes=metadata.st_size,
                    mtime_ns=metadata.st_mtime_ns,
                ),
                reason=OperationReason(
                    kind="manual_selection",
                    description="由用户确认目标目录",
                ),
            )
        ],
    )
    return workspace_root, plan


def _add_approval(
    session: Session,
    *,
    status: str,
    plan_id: UUID = PLAN_ID,
) -> None:
    session.add(
        ApprovalRequest(
            workflow_id=str(WORKFLOW_ID),
            plan_id=str(plan_id),
            status=status,
        )
    )
    session.commit()


@pytest.mark.parametrize("status", ["WAITING_APPROVAL", "REJECTED"])
def test_unapproved_status_leaves_complete_disk_snapshot_unchanged(
    tmp_path: Path,
    status: str,
) -> None:
    workspace_root, plan = _workspace_and_plan(tmp_path)
    engine = create_engine(
        f"sqlite:///{(tmp_path / f'approval-{status}.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            _add_approval(session, status=status)
            before = _snapshot_workspace(workspace_root)

            with pytest.raises(OperationPlanApprovalError) as error:
                require_approved_operation_plan(session, WORKFLOW_ID, plan)

            after = _snapshot_workspace(workspace_root)

        assert error.value.code == OperationPlanApprovalErrorCode.NOT_APPROVED
        assert after == before
    finally:
        engine.dispose()


def test_missing_approval_leaves_complete_disk_snapshot_unchanged(
    tmp_path: Path,
) -> None:
    workspace_root, plan = _workspace_and_plan(tmp_path)
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'approval-missing.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            before = _snapshot_workspace(workspace_root)

            with pytest.raises(OperationPlanApprovalError) as error:
                require_approved_operation_plan(session, WORKFLOW_ID, plan)

            after = _snapshot_workspace(workspace_root)

        assert error.value.code == OperationPlanApprovalErrorCode.NOT_FOUND
        assert after == before
    finally:
        engine.dispose()


def test_mismatched_approved_plan_leaves_disk_unchanged(
    tmp_path: Path,
) -> None:
    workspace_root, plan = _workspace_and_plan(tmp_path)
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'approval-mismatch.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            _add_approval(
                session,
                status="APPROVED",
                plan_id=OTHER_PLAN_ID,
            )
            before = _snapshot_workspace(workspace_root)

            with pytest.raises(OperationPlanApprovalError) as error:
                require_approved_operation_plan(session, WORKFLOW_ID, plan)

            after = _snapshot_workspace(workspace_root)

        assert error.value.code == OperationPlanApprovalErrorCode.PLAN_MISMATCH
        assert after == before
    finally:
        engine.dispose()


def test_matching_approval_only_authorizes_without_writing_disk(
    tmp_path: Path,
) -> None:
    workspace_root, plan = _workspace_and_plan(tmp_path)
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'approval-matched.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            session.add(
                Workspace(
                    id=3,
                    name="审批守卫测试工作区",
                    root_path=str(workspace_root),
                )
            )
            session.flush()
            persisted_plan = build_operation_plan_record(
                plan,
                workflow_id=WORKFLOW_ID,
            )
            persisted_plan.status = "APPROVED"
            session.add(persisted_plan)
            session.flush()
            _add_approval(session, status="APPROVED")
            before = _snapshot_workspace(workspace_root)

            approval = require_approved_operation_plan(
                session,
                WORKFLOW_ID,
                plan,
            )

            after = _snapshot_workspace(workspace_root)

        assert approval.status == "APPROVED"
        assert approval.plan_id == str(PLAN_ID)
        assert after == before
    finally:
        engine.dispose()
