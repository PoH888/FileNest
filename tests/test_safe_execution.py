from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import backend.app.safe_execution as safe_execution_module
from backend.app.database import Base
from backend.app.models import ApprovalRequest, FileEntry, Workspace
from backend.app.operation_plan import (
    FilePrecondition,
    OperationPlan,
    OperationPlanItem,
    OperationReason,
)
from backend.app.safe_execution import (
    SafeExecutionRequest,
    validate_safe_execution_request,
)
from backend.app.services import (
    OperationPlanApprovalError,
    OperationPlanApprovalErrorCode,
)


WORKFLOW_ID = UUID("66c8d4ba-a042-4491-a5d2-ad28cb47b8d9")
PLAN_ID = UUID("2d053752-d3c4-45cb-b696-bd043e78ed92")
OTHER_PLAN_ID = UUID("37cb1621-44db-49cd-9251-31c7e871e34d")
NOW = datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc)


def _workspace_and_plan(tmp_path: Path) -> tuple[Path, OperationPlan]:
    workspace_root = tmp_path / "safe-execution-workspace"
    source_path = workspace_root / "inbox" / "report.pdf"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"approved content")
    (workspace_root / "documents" / "reports").mkdir(parents=True)

    metadata = source_path.stat()
    plan = OperationPlan(
        plan_id=PLAN_ID,
        workspace_id=3,
        created_at=NOW,
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


def test_safe_execution_request_is_strict_and_immutable(tmp_path: Path) -> None:
    _, plan = _workspace_and_plan(tmp_path)
    request = SafeExecutionRequest(workflow_id=WORKFLOW_ID, plan=plan)

    with pytest.raises(ValidationError):
        request.workflow_id = OTHER_PLAN_ID

    with pytest.raises(ValidationError):
        SafeExecutionRequest(
            workflow_id=WORKFLOW_ID,
            plan=plan,
            unexpected=True,
        )


@pytest.mark.parametrize(
    ("status", "approved_plan_id", "expected_code"),
    [
        (None, PLAN_ID, OperationPlanApprovalErrorCode.NOT_FOUND),
        (
            "WAITING_APPROVAL",
            PLAN_ID,
            OperationPlanApprovalErrorCode.NOT_APPROVED,
        ),
        ("REJECTED", PLAN_ID, OperationPlanApprovalErrorCode.NOT_APPROVED),
        (
            "APPROVED",
            OTHER_PLAN_ID,
            OperationPlanApprovalErrorCode.PLAN_MISMATCH,
        ),
    ],
)
def test_unapproved_request_never_reaches_operation_plan_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str | None,
    approved_plan_id: UUID,
    expected_code: OperationPlanApprovalErrorCode,
) -> None:
    _, plan = _workspace_and_plan(tmp_path)
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'safe-execution-rejected.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)
    plan_validator = Mock()
    monkeypatch.setattr(
        safe_execution_module,
        "validate_operation_plan",
        plan_validator,
    )

    try:
        with Session(engine) as session:
            if status is not None:
                _add_approval(
                    session,
                    status=status,
                    plan_id=approved_plan_id,
                )

            request = SafeExecutionRequest(
                workflow_id=WORKFLOW_ID,
                plan=plan,
            )
            with pytest.raises(OperationPlanApprovalError) as error:
                validate_safe_execution_request(session, request, now=NOW)

        assert error.value.code == expected_code
        plan_validator.assert_not_called()
    finally:
        engine.dispose()


def test_matching_approval_validates_current_state_without_writing_disk(
    tmp_path: Path,
) -> None:
    workspace_root, plan = _workspace_and_plan(tmp_path)
    source_path = workspace_root / "inbox" / "report.pdf"
    target_path = workspace_root / "documents" / "reports" / "report.pdf"
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'safe-execution-approved.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            session.add_all(
                [
                    Workspace(
                        id=plan.workspace_id,
                        name="安全执行测试工作区",
                        root_path=str(workspace_root),
                    ),
                    FileEntry(
                        id=7,
                        workspace_id=plan.workspace_id,
                        relative_path="inbox/report.pdf",
                        name="report.pdf",
                        extension=".pdf",
                        size_bytes=source_path.stat().st_size,
                        mtime_ns=source_path.stat().st_mtime_ns,
                    ),
                ]
            )
            session.commit()
            _add_approval(session, status="APPROVED")

            request = SafeExecutionRequest(
                workflow_id=WORKFLOW_ID,
                plan=plan,
            )
            validate_safe_execution_request(session, request, now=NOW)

        assert source_path.read_bytes() == b"approved content"
        assert not target_path.exists()
    finally:
        engine.dispose()
