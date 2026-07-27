from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.approval_recovery import (
    ApprovalRecoveryError,
    ApprovalRecoveryErrorCode,
    recover_waiting_approval_tasks,
)
from backend.app.database import Base
from backend.app.models import ApprovalRequest
from backend.app.operation_plan import (
    FilePrecondition,
    OperationPlan,
    OperationPlanItem,
    OperationReason,
)
from backend.app.workflow import WorkflowEvent, WorkflowState
from backend.app.workflow_graph import (
    open_checkpointed_workflow_graph,
    run_checkpointed_workflow_event,
)


WORKFLOW_ID = UUID("66c8d4ba-a042-4491-a5d2-ad28cb47b8d9")
PLAN_ID = UUID("2d053752-d3c4-45cb-b696-bd043e78ed92")
OTHER_PLAN_ID = UUID("37cb1621-44db-49cd-9251-31c7e871e34d")


def _plan(plan_id: UUID = PLAN_ID) -> OperationPlan:
    return OperationPlan(
        plan_id=plan_id,
        workspace_id=3,
        created_at=datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc),
        operations=[
            OperationPlanItem(
                source_file_id=7,
                source_relative_path="inbox/report.pdf",
                target_relative_path="documents/reports/report.pdf",
                source_precondition=FilePrecondition(
                    size_bytes=4096,
                    mtime_ns=1_777_777_777_000_000_000,
                ),
                reason=OperationReason(
                    kind="manual_selection",
                    description="由用户确认目标目录",
                ),
            )
        ],
    )


def _pause_event(reason_code: str) -> WorkflowEvent:
    return WorkflowEvent(
        workflow_id=WORKFLOW_ID,
        sequence_no=1,
        kind="pause_requested",
        reason_code=reason_code,
    )


def _create_business_database(database_path: Path) -> None:
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    Base.metadata.create_all(bind=engine)
    engine.dispose()


def _add_approval(
    database_path: Path,
    *,
    status: str = "WAITING_APPROVAL",
    plan_id: UUID = PLAN_ID,
    workflow_id: UUID = WORKFLOW_ID,
) -> int:
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    try:
        with Session(engine) as session:
            approval = ApprovalRequest(
                workflow_id=str(workflow_id),
                plan_id=str(plan_id),
                status=status,
            )
            session.add(approval)
            session.commit()
            return approval.id
    finally:
        engine.dispose()


def _write_waiting_checkpoint(
    checkpoint_path: Path,
    *,
    reason_code: str = "human_approval_required",
) -> WorkflowState:
    initial = WorkflowState(
        workflow_id=WORKFLOW_ID,
        operation_plan=_plan(),
    )
    with open_checkpointed_workflow_graph(checkpoint_path) as graph:
        return run_checkpointed_workflow_event(
            graph,
            _pause_event(reason_code),
            workflow=initial,
        )


def test_restart_recovers_matching_waiting_approval_task(
    tmp_path: Path,
) -> None:
    business_path = tmp_path / "business.db"
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    _create_business_database(business_path)
    approval_id = _add_approval(business_path)
    _add_approval(
        business_path,
        status="APPROVED",
        workflow_id=UUID("d42c3835-1f83-4ddb-8276-2cb768a90f46"),
    )
    _add_approval(
        business_path,
        status="REJECTED",
        workflow_id=UUID("20c20ed4-b91c-42ce-98f2-356911db7a86"),
    )
    waiting_before_restart = _write_waiting_checkpoint(checkpoint_path)

    # 新 Engine、Session 和 checkpoint 连接共同模拟服务进程重启。
    restarted_engine = create_engine(f"sqlite:///{business_path.as_posix()}")
    try:
        with (
            Session(restarted_engine) as session,
            open_checkpointed_workflow_graph(checkpoint_path) as graph,
        ):
            recovered = recover_waiting_approval_tasks(session, graph)
    finally:
        restarted_engine.dispose()

    assert len(recovered) == 1
    task = recovered[0]
    assert task.approval_id == approval_id
    assert task.workflow_id == WORKFLOW_ID
    assert task.plan_id == PLAN_ID
    assert task.workflow == waiting_before_restart
    assert task.workflow.operation_plan == _plan()


def test_restart_rejects_waiting_business_state_without_checkpoint(
    tmp_path: Path,
) -> None:
    business_path = tmp_path / "missing-checkpoint-business.db"
    checkpoint_path = tmp_path / "missing-checkpoint.sqlite"
    _create_business_database(business_path)
    _add_approval(business_path)

    restarted_engine = create_engine(f"sqlite:///{business_path.as_posix()}")
    try:
        with (
            Session(restarted_engine) as session,
            open_checkpointed_workflow_graph(checkpoint_path) as graph,
            pytest.raises(ApprovalRecoveryError) as error,
        ):
            recover_waiting_approval_tasks(session, graph)
    finally:
        restarted_engine.dispose()

    assert error.value.code == ApprovalRecoveryErrorCode.CHECKPOINT_NOT_FOUND


def test_restart_rejects_plan_mismatch_between_business_and_checkpoint(
    tmp_path: Path,
) -> None:
    business_path = tmp_path / "mismatched-plan-business.db"
    checkpoint_path = tmp_path / "mismatched-plan-checkpoint.sqlite"
    _create_business_database(business_path)
    _add_approval(business_path, plan_id=OTHER_PLAN_ID)
    _write_waiting_checkpoint(checkpoint_path)

    restarted_engine = create_engine(f"sqlite:///{business_path.as_posix()}")
    try:
        with (
            Session(restarted_engine) as session,
            open_checkpointed_workflow_graph(checkpoint_path) as graph,
            pytest.raises(ApprovalRecoveryError) as error,
        ):
            recover_waiting_approval_tasks(session, graph)
    finally:
        restarted_engine.dispose()

    assert error.value.code == ApprovalRecoveryErrorCode.PLAN_MISMATCH


def test_restart_rejects_checkpoint_not_waiting_for_human_approval(
    tmp_path: Path,
) -> None:
    business_path = tmp_path / "wrong-reason-business.db"
    checkpoint_path = tmp_path / "wrong-reason-checkpoint.sqlite"
    _create_business_database(business_path)
    _add_approval(business_path)
    _write_waiting_checkpoint(
        checkpoint_path,
        reason_code="external_input_required",
    )

    restarted_engine = create_engine(f"sqlite:///{business_path.as_posix()}")
    try:
        with (
            Session(restarted_engine) as session,
            open_checkpointed_workflow_graph(checkpoint_path) as graph,
            pytest.raises(ApprovalRecoveryError) as error,
        ):
            recover_waiting_approval_tasks(session, graph)
    finally:
        restarted_engine.dispose()

    assert (
        error.value.code
        == ApprovalRecoveryErrorCode.NOT_WAITING_FOR_APPROVAL
    )
