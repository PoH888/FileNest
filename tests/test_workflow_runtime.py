import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import backend.app.workflow_runtime as workflow_runtime
from backend.app.database import Base
from backend.app.models import FileEntry, Workspace
from backend.app.operation_plan import (
    FilePrecondition,
    OperationPlan,
    OperationPlanItem,
    OperationReason,
)
from backend.app.workflow import WorkflowEvent, WorkflowState
from backend.app.workflow_graph import run_checkpointed_workflow_event


WORKFLOW_ID = UUID("ae099914-4198-4bf9-8f1b-d0e8b19411e9")
CURRENT_PLAN_ID = UUID("bcd22e91-e18d-40dc-8267-7c442f60ba8c")
REPLACEMENT_PLAN_ID = UUID("dfcb9aba-d95f-4c99-9e85-33f88d72bc0d")


def test_runtime_resolves_configured_checkpoint_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_path = tmp_path / "persistent" / "workflow-checkpoints.sqlite"
    monkeypatch.setenv(
        "FILENEST_WORKFLOW_CHECKPOINT_PATH",
        str(configured_path),
    )

    assert workflow_runtime._resolve_workflow_checkpoint_path() == configured_path


def _plan(
    *,
    plan_id: UUID,
    workspace_id: int,
    file_id: int,
    target_relative_path: str,
    metadata: os.stat_result,
    created_at: datetime,
) -> OperationPlan:
    return OperationPlan(
        plan_id=plan_id,
        workspace_id=workspace_id,
        created_at=created_at,
        operations=(
            OperationPlanItem(
                source_file_id=file_id,
                source_relative_path="inbox/report.txt",
                target_relative_path=target_relative_path,
                source_precondition=FilePrecondition(
                    size_bytes=metadata.st_size,
                    mtime_ns=metadata.st_mtime_ns,
                ),
                reason=OperationReason(
                    kind="manual_selection",
                    description="采用用户确认的目标目录",
                ),
            ),
        ),
    )


def test_runtime_graph_validates_and_checkpoints_plan_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'workflow-runtime.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)
    workspace_root = tmp_path / "workspace"
    source_path = workspace_root / "inbox" / "report.txt"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"validated runtime replacement")
    (workspace_root / "archive").mkdir(parents=True)
    (workspace_root / "reports").mkdir(parents=True)
    checkpoint_path = tmp_path / "workflow-checkpoints.sqlite"
    monkeypatch.setattr(
        workflow_runtime,
        "WORKFLOW_CHECKPOINT_PATH",
        checkpoint_path,
    )

    try:
        with Session(engine) as session:
            workspace = Workspace(
                name="运行时计划校验",
                root_path=str(workspace_root),
            )
            session.add(workspace)
            session.flush()
            metadata = source_path.stat()
            file_entry = FileEntry(
                workspace_id=workspace.id,
                relative_path="inbox/report.txt",
                name="report.txt",
                extension=".txt",
                size_bytes=metadata.st_size,
                mtime_ns=metadata.st_mtime_ns,
            )
            session.add(file_entry)
            session.commit()
            created_at = datetime.now(timezone.utc)
            current_plan = _plan(
                plan_id=CURRENT_PLAN_ID,
                workspace_id=workspace.id,
                file_id=file_entry.id,
                target_relative_path="reports/report.txt",
                metadata=metadata,
                created_at=created_at,
            )
            replacement_plan = _plan(
                plan_id=REPLACEMENT_PLAN_ID,
                workspace_id=workspace.id,
                file_id=file_entry.id,
                target_relative_path="archive/report.txt",
                metadata=metadata,
                created_at=created_at,
            )

            dependency = workflow_runtime.get_workflow_graph(session)
            graph = next(dependency)
            try:
                waiting = run_checkpointed_workflow_event(
                    graph,
                    WorkflowEvent(
                        workflow_id=WORKFLOW_ID,
                        sequence_no=1,
                        kind="pause_requested",
                        reason_code="human_approval_required",
                    ),
                    workflow=WorkflowState(
                        workflow_id=WORKFLOW_ID,
                        operation_plan=current_plan,
                    ),
                )
                updated = run_checkpointed_workflow_event(
                    graph,
                    WorkflowEvent(
                        workflow_id=WORKFLOW_ID,
                        sequence_no=2,
                        kind="plan_replaced",
                        replacement_plan=replacement_plan,
                    ),
                )
            finally:
                dependency.close()

            assert waiting.status == "waiting"
            assert updated.status == "waiting"
            assert updated.operation_plan == replacement_plan
            assert checkpoint_path.is_file()
    finally:
        engine.dispose()
