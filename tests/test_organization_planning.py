from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session

from backend.app.approval_recovery import recover_waiting_approval_tasks
from backend.app.database import Base
from backend.app.models import ApprovalRequest, FileEntry, Workspace
from backend.app.organization_planning import (
    CreateApprovalWorkflowRequest,
    EditOrganizationPlanRequest,
    OrganizationTargetSelection,
    build_organization_plan,
    create_waiting_approval_workflow,
    merge_edit_request,
)
from backend.app.workflow_graph import open_checkpointed_workflow_graph


WORKFLOW_ID = UUID("71ad238f-16a1-48f8-8cb2-7f80821112b5")
PLAN_ID = UUID("cf51a11d-1c42-4690-b591-537136173f90")


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'organization-planning.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)
    yield test_engine
    test_engine.dispose()


def _seed_workspace(
    session: Session,
    workspace_root: Path,
) -> tuple[int, int]:
    source_path = workspace_root / "inbox" / "quarterly-report.txt"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"deterministic organization planning")
    (workspace_root / "reports" / "quarterly").mkdir(parents=True)

    workspace = Workspace(name="确定性计划", root_path=str(workspace_root))
    session.add(workspace)
    session.flush()
    metadata = source_path.stat()
    file_entry = FileEntry(
        workspace_id=workspace.id,
        relative_path="inbox/quarterly-report.txt",
        name="quarterly-report.txt",
        extension=".txt",
        size_bytes=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
    )
    session.add(file_entry)
    session.commit()
    return workspace.id, file_entry.id


def _disk_snapshot(
    workspace_root: Path,
) -> dict[str, tuple[str, bytes | None]]:
    return {
        path.relative_to(workspace_root).as_posix(): (
            "file" if path.is_file() else "directory",
            path.read_bytes() if path.is_file() else None,
        )
        for path in workspace_root.rglob("*")
    }


def _request(
    workspace_id: int,
    file_id: int,
) -> CreateApprovalWorkflowRequest:
    return CreateApprovalWorkflowRequest(
        workspace_id=workspace_id,
        target_directories=("reports/quarterly",),
        selections=(
            OrganizationTargetSelection(
                source_file_id=file_id,
                target_directory="reports/quarterly",
            ),
        ),
    )


def test_create_waiting_workflow_persists_safe_plan_without_disk_mutation(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    checkpoint_path = tmp_path / "workflow-checkpoints.sqlite"
    with Session(engine) as session:
        workspace_id, file_id = _seed_workspace(session, workspace_root)
        before = _disk_snapshot(workspace_root)
        with open_checkpointed_workflow_graph(checkpoint_path) as graph:
            created = create_waiting_approval_workflow(
                session,
                graph,
                _request(workspace_id, file_id),
                workflow_id_factory=lambda: WORKFLOW_ID,
                plan_id_factory=lambda: PLAN_ID,
            )

        approval = session.scalar(select(ApprovalRequest))
        operation = created.workflow.operation_plan.operations[0]
        assert approval is not None
        assert approval.id == created.approval_id
        assert approval.workflow_id == str(WORKFLOW_ID)
        assert approval.plan_id == str(PLAN_ID)
        assert approval.status == "WAITING_APPROVAL"
        assert created.workflow.status == "waiting"
        assert created.workflow.wait_reason_code == "human_approval_required"
        assert operation.source_file_id == file_id
        assert operation.target_relative_path == (
            "reports/quarterly/quarterly-report.txt"
        )
        assert operation.source_precondition.content_hash is not None
        assert _disk_snapshot(workspace_root) == before

    with Session(engine) as restarted_session:
        with open_checkpointed_workflow_graph(checkpoint_path) as graph:
            recovered = recover_waiting_approval_tasks(restarted_session, graph)
        assert len(recovered) == 1
        assert recovered[0].workflow_id == WORKFLOW_ID


def test_request_rejects_target_that_was_not_offered() -> None:
    with pytest.raises(ValidationError):
        CreateApprovalWorkflowRequest(
            workspace_id=1,
            target_directories=("reports/quarterly",),
            selections=(
                OrganizationTargetSelection(
                    source_file_id=1,
                    target_directory="../outside",
                ),
            ),
        )


def test_checkpoint_failure_rolls_back_approval_without_disk_mutation(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "checkpoint-failure-workspace"

    class FailingGraph:
        def get_state(self, config: object) -> SimpleNamespace:
            return SimpleNamespace(values={})

        def invoke(self, input: object, config: object) -> object:
            raise RuntimeError("checkpoint write failed")

    with Session(engine) as session:
        workspace_id, file_id = _seed_workspace(session, workspace_root)
        before = _disk_snapshot(workspace_root)

        with pytest.raises(RuntimeError, match="checkpoint write failed"):
            create_waiting_approval_workflow(
                session,
                FailingGraph(),  # type: ignore[arg-type]
                _request(workspace_id, file_id),
                workflow_id_factory=lambda: WORKFLOW_ID,
                plan_id_factory=lambda: PLAN_ID,
            )

        assert session.scalar(select(ApprovalRequest)) is None
        assert _disk_snapshot(workspace_root) == before


def test_build_plan_does_not_create_approval_or_checkpoint(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "plan-only-workspace"
    with Session(engine) as session:
        workspace_id, file_id = _seed_workspace(session, workspace_root)

        plan = build_organization_plan(
            session,
            _request(workspace_id, file_id),
            plan_id_factory=lambda: PLAN_ID,
        )

        assert plan.plan_id == PLAN_ID
        assert plan.workspace_id == workspace_id
        assert plan.operations[0].target_relative_path == (
            "reports/quarterly/quarterly-report.txt"
        )
        assert session.scalar(select(ApprovalRequest)) is None


def test_merge_edit_request_preserves_sources_and_changes_only_targets(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "edit-request-workspace"
    with Session(engine) as session:
        workspace_id, file_id = _seed_workspace(session, workspace_root)
        (workspace_root / "reports" / "annual").mkdir(parents=True)
        current_plan = build_organization_plan(
            session,
            _request(workspace_id, file_id),
            plan_id_factory=lambda: PLAN_ID,
        )

        merged = merge_edit_request(
            current_plan,
            EditOrganizationPlanRequest(
                changes=(
                    OrganizationTargetSelection(
                        source_file_id=file_id,
                        target_directory="reports/annual",
                    ),
                ),
            ),
        )

        assert merged.workspace_id == workspace_id
        assert merged.selections == (
            OrganizationTargetSelection(
                source_file_id=file_id,
                target_directory="reports/annual",
            ),
        )
        assert merged.target_directories == ("reports/annual",)


def test_merge_edit_request_rejects_unknown_source(
    engine: Engine,
    tmp_path: Path,
) -> None:
    with Session(engine) as session:
        workspace_id, file_id = _seed_workspace(
            session,
            tmp_path / "unknown-source-workspace",
        )
        current_plan = build_organization_plan(
            session,
            _request(workspace_id, file_id),
            plan_id_factory=lambda: PLAN_ID,
        )

        with pytest.raises(ValueError, match="current plan"):
            merge_edit_request(
                current_plan,
                EditOrganizationPlanRequest(
                    changes=(
                        OrganizationTargetSelection(
                            source_file_id=999,
                            target_directory="reports/annual",
                        ),
                    ),
                ),
            )
