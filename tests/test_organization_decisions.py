from functools import partial
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session

import backend.app.organization_decisions as decisions_module
from backend.app.database import Base
from backend.app.models import ApprovalRequest, FileEntry, Workspace
from backend.app.organization_decisions import apply_organization_decision
from backend.app.organization_planning import (
    CreateApprovalWorkflowRequest,
    EditOrganizationPlanRequest,
    OrganizationTargetSelection,
    create_waiting_approval_workflow,
)
from backend.app.organization_decisions import apply_organization_plan_edit
from backend.app.services import validate_operation_plan
from backend.app.workflow_graph import open_checkpointed_workflow_graph


WORKFLOW_ID = UUID("d10e8d32-b68d-482c-a9cb-4f45e79b5178")
PLAN_ID = UUID("b4103c47-0ba5-45de-9621-ad91c600c6fa")


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'organization-decisions.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)
    yield test_engine
    test_engine.dispose()


def _create_waiting_workflow(
    session: Session,
    graph: object,
    workspace_root: Path,
) -> None:
    source_path = workspace_root / "inbox" / "quarterly-report.txt"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"organization decision")
    (workspace_root / "reports").mkdir(parents=True)

    workspace = Workspace(name="审批协调", root_path=str(workspace_root))
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

    create_waiting_approval_workflow(
        session,
        graph,  # type: ignore[arg-type]
        CreateApprovalWorkflowRequest(
            workspace_id=workspace.id,
            target_directories=("reports",),
            selections=(
                OrganizationTargetSelection(
                    source_file_id=file_entry.id,
                    target_directory="reports",
                ),
            ),
        ),
        workflow_id_factory=lambda: WORKFLOW_ID,
        plan_id_factory=lambda: PLAN_ID,
    )


@pytest.mark.parametrize(
    ("action", "approval_status", "workflow_status", "error_code"),
    [
        ("approve", "APPROVED", "ready", None),
        ("reject", "REJECTED", "failed", "human_rejected"),
    ],
)
def test_decision_synchronizes_approval_and_checkpoint(
    engine: Engine,
    tmp_path: Path,
    action: str,
    approval_status: str,
    workflow_status: str,
    error_code: str | None,
) -> None:
    with Session(engine) as session, open_checkpointed_workflow_graph(
        tmp_path / "workflow-checkpoints.sqlite"
    ) as graph:
        _create_waiting_workflow(session, graph, tmp_path / "workspace")

        result = apply_organization_decision(
            session,
            graph,
            WORKFLOW_ID,
            PLAN_ID,
            action,  # type: ignore[arg-type]
        )

        approval = session.scalar(select(ApprovalRequest))
        assert approval is not None
        assert approval.status == approval_status
        assert result.approval_status == approval_status
        assert result.workflow.status == workflow_status
        assert result.workflow.error_code == error_code


def test_decision_retry_repairs_checkpoint_after_first_write_failure(
    engine: Engine,
    tmp_path: Path,
) -> None:
    class FailOnceGraph:
        def __init__(self, graph: object) -> None:
            self._graph = graph
            self._failed = False

        def get_state(self, config: object) -> object:
            return self._graph.get_state(config)  # type: ignore[attr-defined]

        def invoke(self, input: object, config: object) -> object:
            if not self._failed:
                self._failed = True
                raise RuntimeError("checkpoint write failed")
            return self._graph.invoke(input, config)  # type: ignore[attr-defined]

    with Session(engine) as session, open_checkpointed_workflow_graph(
        tmp_path / "retry-workflow-checkpoints.sqlite"
    ) as graph:
        _create_waiting_workflow(session, graph, tmp_path / "retry-workspace")
        fail_once_graph = FailOnceGraph(graph)

        with pytest.raises(RuntimeError, match="checkpoint write failed"):
            apply_organization_decision(
                session,
                fail_once_graph,  # type: ignore[arg-type]
                WORKFLOW_ID,
                PLAN_ID,
                "reject",
            )

        recovered = apply_organization_decision(
            session,
            fail_once_graph,  # type: ignore[arg-type]
            WORKFLOW_ID,
            PLAN_ID,
            "reject",
        )

        assert recovered.approval_status == "REJECTED"
        assert recovered.workflow.status == "failed"


def test_edit_rebuilds_plan_and_keeps_approval_waiting(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "edit-workspace"
    (workspace_root / "archive").mkdir(parents=True)
    with Session(engine) as session, open_checkpointed_workflow_graph(
        tmp_path / "edit-workflow-checkpoints.sqlite",
        operation_plan_validator=partial(validate_operation_plan, session),
    ) as graph:
        _create_waiting_workflow(session, graph, workspace_root)

        result = apply_organization_plan_edit(
            session,
            graph,
            WORKFLOW_ID,
            PLAN_ID,
            EditOrganizationPlanRequest(
                changes=(
                    OrganizationTargetSelection(
                        source_file_id=1,
                        target_directory="archive",
                    ),
                ),
            ),
        )

        approval = session.scalar(select(ApprovalRequest))
        assert approval is not None
        assert approval.status == "WAITING_APPROVAL"
        assert result.approval_status == "WAITING_APPROVAL"
        assert result.workflow.status == "waiting"
        assert result.workflow.operation_plan.plan_id != PLAN_ID
        assert (
            result.workflow.operation_plan.operations[0].target_relative_path
            == "archive/quarterly-report.txt"
        )
        assert approval.plan_id == str(result.workflow.operation_plan.plan_id)


def test_edit_retries_after_approval_commit_failure(
    engine: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "edit-retry-workspace"
    (workspace_root / "archive").mkdir(parents=True)
    with Session(engine) as session, open_checkpointed_workflow_graph(
        tmp_path / "edit-retry-workflow-checkpoints.sqlite",
        operation_plan_validator=partial(validate_operation_plan, session),
    ) as graph:
        _create_waiting_workflow(session, graph, workspace_root)
        request = EditOrganizationPlanRequest(
            changes=(
                OrganizationTargetSelection(
                    source_file_id=1,
                    target_directory="archive",
                ),
            ),
        )
        original_edit = decisions_module.edit_operation_plan
        failed = True

        def fail_once(*args: object, **kwargs: object) -> object:
            nonlocal failed
            if failed:
                failed = False
                raise RuntimeError("approval commit failed")
            return original_edit(*args, **kwargs)

        monkeypatch.setattr(decisions_module, "edit_operation_plan", fail_once)
        with pytest.raises(RuntimeError, match="approval commit failed"):
            apply_organization_plan_edit(
                session,
                graph,
                WORKFLOW_ID,
                PLAN_ID,
                request,
            )

        recovered = apply_organization_plan_edit(
            session,
            graph,
            WORKFLOW_ID,
            PLAN_ID,
            request,
        )

        assert recovered.approval_status == "WAITING_APPROVAL"
        assert recovered.workflow.operation_plan.plan_id != PLAN_ID
        approval = session.scalar(select(ApprovalRequest))
        assert approval is not None
        assert approval.plan_id == str(recovered.workflow.operation_plan.plan_id)
