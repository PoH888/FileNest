from collections.abc import Iterator
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import FileEntry, Workspace
from backend.app.operation_plan import (
    FilePrecondition,
    OperationPlan,
    OperationPlanItem,
    OperationReason,
)
from backend.app.path_policy import PathPolicyError
from backend.app.services import validate_operation_plan
from backend.app.workflow import (
    WorkflowEvent,
    WorkflowState,
    WorkflowTransitionError,
    WorkflowTransitionErrorCode,
)
from backend.app.workflow_graph import (
    WorkflowBoundaryError,
    WorkflowBoundaryErrorCode,
    WorkflowCheckpointError,
    WorkflowCheckpointErrorCode,
    build_workflow_graph,
    open_checkpointed_workflow_graph,
    run_checkpointed_workflow_event,
    run_workflow_event,
    workflow_checkpoint_config,
)


WORKFLOW_ID = UUID("66c8d4ba-a042-4491-a5d2-ad28cb47b8d9")
PLAN_ID = UUID("2d053752-d3c4-45cb-b696-bd043e78ed92")
PLAN_CREATED_AT = datetime(2026, 8, 30, 8, 0, tzinfo=timezone.utc)


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'workflow-boundary.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _plan() -> OperationPlan:
    return OperationPlan(
        plan_id=PLAN_ID,
        workspace_id=3,
        created_at=PLAN_CREATED_AT,
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


def _state() -> WorkflowState:
    return WorkflowState(
        workflow_id=WORKFLOW_ID,
        operation_plan=_plan(),
    )


def _persisted_state(
    session: Session,
    workspace_root: Path,
    *,
    target_relative_path: str,
) -> WorkflowState:
    source_path = workspace_root / "inbox" / "report.pdf"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"workflow service boundary")
    (workspace_root / "documents" / "reports").mkdir(parents=True)

    workspace = Workspace(name="工作流边界", root_path=str(workspace_root))
    session.add(workspace)
    session.flush()

    source_metadata = source_path.stat()
    source_entry = FileEntry(
        workspace_id=workspace.id,
        relative_path="inbox/report.pdf",
        name="report.pdf",
        extension=".pdf",
        size_bytes=source_metadata.st_size,
        mtime_ns=source_metadata.st_mtime_ns,
    )
    session.add(source_entry)
    session.commit()

    plan = OperationPlan(
        plan_id=PLAN_ID,
        workspace_id=workspace.id,
        created_at=PLAN_CREATED_AT,
        operations=[
            OperationPlanItem(
                source_file_id=source_entry.id,
                source_relative_path="inbox/report.pdf",
                target_relative_path=target_relative_path,
                source_precondition=FilePrecondition(
                    size_bytes=source_metadata.st_size,
                    mtime_ns=source_metadata.st_mtime_ns,
                ),
                reason=OperationReason(
                    kind="manual_selection",
                    description="由用户确认目标目录",
                ),
            )
        ],
    )
    return WorkflowState(
        workflow_id=WORKFLOW_ID,
        operation_plan=plan,
    )


def _event(
    kind: str,
    sequence_no: int,
    **details: object,
) -> WorkflowEvent:
    values: dict[str, object] = {
        "workflow_id": WORKFLOW_ID,
        "sequence_no": sequence_no,
        "kind": kind,
    }
    values.update(details)
    return WorkflowEvent(**values)


def test_graph_has_one_explicit_business_node_between_start_and_end() -> None:
    graph = build_workflow_graph()
    drawable = graph.get_graph()

    assert set(drawable.nodes) == {"__start__", "apply_event", "__end__"}
    assert {(edge.source, edge.target) for edge in drawable.edges} == {
        ("__start__", "apply_event"),
        ("apply_event", "__end__"),
    }


def test_graph_runs_pause_and_resume_through_existing_state_machine() -> None:
    graph = build_workflow_graph()
    initial = _state()

    waiting = run_workflow_event(
        graph,
        initial,
        _event(
            "pause_requested",
            1,
            reason_code="external_input_required",
        ),
    )
    resumed = run_workflow_event(
        graph,
        waiting,
        _event("resume_requested", 2),
    )

    assert waiting.status == "waiting"
    assert waiting.revision == 1
    assert resumed.status == "ready"
    assert resumed.revision == 2
    assert resumed.operation_plan == initial.operation_plan


def test_graph_preserves_workflow_transition_errors() -> None:
    graph = build_workflow_graph(operation_plan_validator=lambda plan: None)
    completed = run_workflow_event(
        graph,
        _state(),
        _event("workflow_completed", 1),
    )

    with pytest.raises(WorkflowTransitionError) as error:
        run_workflow_event(
            graph,
            completed,
            _event("resume_requested", 2),
        )

    assert error.value.code == WorkflowTransitionErrorCode.INVALID_TRANSITION


def test_graph_output_is_revalidated_as_filenest_workflow_state() -> None:
    validated_plans: list[OperationPlan] = []
    graph = build_workflow_graph(
        operation_plan_validator=validated_plans.append,
    )
    initial = _state()

    result = run_workflow_event(
        graph,
        initial,
        _event("workflow_completed", 1),
    )

    assert isinstance(result, WorkflowState)
    assert result.status == "completed"
    assert validated_plans == [initial.operation_plan]


def test_graph_fails_closed_when_completion_service_is_missing() -> None:
    graph = build_workflow_graph()

    with pytest.raises(WorkflowBoundaryError) as error:
        run_workflow_event(
            graph,
            _state(),
            _event("workflow_completed", 1),
        )

    assert (
        error.value.code
        == WorkflowBoundaryErrorCode.OPERATION_PLAN_VALIDATOR_REQUIRED
    )


def test_graph_completion_delegates_to_real_service_boundary(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "valid-workspace"
    with Session(engine) as session:
        initial = _persisted_state(
            session,
            workspace_root,
            target_relative_path="documents/reports/report.pdf",
        )
        graph = build_workflow_graph(
            operation_plan_validator=partial(
                validate_operation_plan,
                session,
                now=PLAN_CREATED_AT,
            )
        )

        completed = run_workflow_event(
            graph,
            initial,
            _event("workflow_completed", 1),
        )

    assert completed.status == "completed"
    assert (workspace_root / "inbox" / "report.pdf").is_file()
    assert not (
        workspace_root / "documents" / "reports" / "report.pdf"
    ).exists()


def test_graph_preserves_policy_rejection_without_mutating_disk(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "blocked-workspace"
    with Session(engine) as session:
        initial = _persisted_state(
            session,
            workspace_root,
            target_relative_path=".git/report.pdf",
        )
        graph = build_workflow_graph(
            operation_plan_validator=partial(
                validate_operation_plan,
                session,
                now=PLAN_CREATED_AT,
            )
        )

        with pytest.raises(PathPolicyError) as error:
            run_workflow_event(
                graph,
                initial,
                _event("workflow_completed", 1),
            )

    assert error.value.code.value == "sensitive_path"
    assert initial.status == "ready"
    assert (workspace_root / "inbox" / "report.pdf").is_file()
    assert not (workspace_root / ".git" / "report.pdf").exists()


def test_sqlite_checkpoint_restores_waiting_workflow_after_reopen(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "workflow-checkpoints.sqlite"
    initial = _state()

    with open_checkpointed_workflow_graph(checkpoint_path) as graph:
        waiting = run_checkpointed_workflow_event(
            graph,
            _event(
                "pause_requested",
                1,
                reason_code="external_input_required",
            ),
            workflow=initial,
        )
        snapshot = graph.get_state(
            workflow_checkpoint_config(WORKFLOW_ID)
        ).values

        assert waiting.status == "waiting"
        assert isinstance(snapshot["workflow"], dict)
        assert snapshot["workflow"]["status"] == "waiting"

    assert checkpoint_path.is_file()

    with open_checkpointed_workflow_graph(checkpoint_path) as graph:
        resumed = run_checkpointed_workflow_event(
            graph,
            _event("resume_requested", 2),
        )

    assert resumed.status == "ready"
    assert resumed.revision == 2
    assert resumed.operation_plan == initial.operation_plan


def test_checkpoint_resume_rejects_missing_workflow(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "missing-workflow.sqlite"

    with open_checkpointed_workflow_graph(checkpoint_path) as graph:
        with pytest.raises(WorkflowCheckpointError) as error:
            run_checkpointed_workflow_event(
                graph,
                _event("resume_requested", 1),
            )

    assert error.value.code == WorkflowCheckpointErrorCode.NOT_FOUND


def test_checkpoint_create_rejects_existing_workflow(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "existing-workflow.sqlite"

    with open_checkpointed_workflow_graph(checkpoint_path) as graph:
        run_checkpointed_workflow_event(
            graph,
            _event(
                "pause_requested",
                1,
                reason_code="external_input_required",
            ),
            workflow=_state(),
        )

        with pytest.raises(WorkflowCheckpointError) as error:
            run_checkpointed_workflow_event(
                graph,
                _event("workflow_completed", 1),
                workflow=_state(),
            )

    assert error.value.code == WorkflowCheckpointErrorCode.ALREADY_EXISTS
