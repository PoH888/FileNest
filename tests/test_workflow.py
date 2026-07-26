from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from backend.app.operation_plan import (
    FilePrecondition,
    OperationPlan,
    OperationPlanItem,
    OperationReason,
)
from backend.app.workflow import (
    WorkflowEvent,
    WorkflowState,
    WorkflowTransitionError,
    WorkflowTransitionErrorCode,
    transition_workflow,
)


WORKFLOW_ID = UUID("66c8d4ba-a042-4491-a5d2-ad28cb47b8d9")
OTHER_WORKFLOW_ID = UUID("90be200b-48f0-4209-8074-ddf55046bd08")
PLAN_ID = UUID("2d053752-d3c4-45cb-b696-bd043e78ed92")


def _plan() -> OperationPlan:
    return OperationPlan(
        plan_id=PLAN_ID,
        workspace_id=3,
        created_at=datetime(2026, 8, 30, 8, 0, tzinfo=timezone.utc),
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


def _event(
    kind: str,
    sequence_no: int,
    **overrides: object,
) -> WorkflowEvent:
    values: dict[str, object] = {
        "workflow_id": WORKFLOW_ID,
        "sequence_no": sequence_no,
        "kind": kind,
    }
    values.update(overrides)
    return WorkflowEvent(**values)


def test_workflow_pauses_and_resumes_without_changing_operation_plan() -> None:
    initial = _state()

    waiting = transition_workflow(
        initial,
        _event(
            "pause_requested",
            1,
            reason_code="external_input_required",
        ),
    )
    resumed = transition_workflow(
        waiting,
        _event("resume_requested", 2),
    )

    assert waiting.status == "waiting"
    assert waiting.wait_reason_code == "external_input_required"
    assert waiting.revision == 1
    assert resumed.status == "ready"
    assert resumed.wait_reason_code is None
    assert resumed.revision == 2
    assert resumed.operation_plan == initial.operation_plan


def test_workflow_completes_only_from_ready_state() -> None:
    completed = transition_workflow(
        _state(),
        _event("workflow_completed", 1),
    )

    assert completed.status == "completed"
    assert completed.revision == 1

    with pytest.raises(WorkflowTransitionError) as error:
        transition_workflow(
            completed,
            _event("pause_requested", 2, reason_code="too_late"),
        )

    assert error.value.code == WorkflowTransitionErrorCode.INVALID_TRANSITION


def test_workflow_failure_records_only_a_stable_error_code() -> None:
    failed = transition_workflow(
        _state(),
        _event("workflow_failed", 1, error_code="checkpoint_write_failed"),
    )

    assert failed.status == "failed"
    assert failed.error_code == "checkpoint_write_failed"
    assert failed.wait_reason_code is None


def test_workflow_rejects_event_for_another_workflow() -> None:
    event = _event("workflow_completed", 1, workflow_id=OTHER_WORKFLOW_ID)

    with pytest.raises(WorkflowTransitionError) as error:
        transition_workflow(_state(), event)

    assert error.value.code == WorkflowTransitionErrorCode.WORKFLOW_MISMATCH


def test_workflow_rejects_out_of_order_event() -> None:
    with pytest.raises(WorkflowTransitionError) as error:
        transition_workflow(
            _state(),
            _event("workflow_completed", 2),
        )

    assert error.value.code == WorkflowTransitionErrorCode.EVENT_SEQUENCE_MISMATCH


@pytest.mark.parametrize(
    "values",
    [
        {
            "kind": "pause_requested",
            "sequence_no": 1,
        },
        {
            "kind": "resume_requested",
            "sequence_no": 1,
            "reason_code": "unexpected_reason",
        },
        {
            "kind": "workflow_failed",
            "sequence_no": 1,
        },
        {
            "kind": "workflow_completed",
            "sequence_no": 1,
            "error_code": "unexpected_error",
        },
    ],
)
def test_workflow_event_requires_only_kind_specific_details(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        WorkflowEvent(workflow_id=WORKFLOW_ID, **values)


def test_workflow_state_is_strict_immutable_and_checkpoint_serializable() -> None:
    state = transition_workflow(
        _state(),
        _event(
            "pause_requested",
            1,
            reason_code="external_input_required",
        ),
    )

    restored = WorkflowState.model_validate_json(state.model_dump_json())

    assert restored == state
    assert restored.operation_plan == state.operation_plan

    with pytest.raises(ValidationError):
        state.status = "ready"

    with pytest.raises(ValidationError):
        WorkflowState(
            workflow_id=WORKFLOW_ID,
            operation_plan=_plan(),
            raw_session="must not be persisted",
        )
