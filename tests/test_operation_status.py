import pytest

from backend.app.operation_status import (
    OPERATION_STATUS_DEFINITIONS,
    OPERATION_STATUS_TRANSITIONS,
    OperationStatus,
    OperationStatusTransitionErrorCode,
    OperationStatusTransitionError,
    TERMINAL_OPERATION_STATUSES,
    is_terminal_operation_status,
    map_agent_run_status_to_operation_status,
    map_approval_status_to_operation_status,
    map_workflow_status_to_operation_status,
    transition_operation_status,
)


def test_operation_status_catalog_and_normal_lifecycle() -> None:
    assert set(OPERATION_STATUS_DEFINITIONS) == set(OperationStatus)
    assert TERMINAL_OPERATION_STATUSES == {
        OperationStatus.REJECTED,
        OperationStatus.CANCELLED,
        OperationStatus.UNDONE,
        OperationStatus.COMPENSATED,
    }

    status = OperationStatus.PROPOSED
    for next_status in (
        OperationStatus.WAITING_APPROVAL,
        OperationStatus.APPROVED,
        OperationStatus.EXECUTING,
        OperationStatus.COMPLETED,
    ):
        status = transition_operation_status(status, next_status)

    assert status is OperationStatus.COMPLETED
    assert not is_terminal_operation_status(status)


def test_cancelled_is_a_terminal_status_before_file_side_effects() -> None:
    assert (
        transition_operation_status(
            OperationStatus.WAITING_APPROVAL,
            OperationStatus.CANCELLED,
        )
        is OperationStatus.CANCELLED
    )
    assert is_terminal_operation_status(OperationStatus.CANCELLED)


def test_cancelled_maps_across_workflow_approval_and_agent_lifecycles() -> None:
    assert (
        map_workflow_status_to_operation_status("cancelled")
        is OperationStatus.CANCELLED
    )
    assert (
        map_approval_status_to_operation_status("CANCELLED")
        is OperationStatus.CANCELLED
    )
    assert (
        map_agent_run_status_to_operation_status("cancelled")
        is OperationStatus.CANCELLED
    )


@pytest.mark.parametrize(
    ("current_status", "next_status"),
    [
        (OperationStatus.PROPOSED, OperationStatus.WAITING_APPROVAL),
        (OperationStatus.PROPOSED, OperationStatus.CANCELLED),
        (OperationStatus.WAITING_APPROVAL, OperationStatus.APPROVED),
        (OperationStatus.WAITING_APPROVAL, OperationStatus.REJECTED),
        (OperationStatus.WAITING_APPROVAL, OperationStatus.CANCELLED),
        (OperationStatus.APPROVED, OperationStatus.EXECUTING),
        (OperationStatus.APPROVED, OperationStatus.CANCELLED),
        (OperationStatus.EXECUTING, OperationStatus.PARTIAL_FAILED),
        (OperationStatus.EXECUTING, OperationStatus.COMPLETED),
        (OperationStatus.EXECUTING, OperationStatus.FAILED),
        (OperationStatus.PARTIAL_FAILED, OperationStatus.EXECUTING),
        (OperationStatus.PARTIAL_FAILED, OperationStatus.UNDOING),
        (OperationStatus.PARTIAL_FAILED, OperationStatus.FAILED),
        (OperationStatus.COMPLETED, OperationStatus.UNDOING),
        (OperationStatus.UNDOING, OperationStatus.UNDONE),
        (OperationStatus.UNDOING, OperationStatus.COMPENSATED),
        (OperationStatus.UNDOING, OperationStatus.FAILED),
        (OperationStatus.FAILED, OperationStatus.EXECUTING),
    ],
)
def test_every_declared_operation_transition_is_accepted(
    current_status: OperationStatus,
    next_status: OperationStatus,
) -> None:
    assert next_status in OPERATION_STATUS_TRANSITIONS[current_status]
    assert transition_operation_status(current_status, next_status) is next_status


@pytest.mark.parametrize("terminal_status", sorted(TERMINAL_OPERATION_STATUSES, key=lambda status: status.value))
def test_terminal_operation_statuses_reject_all_follow_up_transitions(
    terminal_status: OperationStatus,
) -> None:
    for next_status in OperationStatus:
        with pytest.raises(OperationStatusTransitionError) as error:
            transition_operation_status(terminal_status, next_status)

        assert error.value.code is OperationStatusTransitionErrorCode.INVALID_TRANSITION


def test_failed_and_partial_failed_paths_preserve_retry_and_compensation() -> None:
    assert (
        transition_operation_status(
            OperationStatus.PARTIAL_FAILED,
            OperationStatus.FAILED,
        )
        is OperationStatus.FAILED
    )
    assert (
        transition_operation_status(
            OperationStatus.FAILED,
            OperationStatus.EXECUTING,
        )
        is OperationStatus.EXECUTING
    )
    assert (
        transition_operation_status(
            OperationStatus.PARTIAL_FAILED,
            OperationStatus.UNDOING,
        )
        is OperationStatus.UNDOING
    )


@pytest.mark.parametrize("status", list(OperationStatus))
def test_operation_status_self_transition_is_rejected(
    status: OperationStatus,
) -> None:
    with pytest.raises(OperationStatusTransitionError) as error:
        transition_operation_status(status, status)

    assert error.value.code is OperationStatusTransitionErrorCode.INVALID_TRANSITION


@pytest.mark.parametrize(
    ("current_status", "next_status"),
    [
        (OperationStatus.PROPOSED, OperationStatus.APPROVED),
        (OperationStatus.COMPLETED, OperationStatus.EXECUTING),
        (OperationStatus.EXECUTING, OperationStatus.CANCELLED),
        (OperationStatus.CANCELLED, OperationStatus.WAITING_APPROVAL),
    ],
)
def test_illegal_operation_status_transition_is_rejected(
    current_status: OperationStatus,
    next_status: OperationStatus,
) -> None:
    with pytest.raises(OperationStatusTransitionError) as error:
        transition_operation_status(current_status, next_status)

    assert error.value.current_status is current_status
    assert error.value.next_status is next_status
    assert error.value.code is OperationStatusTransitionErrorCode.INVALID_TRANSITION
