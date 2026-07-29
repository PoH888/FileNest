from collections.abc import Callable, Iterator
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session

import backend.app.services as services_module
from backend.app.database import Base
from backend.app.models import ApprovalAuditEvent, ApprovalRequest
from backend.app.repositories import (
    add_approval_audit_event,
    compare_and_set_approval_request,
    find_approval_audit_events,
)
from backend.app.services import (
    ApprovalTransitionError,
    ApprovalTransitionErrorCode,
    approve_operation_plan,
    edit_operation_plan,
    reject_operation_plan,
)


WORKFLOW_ID = UUID("66c8d4ba-a042-4491-a5d2-ad28cb47b8d9")
PLAN_ID = UUID("2d053752-d3c4-45cb-b696-bd043e78ed92")
REPLACEMENT_PLAN_ID = UUID("37cb1621-44db-49cd-9251-31c7e871e34d")


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'approval-transitions.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _add_waiting_approval(session: Session) -> int:
    approval = ApprovalRequest(
        workflow_id=str(WORKFLOW_ID),
        plan_id=str(PLAN_ID),
    )
    session.add(approval)
    session.commit()
    return approval.id


@pytest.mark.parametrize(
    ("transition", "expected_status", "expected_action"),
    [
        (approve_operation_plan, "APPROVED", "approve"),
        (reject_operation_plan, "REJECTED", "reject"),
    ],
)
def test_approve_and_reject_persist_terminal_state(
    engine: Engine,
    transition: Callable[[Session, UUID, UUID], ApprovalRequest],
    expected_status: str,
    expected_action: str,
) -> None:
    with Session(engine) as session:
        approval_id = _add_waiting_approval(session)
        result = transition(session, WORKFLOW_ID, PLAN_ID)

        assert result.id == approval_id
        assert result.status == expected_status
        assert result.plan_id == str(PLAN_ID)

    with Session(engine) as session:
        restored = session.get(ApprovalRequest, approval_id)
        audit_events = find_approval_audit_events(session, approval_id)

        assert restored is not None
        assert restored.status == expected_status
        assert len(audit_events) == 1
        assert audit_events[0].action == expected_action
        assert audit_events[0].previous_status == "WAITING_APPROVAL"
        assert audit_events[0].next_status == expected_status
        assert audit_events[0].previous_plan_id == str(PLAN_ID)
        assert audit_events[0].next_plan_id == str(PLAN_ID)


def test_edit_replaces_plan_and_stale_approval_is_rejected(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        approval_id = _add_waiting_approval(session)
        edited = edit_operation_plan(
            session,
            WORKFLOW_ID,
            PLAN_ID,
            REPLACEMENT_PLAN_ID,
        )

        assert edited.status == "WAITING_APPROVAL"
        assert edited.plan_id == str(REPLACEMENT_PLAN_ID)

        with pytest.raises(ApprovalTransitionError) as error:
            approve_operation_plan(session, WORKFLOW_ID, PLAN_ID)

        assert error.value.code == ApprovalTransitionErrorCode.PLAN_MISMATCH

        approved = approve_operation_plan(
            session,
            WORKFLOW_ID,
            REPLACEMENT_PLAN_ID,
        )

        assert approved.status == "APPROVED"

    with Session(engine) as session:
        restored = session.get(ApprovalRequest, approval_id)
        audit_events = find_approval_audit_events(session, approval_id)

        assert restored is not None
        assert restored.status == "APPROVED"
        assert restored.plan_id == str(REPLACEMENT_PLAN_ID)
        assert [event.action for event in audit_events] == ["edit", "approve"]
        assert audit_events[0].previous_status == "WAITING_APPROVAL"
        assert audit_events[0].next_status == "WAITING_APPROVAL"
        assert audit_events[0].previous_plan_id == str(PLAN_ID)
        assert audit_events[0].next_plan_id == str(REPLACEMENT_PLAN_ID)
        assert audit_events[1].previous_status == "WAITING_APPROVAL"
        assert audit_events[1].next_status == "APPROVED"
        assert audit_events[1].previous_plan_id == str(REPLACEMENT_PLAN_ID)
        assert audit_events[1].next_plan_id == str(REPLACEMENT_PLAN_ID)


def test_terminal_approval_cannot_transition_again(engine: Engine) -> None:
    with Session(engine) as session:
        approval_id = _add_waiting_approval(session)
        approve_operation_plan(session, WORKFLOW_ID, PLAN_ID)

        with pytest.raises(ApprovalTransitionError) as error:
            reject_operation_plan(session, WORKFLOW_ID, PLAN_ID)

        assert error.value.code == ApprovalTransitionErrorCode.NOT_WAITING

    with Session(engine) as session:
        restored = session.get(ApprovalRequest, approval_id)
        audit_events = find_approval_audit_events(session, approval_id)

        assert restored is not None
        assert restored.status == "APPROVED"
        assert [event.action for event in audit_events] == ["approve"]


def test_repeated_approval_returns_existing_state_without_duplicate_audit(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        approval_id = _add_waiting_approval(session)
        first_result = approve_operation_plan(session, WORKFLOW_ID, PLAN_ID)
        repeated_result = approve_operation_plan(
            session,
            WORKFLOW_ID,
            PLAN_ID,
        )

        assert first_result.id == approval_id
        assert repeated_result.id == approval_id
        assert repeated_result.status == "APPROVED"

    with Session(engine) as session:
        audit_events = find_approval_audit_events(session, approval_id)
        assert [event.action for event in audit_events] == ["approve"]


def test_repeated_approval_does_not_hide_plan_mismatch(engine: Engine) -> None:
    with Session(engine) as session:
        approval_id = _add_waiting_approval(session)
        approve_operation_plan(session, WORKFLOW_ID, PLAN_ID)

        with pytest.raises(ApprovalTransitionError) as error:
            approve_operation_plan(
                session,
                WORKFLOW_ID,
                REPLACEMENT_PLAN_ID,
            )

        assert error.value.code == ApprovalTransitionErrorCode.PLAN_MISMATCH

    with Session(engine) as session:
        audit_events = find_approval_audit_events(session, approval_id)
        assert [event.action for event in audit_events] == ["approve"]


def test_concurrent_repeated_approval_reloads_winning_state(
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(engine) as setup_session:
        approval_id = _add_waiting_approval(setup_session)

    def approve_in_competing_transaction(
        current_session: Session,
        workflow_id: str,
        expected_plan_id: str,
        *,
        next_status: str,
        next_plan_id: str,
    ) -> bool:
        current_session.rollback()
        with Session(engine) as winning_session:
            assert compare_and_set_approval_request(
                winning_session,
                workflow_id,
                expected_plan_id,
                next_status=next_status,
                next_plan_id=next_plan_id,
            )
            add_approval_audit_event(
                winning_session,
                ApprovalAuditEvent(
                    approval_request_id=approval_id,
                    action="approve",
                    previous_status="WAITING_APPROVAL",
                    next_status="APPROVED",
                    previous_plan_id=str(PLAN_ID),
                    next_plan_id=str(PLAN_ID),
                ),
            )
            winning_session.commit()
        return False

    monkeypatch.setattr(
        services_module,
        "compare_and_set_approval_request",
        approve_in_competing_transaction,
    )

    with Session(engine) as session:
        result = approve_operation_plan(session, WORKFLOW_ID, PLAN_ID)

        assert result.id == approval_id
        assert result.status == "APPROVED"

    with Session(engine) as session:
        audit_events = find_approval_audit_events(session, approval_id)
        assert [event.action for event in audit_events] == ["approve"]


def test_missing_approval_request_is_rejected(engine: Engine) -> None:
    with Session(engine) as session, pytest.raises(
        ApprovalTransitionError
    ) as error:
        approve_operation_plan(session, WORKFLOW_ID, PLAN_ID)

    assert error.value.code == ApprovalTransitionErrorCode.NOT_FOUND

    with Session(engine) as session:
        assert session.query(ApprovalAuditEvent).count() == 0


def test_edit_requires_a_new_plan_id(engine: Engine) -> None:
    with Session(engine) as session:
        approval_id = _add_waiting_approval(session)

        with pytest.raises(ApprovalTransitionError) as error:
            edit_operation_plan(
                session,
                WORKFLOW_ID,
                PLAN_ID,
                PLAN_ID,
            )

        assert error.value.code == ApprovalTransitionErrorCode.PLAN_UNCHANGED

    with Session(engine) as session:
        restored = session.get(ApprovalRequest, approval_id)
        audit_events = find_approval_audit_events(session, approval_id)

        assert restored is not None
        assert restored.status == "WAITING_APPROVAL"
        assert restored.plan_id == str(PLAN_ID)
        assert audit_events == []


def test_commit_failure_rolls_back_approval_transition(engine: Engine) -> None:
    with Session(engine) as session:
        approval_id = _add_waiting_approval(session)

        def fail_commit(current_session: Session) -> None:
            raise RuntimeError("simulated commit failure")

        event.listen(session, "before_commit", fail_commit)
        try:
            with pytest.raises(RuntimeError, match="simulated commit failure"):
                approve_operation_plan(session, WORKFLOW_ID, PLAN_ID)
        finally:
            event.remove(session, "before_commit", fail_commit)

    with Session(engine) as session:
        restored = session.get(ApprovalRequest, approval_id)
        audit_events = find_approval_audit_events(session, approval_id)

        assert restored is not None
        assert restored.status == "WAITING_APPROVAL"
        assert restored.plan_id == str(PLAN_ID)
        assert audit_events == []
