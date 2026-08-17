from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import OperationStatusRecord
from backend.app.operation_projection import OperationProjection
from backend.app.operation_status import (
    OperationStatus,
    OperationStatusTransitionError,
)
from backend.app.repositories import (
    add_operation_status,
    compare_and_set_operation_status,
    get_operation_projection_by_workflow_id,
)


WORKFLOW_ID = "66c8d4ba-a042-4491-a5d2-ad28cb47b8d9"
PLAN_ID = "2d053752-d3c4-45cb-b696-bd043e78ed92"


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'operation-status.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)
    try:
        yield test_engine
    finally:
        test_engine.dispose()


def _add_status(session: Session) -> None:
    add_operation_status(
        session,
        OperationStatusRecord(
            workflow_id=WORKFLOW_ID,
            plan_id=PLAN_ID,
            approval_id=7,
            overall_status=OperationStatus.PROPOSED.value,
        ),
    )
    session.commit()


def test_operation_status_persists_and_projects_all_associations(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        _add_status(session)

        assert compare_and_set_operation_status(
            session,
            WORKFLOW_ID,
            OperationStatus.PROPOSED,
            expected_revision=0,
            next_status=OperationStatus.WAITING_APPROVAL,
            approval_id=7,
        )
        assert compare_and_set_operation_status(
            session,
            WORKFLOW_ID,
            OperationStatus.WAITING_APPROVAL,
            expected_revision=1,
            next_status=OperationStatus.APPROVED,
            execution_id=11,
        )
        session.commit()

        projection = get_operation_projection_by_workflow_id(
            session,
            WORKFLOW_ID,
        )

    assert projection == OperationProjection(
        workflow_id=UUID(WORKFLOW_ID),
        plan_id=UUID(PLAN_ID),
        approval_id=7,
        execution_id=11,
        overall_status=OperationStatus.APPROVED,
        revision=2,
    )


def test_operation_status_cas_rejects_stale_revision_and_illegal_transition(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        _add_status(session)

        assert not compare_and_set_operation_status(
            session,
            WORKFLOW_ID,
            OperationStatus.PROPOSED,
            expected_revision=1,
            next_status=OperationStatus.WAITING_APPROVAL,
        )

        with pytest.raises(OperationStatusTransitionError):
            compare_and_set_operation_status(
                session,
                WORKFLOW_ID,
                OperationStatus.PROPOSED,
                expected_revision=0,
                next_status=OperationStatus.COMPLETED,
            )
