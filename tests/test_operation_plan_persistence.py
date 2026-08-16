from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import (
    ApprovalRequest,
    OperationItemRecord,
    OperationPlanRecord,
    Workspace,
)
from backend.app.services import (
    OperationPlanApprovalError,
    OperationPlanApprovalErrorCode,
    OperationPlanPersistenceError,
    get_operation_plan,
    list_operation_plan_history,
    list_operation_plan_items,
    require_approved_operation_plan,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_CONFIG_PATH = PROJECT_ROOT / "backend" / "alembic.ini"
PLAN_ID = "a5d0c4af-142e-47d7-bb6d-2b7f8bb9cf20"
HISTORY_PLAN_ID = "b6e1d5bf-253f-58e8-cc7e-3c8d9cc0d131"
WORKFLOW_ID = "8c321c8d-2904-46f2-a29a-7d77f6e4b0b8"


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'operation-plan-persistence.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)
    yield test_engine
    test_engine.dispose()


def _plan() -> OperationPlanRecord:
    return OperationPlanRecord(
        plan_id=PLAN_ID,
        workspace_id=1,
        workflow_id=WORKFLOW_ID,
        operation_type="move",
        metadata_json='{"schema_version":1}',
        status="WAITING_APPROVAL",
        created_at=datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc),
        items=[
            OperationItemRecord(
                sequence_no=1,
                operation_type="move",
                source_file_id=7,
                source_relative_path="inbox/report.txt",
                target_relative_path="reports/report.txt",
                source_size_bytes=128,
                source_mtime_ns=1_777_777_777_000_000_000,
                source_hash_algorithm="sha256",
                source_sha256="a" * 64,
                reason_kind="matched_candidate",
                reason_description="采用预览中的候选目录",
                reason_match_score=95,
                risks_json="[]",
            ),
        ],
    )


def test_plan_and_items_survive_session_restart_without_checkpoint(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        workspace = Workspace(
            name="持久化测试工作区",
            root_path="C:/filenest-operation-plan-test",
        )
        session.add(workspace)
        session.flush()
        workspace_id = workspace.id

        plan = _plan()
        plan.workspace_id = workspace_id
        session.add(plan)
        session.commit()

    with Session(engine) as restarted_session:
        restored = restarted_session.get(OperationPlanRecord, PLAN_ID)

        assert restored is not None
        assert restored.workspace_id == workspace_id
        assert restored.workflow_id == WORKFLOW_ID
        assert restored.operation_type == "move"
        assert restored.metadata_json == '{"schema_version":1}'
        assert restored.status == "WAITING_APPROVAL"
        assert restored.created_at.replace(tzinfo=timezone.utc) == datetime(
            2026,
            9,
            2,
            10,
            0,
            tzinfo=timezone.utc,
        )
        assert len(restored.items) == 1
        assert restored.items[0].source_relative_path == "inbox/report.txt"
        assert restored.items[0].target_relative_path == "reports/report.txt"
        assert restored.items[0].source_sha256 == "a" * 64
        assert restored.items[0].reason_match_score == 95
        assert restored.items[0].status == "PENDING"

        restored_contract = get_operation_plan(restarted_session, PLAN_ID)
        assert restored_contract is not None
        assert restored_contract.plan_id == UUID(PLAN_ID)
        assert restored_contract.workspace_id == workspace_id
        assert restored_contract.operations[0].target_relative_path == (
            "reports/report.txt"
        )
        assert restored_contract.operations[0].source_precondition.content_hash is not None


def test_corrupt_persisted_plan_fails_closed(engine: Engine) -> None:
    with Session(engine) as session:
        workspace = Workspace(
            name="损坏数据测试工作区",
            root_path="C:/filenest-operation-plan-corrupt",
        )
        session.add(workspace)
        session.flush()

        plan = _plan()
        plan.workspace_id = workspace.id
        plan.status = "APPROVED"
        session.add(plan)
        session.add(
            ApprovalRequest(
                workflow_id=WORKFLOW_ID,
                plan_id=PLAN_ID,
                status="APPROVED",
            )
        )
        session.commit()

        valid_contract = get_operation_plan(session, PLAN_ID)
        assert valid_contract is not None
        persisted = session.get(OperationPlanRecord, PLAN_ID)
        assert persisted is not None
        persisted.items[0].risks_json = '{"unexpected":"object"}'
        session.commit()

    with Session(engine) as restarted_session:
        with pytest.raises(OperationPlanPersistenceError):
            get_operation_plan(restarted_session, PLAN_ID)
        with pytest.raises(OperationPlanApprovalError) as error:
            require_approved_operation_plan(
                restarted_session,
                UUID(WORKFLOW_ID),
                valid_contract,
            )
        assert error.value.code == OperationPlanApprovalErrorCode.PLAN_MISMATCH


def test_approved_guard_rejects_missing_business_plan(engine: Engine) -> None:
    with Session(engine) as session:
        workspace = Workspace(
            name="审批守卫测试工作区",
            root_path="C:/filenest-operation-plan-guard",
        )
        session.add(workspace)
        session.flush()

        plan = _plan()
        plan.workspace_id = workspace.id
        plan.status = "APPROVED"
        session.add(plan)
        session.add(
            ApprovalRequest(
                workflow_id=WORKFLOW_ID,
                plan_id=PLAN_ID,
                status="APPROVED",
            )
        )
        session.commit()

        valid_contract = get_operation_plan(session, PLAN_ID)
        assert valid_contract is not None
        session.delete(session.get(OperationPlanRecord, PLAN_ID))
        session.commit()

        with pytest.raises(OperationPlanApprovalError) as error:
            require_approved_operation_plan(
                session,
                UUID(WORKFLOW_ID),
                valid_contract,
            )

    assert error.value.code == OperationPlanApprovalErrorCode.PLAN_MISMATCH


def test_service_lists_items_and_history_after_session_restart(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        workspace = Workspace(
            name="计划历史测试工作区",
            root_path="C:/filenest-operation-plan-history",
        )
        session.add(workspace)
        session.flush()

        first_plan = _plan()
        first_plan.workspace_id = workspace.id
        second_plan = _plan()
        second_plan.plan_id = HISTORY_PLAN_ID
        second_plan.workspace_id = workspace.id
        second_plan.parent_plan_id = PLAN_ID
        second_plan.created_at = datetime(
            2026,
            9,
            2,
            10,
            5,
            tzinfo=timezone.utc,
        )
        session.add_all([first_plan, second_plan])
        session.commit()

    with Session(engine) as restarted_session:
        items = list_operation_plan_items(restarted_session, PLAN_ID)
        history = list_operation_plan_history(restarted_session, WORKFLOW_ID)

    assert [item.target_relative_path for item in items] == [
        "reports/report.txt"
    ]
    assert [plan.plan_id for plan in history] == [
        UUID(PLAN_ID),
        UUID(HISTORY_PLAN_ID),
    ]
    assert history[1].operations[0].source_relative_path == "inbox/report.txt"


def test_plan_schema_has_relationship_indexes_and_constraints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "operation-plan-migration.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("FILENEST_DATABASE_URL", database_url)

    alembic_config = Config(str(ALEMBIC_CONFIG_PATH))
    command.upgrade(alembic_config, "head")

    engine = create_engine(database_url)
    try:
        schema = inspect(engine)
        assert {"operation_plans", "operation_items"}.issubset(
            schema.get_table_names()
        )
        assert schema.get_foreign_keys("operation_plans")
        assert schema.get_foreign_keys("operation_items") == [
            {
                "name": None,
                "constrained_columns": ["plan_id"],
                "referred_schema": None,
                "referred_table": "operation_plans",
                "referred_columns": ["plan_id"],
                "options": {"ondelete": "CASCADE"},
            }
        ]
        assert schema.get_unique_constraints("operation_items") == [
            {
                "name": "uq_operation_items_plan_sequence",
                "column_names": ["plan_id", "sequence_no"],
            }
        ]
        assert {
            index["name"] for index in schema.get_indexes("operation_plans")
        } == {
            "ix_operation_plans_workspace_id",
            "ix_operation_plans_workflow_id",
            "ix_operation_plans_parent_plan_id",
        }
        assert {
            index["name"] for index in schema.get_indexes("operation_items")
        } == {"ix_operation_items_plan_id"}
    finally:
        engine.dispose()

    command.downgrade(alembic_config, "f36a01b2c3d4")
    downgraded_engine = create_engine(database_url)
    try:
        tables = inspect(downgraded_engine).get_table_names()
        assert "operation_plans" not in tables
        assert "operation_items" not in tables
    finally:
        downgraded_engine.dispose()


def test_item_constraints_reject_invalid_checksum_and_duplicate_sequence(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        workspace = Workspace(
            name="约束测试工作区",
            root_path="C:/filenest-operation-plan-constraints",
        )
        session.add(workspace)
        session.flush()

        plan = _plan()
        plan.workspace_id = workspace.id
        session.add(plan)
        session.commit()

        invalid_checksum = OperationItemRecord(
            plan_id=PLAN_ID,
            sequence_no=1,
            operation_type="move",
            source_file_id=8,
            source_relative_path="inbox/other.txt",
            target_relative_path="reports/other.txt",
            source_size_bytes=1,
            source_mtime_ns=1,
            source_hash_algorithm="sha256",
            source_sha256="b" * 63,
            reason_kind="manual_selection",
            reason_description="用户确认目标目录",
            risks_json="[]",
        )
        session.add(invalid_checksum)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

        duplicate = OperationItemRecord(
            plan_id=PLAN_ID,
            sequence_no=1,
            operation_type="move",
            source_file_id=8,
            source_relative_path="inbox/other.txt",
            target_relative_path="reports/other.txt",
            source_size_bytes=1,
            source_mtime_ns=1,
            source_hash_algorithm="sha256",
            source_sha256="b" * 64,
            reason_kind="manual_selection",
            reason_description="用户确认目标目录",
            risks_json="[]",
        )
        session.add(duplicate)
        with pytest.raises(IntegrityError):
            session.commit()
