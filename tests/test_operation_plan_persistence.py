from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import (
    OperationItemRecord,
    OperationPlanRecord,
    Workspace,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_CONFIG_PATH = PROJECT_ROOT / "backend" / "alembic.ini"
PLAN_ID = "a5d0c4af-142e-47d7-bb6d-2b7f8bb9cf20"
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


def test_plan_and_items_survive_session_restart(engine: Engine) -> None:
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
