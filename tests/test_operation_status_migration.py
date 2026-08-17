from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_CONFIG_PATH = PROJECT_ROOT / "backend" / "alembic.ini"


def test_operation_status_migration_creates_independent_projection_table(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "operation-status-migration.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("FILENEST_DATABASE_URL", database_url)

    command.upgrade(Config(str(ALEMBIC_CONFIG_PATH)), "head")

    engine = create_engine(database_url)
    try:
        schema = inspect(engine)

        assert "operation_statuses" in schema.get_table_names()
        assert [
            column["name"]
            for column in schema.get_columns("operation_statuses")
        ] == [
            "workflow_id",
            "plan_id",
            "approval_id",
            "execution_id",
            "overall_status",
            "revision",
            "created_at",
            "updated_at",
        ]
        assert {
            constraint["name"]
            for constraint in schema.get_check_constraints("operation_statuses")
        } == {
            "ck_operation_statuses_overall_status",
            "ck_operation_statuses_revision_non_negative",
        }
        foreign_keys = {
            tuple(key["constrained_columns"]): key["referred_table"]
            for key in schema.get_foreign_keys("operation_statuses")
        }
        assert foreign_keys == {
            ("plan_id",): "operation_plans",
            ("approval_id",): "approval_requests",
            ("execution_id",): "operation_executions",
        }
        for table_name in (
            "operation_plans",
            "approval_requests",
            "approval_audit_events",
        ):
            checks = schema.get_check_constraints(table_name)
            assert any(
                "CANCELLED" in (constraint["sqltext"] or "")
                for constraint in checks
            )
    finally:
        engine.dispose()

    command.downgrade(
        Config(str(ALEMBIC_CONFIG_PATH)),
        "g06a01b2c3d4",
    )

    downgraded_engine = create_engine(database_url)
    try:
        downgraded_tables = inspect(downgraded_engine).get_table_names()
        assert "operation_statuses" not in downgraded_tables
        assert "operation_plans" in downgraded_tables
    finally:
        downgraded_engine.dispose()
