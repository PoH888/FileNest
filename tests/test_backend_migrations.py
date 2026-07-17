from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_CONFIG_PATH = PROJECT_ROOT / "backend" / "alembic.ini"


def test_migrations_build_schema_from_empty_database_and_downgrade_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "migration-test.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("FILENEST_DATABASE_URL", database_url)

    alembic_config = Config(str(ALEMBIC_CONFIG_PATH))
    command.upgrade(alembic_config, "head")

    engine = create_engine(database_url)
    try:
        schema = inspect(engine)

        assert "workspaces" in schema.get_table_names()
        assert [
            column["name"] for column in schema.get_columns("workspaces")
        ] == ["id", "name", "root_path"]
        assert schema.get_unique_constraints("workspaces") == [
            {"name": None, "column_names": ["root_path"]}
        ]

        assert "file_entries" in schema.get_table_names()
        assert [
            column["name"] for column in schema.get_columns("file_entries")
        ] == [
            "id",
            "workspace_id",
            "relative_path",
            "name",
            "extension",
            "size_bytes",
            "mtime_ns",
        ]
        assert all(
            not column["nullable"]
            for column in schema.get_columns("file_entries")
        )
        assert schema.get_unique_constraints("file_entries") == [
            {
                "name": "uq_file_entries_workspace_relative_path",
                "column_names": ["workspace_id", "relative_path"],
            }
        ]

        foreign_keys = schema.get_foreign_keys("file_entries")
        assert len(foreign_keys) == 1
        assert foreign_keys[0]["constrained_columns"] == ["workspace_id"]
        assert foreign_keys[0]["referred_table"] == "workspaces"
        assert foreign_keys[0]["referred_columns"] == ["id"]
    finally:
        engine.dispose()

    command.downgrade(alembic_config, "4eb613c09cae")

    downgraded_engine = create_engine(database_url)
    try:
        downgraded_schema = inspect(downgraded_engine)

        assert "workspaces" in downgraded_schema.get_table_names()
        assert "file_entries" not in downgraded_schema.get_table_names()
    finally:
        downgraded_engine.dispose()
