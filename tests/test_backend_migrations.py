from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_CONFIG_PATH = PROJECT_ROOT / "backend" / "alembic.ini"


def test_migration_builds_workspace_schema_from_empty_database(
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
    finally:
        engine.dispose()
