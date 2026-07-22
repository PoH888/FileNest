from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_CONFIG_PATH = PROJECT_ROOT / "backend" / "alembic.ini"


def test_migrations_build_schema_and_downgrade_each_latest_layer(
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

        assert "agent_runs" in schema.get_table_names()
        assert [
            column["name"] for column in schema.get_columns("agent_runs")
        ] == [
            "id",
            "status",
            "started_at",
            "finished_at",
            "model_turns",
            "error_code",
        ]
        assert {
            constraint["name"]
            for constraint in schema.get_check_constraints("agent_runs")
        } == {
            "ck_agent_runs_status",
            "ck_agent_runs_model_turns_non_negative",
        }

        assert "agent_tool_calls" in schema.get_table_names()
        assert [
            column["name"]
            for column in schema.get_columns("agent_tool_calls")
        ] == [
            "id",
            "agent_run_id",
            "sequence_no",
            "model_call_id",
            "tool_name",
            "status",
            "started_at",
            "finished_at",
            "error_code",
        ]
        assert {
            constraint["name"]
            for constraint in schema.get_check_constraints("agent_tool_calls")
        } == {
            "ck_agent_tool_calls_sequence_positive",
            "ck_agent_tool_calls_status",
        }
        assert {
            constraint["name"]: constraint["column_names"]
            for constraint in schema.get_unique_constraints("agent_tool_calls")
        } == {
            "uq_agent_tool_calls_run_model_call_id": [
                "agent_run_id",
                "model_call_id",
            ],
            "uq_agent_tool_calls_run_sequence": [
                "agent_run_id",
                "sequence_no",
            ],
        }

        agent_tool_foreign_keys = schema.get_foreign_keys("agent_tool_calls")
        assert len(agent_tool_foreign_keys) == 1
        assert agent_tool_foreign_keys[0]["constrained_columns"] == [
            "agent_run_id"
        ]
        assert agent_tool_foreign_keys[0]["referred_table"] == "agent_runs"
        assert agent_tool_foreign_keys[0]["referred_columns"] == ["id"]
    finally:
        engine.dispose()

    command.downgrade(alembic_config, "8b872f337530")

    previous_head_engine = create_engine(database_url)
    try:
        previous_head_schema = inspect(previous_head_engine)

        assert "workspaces" in previous_head_schema.get_table_names()
        assert "file_entries" in previous_head_schema.get_table_names()
        assert "agent_runs" not in previous_head_schema.get_table_names()
        assert "agent_tool_calls" not in previous_head_schema.get_table_names()
    finally:
        previous_head_engine.dispose()

    command.downgrade(alembic_config, "4eb613c09cae")

    downgraded_engine = create_engine(database_url)
    try:
        downgraded_schema = inspect(downgraded_engine)

        assert "workspaces" in downgraded_schema.get_table_names()
        assert "file_entries" not in downgraded_schema.get_table_names()
        assert "agent_runs" not in downgraded_schema.get_table_names()
        assert "agent_tool_calls" not in downgraded_schema.get_table_names()
    finally:
        downgraded_engine.dispose()
