from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.models import ApprovalAuditEvent, ApprovalRequest


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

        assert "approval_requests" in schema.get_table_names()
        assert [
            column["name"]
            for column in schema.get_columns("approval_requests")
        ] == [
            "id",
            "workflow_id",
            "plan_id",
            "status",
            "created_at",
        ]
        assert schema.get_unique_constraints("approval_requests") == [
            {
                "name": "uq_approval_requests_workflow_id",
                "column_names": ["workflow_id"],
            }
        ]
        assert {
            constraint["name"]
            for constraint in schema.get_check_constraints(
                "approval_requests"
            )
        } == {"ck_approval_requests_status"}

        assert "approval_audit_events" in schema.get_table_names()
        assert [
            column["name"]
            for column in schema.get_columns("approval_audit_events")
        ] == [
            "id",
            "approval_request_id",
            "action",
            "previous_status",
            "next_status",
            "previous_plan_id",
            "next_plan_id",
            "recorded_at",
        ]
        assert {
            constraint["name"]
            for constraint in schema.get_check_constraints(
                "approval_audit_events"
            )
        } == {
            "ck_approval_audit_events_action",
            "ck_approval_audit_events_previous_status",
            "ck_approval_audit_events_next_status",
        }
        approval_audit_foreign_keys = schema.get_foreign_keys(
            "approval_audit_events"
        )
        assert len(approval_audit_foreign_keys) == 1
        assert approval_audit_foreign_keys[0]["constrained_columns"] == [
            "approval_request_id"
        ]
        assert approval_audit_foreign_keys[0]["referred_table"] == (
            "approval_requests"
        )
        assert approval_audit_foreign_keys[0]["referred_columns"] == ["id"]
        assert schema.get_indexes("approval_audit_events") == [
            {
                "name": "ix_approval_audit_events_approval_request_id",
                "column_names": ["approval_request_id"],
                "unique": 0,
                "dialect_options": {},
            }
        ]

        with Session(engine) as session:
            approval = ApprovalRequest(
                workflow_id="66c8d4ba-a042-4491-a5d2-ad28cb47b8d9",
                plan_id="2d053752-d3c4-45cb-b696-bd043e78ed92",
            )
            session.add(approval)
            session.commit()
            approval_id = approval.id

            with pytest.raises(IntegrityError):
                session.add(
                    ApprovalRequest(
                        workflow_id=approval.workflow_id,
                        plan_id="37cb1621-44db-49cd-9251-31c7e871e34d",
                    )
                )
                session.commit()
            session.rollback()

            audit_event = ApprovalAuditEvent(
                approval_request_id=approval_id,
                action="approve",
                previous_status="WAITING_APPROVAL",
                next_status="APPROVED",
                previous_plan_id=approval.plan_id,
                next_plan_id=approval.plan_id,
            )
            session.add(audit_event)
            session.commit()
            audit_event_id = audit_event.id

            with pytest.raises(IntegrityError):
                session.add(
                    ApprovalAuditEvent(
                        approval_request_id=approval_id,
                        action="execute_without_approval",
                        previous_status="WAITING_APPROVAL",
                        next_status="APPROVED",
                        previous_plan_id=approval.plan_id,
                        next_plan_id=approval.plan_id,
                    )
                )
                session.commit()
            session.rollback()

            with pytest.raises(IntegrityError):
                session.add(
                    ApprovalRequest(
                        workflow_id="8933c981-fe44-4d3f-a4e0-3d7ed66be0ca",
                        plan_id="37cb1621-44db-49cd-9251-31c7e871e34d",
                        status="EXECUTING",
                    )
                )
                session.commit()
            session.rollback()
    finally:
        engine.dispose()

    reopened_engine = create_engine(database_url)
    try:
        with Session(reopened_engine) as session:
            restored = session.get(ApprovalRequest, approval_id)

            assert restored is not None
            assert restored.status == "WAITING_APPROVAL"
            assert restored.workflow_id == (
                "66c8d4ba-a042-4491-a5d2-ad28cb47b8d9"
            )
            assert restored.plan_id == (
                "2d053752-d3c4-45cb-b696-bd043e78ed92"
            )

            restored_event = session.get(
                ApprovalAuditEvent,
                audit_event_id,
            )
            assert restored_event is not None
            assert restored_event.approval_request_id == approval_id
            assert restored_event.action == "approve"
            assert restored_event.previous_status == "WAITING_APPROVAL"
            assert restored_event.next_status == "APPROVED"
            assert restored_event.recorded_at is not None
    finally:
        reopened_engine.dispose()

    command.downgrade(alembic_config, "c3f4a1b92d6e")

    previous_approval_engine = create_engine(database_url)
    try:
        previous_approval_schema = inspect(previous_approval_engine)

        assert "approval_requests" not in (
            previous_approval_schema.get_table_names()
        )
        assert "approval_audit_events" not in (
            previous_approval_schema.get_table_names()
        )
        assert "agent_runs" in previous_approval_schema.get_table_names()
        assert "agent_tool_calls" in (
            previous_approval_schema.get_table_names()
        )
    finally:
        previous_approval_engine.dispose()

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
