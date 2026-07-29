from datetime import datetime, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.models import (
    ApprovalAuditEvent,
    ApprovalRequest,
    OperationExecution,
    OperationExecutionItem,
    Workspace,
)


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

        assert "operation_executions" in schema.get_table_names()
        assert [
            column["name"]
            for column in schema.get_columns("operation_executions")
        ] == [
            "id",
            "workflow_id",
            "plan_id",
            "workspace_id",
            "status",
            "started_at",
            "completed_at",
            "undone_at",
            "attempt",
        ]
        assert {
            constraint["name"]: constraint["column_names"]
            for constraint in schema.get_unique_constraints(
                "operation_executions"
            )
        } == {
            "uq_operation_executions_plan_id": ["plan_id"],
            "uq_operation_executions_workflow_id": ["workflow_id"],
        }
        assert {
            constraint["name"]
            for constraint in schema.get_check_constraints(
                "operation_executions"
            )
        } == {
            "ck_operation_executions_attempt_positive",
            "ck_operation_executions_status",
        }
        execution_foreign_keys = schema.get_foreign_keys(
            "operation_executions"
        )
        assert len(execution_foreign_keys) == 1
        assert execution_foreign_keys[0]["constrained_columns"] == [
            "workspace_id"
        ]
        assert execution_foreign_keys[0]["referred_table"] == "workspaces"
        assert execution_foreign_keys[0]["referred_columns"] == ["id"]
        assert schema.get_indexes("operation_executions") == [
            {
                "name": "ix_operation_executions_workspace_id",
                "column_names": ["workspace_id"],
                "unique": 0,
                "dialect_options": {},
            }
        ]

        assert "operation_execution_items" in schema.get_table_names()
        assert [
            column["name"]
            for column in schema.get_columns("operation_execution_items")
        ] == [
            "id",
            "execution_id",
            "sequence_no",
            "operation_type",
            "source_file_id",
            "before_location",
            "before_relative_path",
            "before_size_bytes",
            "before_mtime_ns",
            "before_sha256",
            "after_location",
            "after_relative_path",
            "after_size_bytes",
            "after_mtime_ns",
            "after_sha256",
            "undo_source_relative_path",
            "undo_target_relative_path",
            "status",
            "recorded_at",
            "completed_at",
            "undone_at",
            "error_code",
            "failed_at",
        ]
        assert {
            constraint["name"]
            for constraint in schema.get_check_constraints(
                "operation_execution_items"
            )
        } == {
            "ck_operation_execution_items_after_location",
            "ck_operation_execution_items_after_metadata",
            "ck_operation_execution_items_before_location",
            "ck_operation_execution_items_before_metadata",
            "ck_operation_execution_items_sequence_positive",
            "ck_operation_execution_items_status",
            "ck_operation_execution_items_type",
        }
        assert schema.get_unique_constraints(
            "operation_execution_items"
        ) == [
            {
                "name": "uq_operation_execution_items_execution_sequence",
                "column_names": ["execution_id", "sequence_no"],
            }
        ]
        execution_item_foreign_keys = schema.get_foreign_keys(
            "operation_execution_items"
        )
        assert len(execution_item_foreign_keys) == 1
        assert execution_item_foreign_keys[0]["constrained_columns"] == [
            "execution_id"
        ]
        assert execution_item_foreign_keys[0]["referred_table"] == (
            "operation_executions"
        )
        assert execution_item_foreign_keys[0]["referred_columns"] == ["id"]
        assert schema.get_indexes("operation_execution_items") == [
            {
                "name": "ix_operation_execution_items_execution_id",
                "column_names": ["execution_id"],
                "unique": 0,
                "dialect_options": {},
            }
        ]

        with Session(engine) as session:
            workspace = Workspace(
                name="迁移测试工作区",
                root_path=str(tmp_path / "workspace"),
            )
            session.add(workspace)
            session.commit()

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

            execution = OperationExecution(
                workflow_id=approval.workflow_id,
                plan_id=approval.plan_id,
                workspace_id=workspace.id,
            )
            session.add(execution)
            session.commit()
            execution_id = execution.id

            assert execution.idempotency_key == approval.plan_id
            assert execution.attempt == 1

            partial_execution = OperationExecution(
                workflow_id="8933c981-fe44-4d3f-a4e0-3d7ed66be0ca",
                plan_id="37cb1621-44db-49cd-9251-31c7e871e34d",
                workspace_id=workspace.id,
                status="PARTIALLY_COMPLETED",
            )
            session.add(partial_execution)
            session.commit()

            assert partial_execution.attempt == 1

            with pytest.raises(IntegrityError):
                session.add(
                    OperationExecution(
                        workflow_id="f3ce116c-118c-48cb-ac15-ad189e32ace4",
                        plan_id="5c11356a-1cb5-466c-be44-7d774ba4390c",
                        workspace_id=workspace.id,
                        attempt=0,
                    )
                )
                session.commit()
            session.rollback()

            execution_item = OperationExecutionItem(
                execution_id=execution.id,
                sequence_no=1,
                operation_type="move",
                source_file_id=7,
                before_location="workspace",
                before_relative_path="inbox/report.pdf",
                before_size_bytes=15,
                before_mtime_ns=123456,
                after_location="workspace",
                after_relative_path="documents/report.pdf",
                undo_source_relative_path="documents/report.pdf",
                undo_target_relative_path="inbox/report.pdf",
            )
            session.add(execution_item)
            session.commit()
            execution_item_id = execution_item.id

            assert execution_item.error_code is None
            assert execution_item.failed_at is None

            failure_time = datetime(
                2026,
                8,
                31,
                10,
                30,
                tzinfo=timezone.utc,
            )
            execution_item.status = "FAILED"
            execution_item.error_code = "safe_move_target_conflict"
            execution_item.failed_at = failure_time
            session.commit()

            with pytest.raises(IntegrityError):
                session.add(
                    OperationExecution(
                        workflow_id=execution.workflow_id,
                        plan_id="8933c981-fe44-4d3f-a4e0-3d7ed66be0ca",
                        workspace_id=workspace.id,
                    )
                )
                session.commit()
            session.rollback()

            with pytest.raises(IntegrityError):
                session.add(
                    OperationExecution(
                        workflow_id="6795b2e2-6f96-47ce-9ff8-7535845435b1",
                        plan_id="ae4bd5b0-bf54-4c91-8a64-03da759255ad",
                        workspace_id=workspace.id,
                        status="EXECUTED_WITHOUT_HISTORY",
                    )
                )
                session.commit()
            session.rollback()

            with pytest.raises(IntegrityError):
                session.add(
                    OperationExecutionItem(
                        execution_id=execution.id,
                        sequence_no=2,
                        operation_type="overwrite",
                        source_file_id=8,
                        before_location="workspace",
                        before_relative_path="inbox/unsafe.txt",
                        before_size_bytes=1,
                        before_mtime_ns=1,
                        after_location="workspace",
                        after_relative_path="documents/unsafe.txt",
                        undo_source_relative_path="documents/unsafe.txt",
                        undo_target_relative_path="inbox/unsafe.txt",
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

            restored_execution = session.get(
                OperationExecution,
                execution_id,
            )
            assert restored_execution is not None
            assert restored_execution.status == "EXECUTING"
            assert restored_execution.attempt == 1
            assert restored_execution.idempotency_key == (
                "2d053752-d3c4-45cb-b696-bd043e78ed92"
            )
            assert restored_execution.completed_at is None
            assert restored_execution.undone_at is None

            restored_execution_item = session.get(
                OperationExecutionItem,
                execution_item_id,
            )
            assert restored_execution_item is not None
            assert restored_execution_item.status == "FAILED"
            assert restored_execution_item.error_code == (
                "safe_move_target_conflict"
            )
            assert restored_execution_item.failed_at == failure_time.replace(
                tzinfo=None
            )
            assert restored_execution_item.before_relative_path == (
                "inbox/report.pdf"
            )
            assert restored_execution_item.after_relative_path == (
                "documents/report.pdf"
            )
            assert restored_execution_item.undo_source_relative_path == (
                "documents/report.pdf"
            )
            assert restored_execution_item.undo_target_relative_path == (
                "inbox/report.pdf"
            )
    finally:
        reopened_engine.dispose()

    command.downgrade(alembic_config, "a25e01a7c4d1")

    previous_failure_evidence_engine = create_engine(database_url)
    try:
        previous_failure_schema = inspect(previous_failure_evidence_engine)
        assert "error_code" not in {
            column["name"]
            for column in previous_failure_schema.get_columns(
                "operation_execution_items"
            )
        }
        assert "failed_at" not in {
            column["name"]
            for column in previous_failure_schema.get_columns(
                "operation_execution_items"
            )
        }

        with previous_failure_evidence_engine.connect() as connection:
            preserved_item = connection.execute(
                text(
                    "SELECT id, status FROM operation_execution_items "
                    "WHERE id = :execution_item_id"
                ),
                {"execution_item_id": execution_item_id},
            ).mappings().one()
            assert dict(preserved_item) == {
                "id": execution_item_id,
                "status": "FAILED",
            }
    finally:
        previous_failure_evidence_engine.dispose()

    command.downgrade(alembic_config, "e23a01c7d4f2")

    previous_execution_engine = create_engine(database_url)
    try:
        previous_execution_schema = inspect(previous_execution_engine)

        assert "operation_executions" not in (
            previous_execution_schema.get_table_names()
        )
        assert "operation_execution_items" not in (
            previous_execution_schema.get_table_names()
        )
        assert "approval_requests" in (
            previous_execution_schema.get_table_names()
        )
        assert "approval_audit_events" in (
            previous_execution_schema.get_table_names()
        )
    finally:
        previous_execution_engine.dispose()

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


def test_execution_attempt_migration_preserves_existing_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "execution-attempt-migration.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("FILENEST_DATABASE_URL", database_url)

    alembic_config = Config(str(ALEMBIC_CONFIG_PATH))
    command.upgrade(alembic_config, "f24e05a1b2c3")

    old_engine = create_engine(database_url)
    try:
        with old_engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO workspaces (name, root_path) "
                    "VALUES (:name, :root_path)"
                ),
                {
                    "name": "旧执行记录工作区",
                    "root_path": str(tmp_path / "legacy-workspace"),
                },
            )
            workspace_id = connection.execute(
                text("SELECT id FROM workspaces")
            ).scalar_one()
            execution_insert = connection.execute(
                text(
                    "INSERT INTO operation_executions "
                    "(workflow_id, plan_id, workspace_id, status) "
                    "VALUES "
                    "(:workflow_id, :plan_id, :workspace_id, 'EXECUTING')"
                ),
                {
                    "workflow_id": "428981e6-97e3-4fd7-9460-38b366061490",
                    "plan_id": "b6f45f30-55b0-491c-b243-cf57de3ef774",
                    "workspace_id": workspace_id,
                },
            )
            execution_id = execution_insert.lastrowid
            connection.execute(
                text(
                    "INSERT INTO operation_execution_items "
                    "(execution_id, sequence_no, operation_type, "
                    "source_file_id, before_location, "
                    "before_relative_path, before_size_bytes, "
                    "before_mtime_ns, after_location, "
                    "after_relative_path, undo_source_relative_path, "
                    "undo_target_relative_path, status) "
                    "VALUES (:execution_id, 1, 'move', 7, 'workspace', "
                    "'inbox/legacy.txt', 10, 100, 'workspace', "
                    "'archive/legacy.txt', 'archive/legacy.txt', "
                    "'inbox/legacy.txt', 'PENDING')"
                ),
                {"execution_id": execution_id},
            )
    finally:
        old_engine.dispose()

    command.upgrade(alembic_config, "head")

    upgraded_engine = create_engine(database_url)
    try:
        upgraded_schema = inspect(upgraded_engine)
        assert "attempt" in {
            column["name"]
            for column in upgraded_schema.get_columns("operation_executions")
        }

        with upgraded_engine.begin() as connection:
            restored = connection.execute(
                text(
                    "SELECT plan_id, status, attempt "
                    "FROM operation_executions"
                )
            ).mappings().one()

            assert dict(restored) == {
                "plan_id": "b6f45f30-55b0-491c-b243-cf57de3ef774",
                "status": "EXECUTING",
                "attempt": 1,
            }

            restored_item = connection.execute(
                text(
                    "SELECT status, error_code, failed_at "
                    "FROM operation_execution_items"
                )
            ).mappings().one()
            assert dict(restored_item) == {
                "status": "PENDING",
                "error_code": None,
                "failed_at": None,
            }

            connection.execute(
                text(
                    "UPDATE operation_executions "
                    "SET status = 'PARTIALLY_COMPLETED'"
                )
            )
            connection.execute(
                text(
                    "UPDATE operation_execution_items "
                    "SET status = 'FAILED', "
                    "error_code = 'safe_move_source_unavailable', "
                    "failed_at = '2026-08-31 10:30:00'"
                )
            )
    finally:
        upgraded_engine.dispose()

    command.downgrade(alembic_config, "f24e05a1b2c3")

    downgraded_engine = create_engine(database_url)
    try:
        downgraded_schema = inspect(downgraded_engine)
        assert "attempt" not in {
            column["name"]
            for column in downgraded_schema.get_columns(
                "operation_executions"
            )
        }

        with downgraded_engine.connect() as connection:
            restored = connection.execute(
                text(
                    "SELECT plan_id, status "
                    "FROM operation_executions"
                )
            ).mappings().one()

            assert dict(restored) == {
                "plan_id": "b6f45f30-55b0-491c-b243-cf57de3ef774",
                "status": "FAILED",
            }

            restored_item = connection.execute(
                text(
                    "SELECT status FROM operation_execution_items"
                )
            ).mappings().one()
            assert dict(restored_item) == {"status": "FAILED"}
    finally:
        downgraded_engine.dispose()
