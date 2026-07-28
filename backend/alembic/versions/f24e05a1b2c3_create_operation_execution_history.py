"""create operation execution history

Revision ID: f24e05a1b2c3
Revises: e23a01c7d4f2
Create Date: 2026-08-31

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f24e05a1b2c3"
down_revision: Union[str, Sequence[str], None] = "e23a01c7d4f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create durable execution state and per-file undo evidence."""

    op.create_table(
        "operation_executions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("workflow_id", sa.String(length=36), nullable=False),
        sa.Column("plan_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.String(),
            server_default="EXECUTING",
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "undone_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.CheckConstraint(
            "status IN "
            "('EXECUTING', 'COMPLETED', 'UNDOING', 'UNDONE', 'FAILED')",
            name="ck_operation_executions_status",
        ),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "plan_id",
            name="uq_operation_executions_plan_id",
        ),
        sa.UniqueConstraint(
            "workflow_id",
            name="uq_operation_executions_workflow_id",
        ),
    )
    op.create_index(
        "ix_operation_executions_workspace_id",
        "operation_executions",
        ["workspace_id"],
        unique=False,
    )

    op.create_table(
        "operation_execution_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("execution_id", sa.Integer(), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("operation_type", sa.String(), nullable=False),
        sa.Column("source_file_id", sa.Integer(), nullable=False),
        sa.Column("before_location", sa.String(), nullable=False),
        sa.Column("before_relative_path", sa.String(), nullable=False),
        sa.Column("before_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("before_mtime_ns", sa.BigInteger(), nullable=False),
        sa.Column("before_sha256", sa.String(length=64), nullable=True),
        sa.Column("after_location", sa.String(), nullable=False),
        sa.Column("after_relative_path", sa.String(), nullable=False),
        sa.Column("after_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("after_mtime_ns", sa.BigInteger(), nullable=True),
        sa.Column("after_sha256", sa.String(length=64), nullable=True),
        sa.Column("undo_source_relative_path", sa.String(), nullable=False),
        sa.Column("undo_target_relative_path", sa.String(), nullable=False),
        sa.Column(
            "status",
            sa.String(),
            server_default="PENDING",
            nullable=False,
        ),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "undone_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.CheckConstraint(
            "after_location IN ('workspace', 'quarantine')",
            name="ck_operation_execution_items_after_location",
        ),
        sa.CheckConstraint(
            "(after_size_bytes IS NULL OR after_size_bytes >= 0) AND "
            "(after_mtime_ns IS NULL OR after_mtime_ns >= 0)",
            name="ck_operation_execution_items_after_metadata",
        ),
        sa.CheckConstraint(
            "before_location IN ('workspace', 'quarantine')",
            name="ck_operation_execution_items_before_location",
        ),
        sa.CheckConstraint(
            "before_size_bytes >= 0 AND before_mtime_ns >= 0",
            name="ck_operation_execution_items_before_metadata",
        ),
        sa.CheckConstraint(
            "sequence_no >= 1",
            name="ck_operation_execution_items_sequence_positive",
        ),
        sa.CheckConstraint(
            "status IN ('PENDING', 'COMPLETED', 'UNDOING', 'UNDONE', "
            "'FAILED')",
            name="ck_operation_execution_items_status",
        ),
        sa.CheckConstraint(
            "operation_type IN ('move', 'quarantine')",
            name="ck_operation_execution_items_type",
        ),
        sa.ForeignKeyConstraint(
            ["execution_id"],
            ["operation_executions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "execution_id",
            "sequence_no",
            name="uq_operation_execution_items_execution_sequence",
        ),
    )
    op.create_index(
        "ix_operation_execution_items_execution_id",
        "operation_execution_items",
        ["execution_id"],
        unique=False,
    )


def downgrade() -> None:
    """Remove execution history while preserving approval state."""

    op.drop_index(
        "ix_operation_execution_items_execution_id",
        table_name="operation_execution_items",
    )
    op.drop_table("operation_execution_items")
    op.drop_index(
        "ix_operation_executions_workspace_id",
        table_name="operation_executions",
    )
    op.drop_table("operation_executions")
