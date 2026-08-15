"""create operation plan tables

Revision ID: g06a01b2c3d4
Revises: f36a01b2c3d4
Create Date: 2026-09-02

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "g06a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "f36a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Persist complete operation plans independently from checkpoints."""

    op.create_table(
        "operation_plans",
        sa.Column("plan_id", sa.String(length=36), nullable=False),
        sa.Column(
            "schema_version",
            sa.Integer(),
            server_default="1",
            nullable=False,
        ),
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("workflow_id", sa.String(length=36), nullable=False),
        sa.Column(
            "operation_type",
            sa.String(length=32),
            server_default="move",
            nullable=False,
        ),
        sa.Column(
            "metadata_json",
            sa.Text(),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default="WAITING_APPROVAL",
            nullable=False,
        ),
        sa.Column("parent_plan_id", sa.String(length=36), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "schema_version = 1",
            name="ck_operation_plans_schema_version",
        ),
        sa.CheckConstraint(
            "operation_type IN ('move')",
            name="ck_operation_plans_operation_type",
        ),
        sa.CheckConstraint(
            "status IN ('WAITING_APPROVAL', 'APPROVED', 'REJECTED', 'SUPERSEDED')",
            name="ck_operation_plans_status",
        ),
        sa.CheckConstraint(
            "parent_plan_id IS NULL OR parent_plan_id <> plan_id",
            name="ck_operation_plans_parent_not_self",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
        ),
        sa.ForeignKeyConstraint(
            ["parent_plan_id"],
            ["operation_plans.plan_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("plan_id"),
    )
    op.create_index(
        "ix_operation_plans_workspace_id",
        "operation_plans",
        ["workspace_id"],
        unique=False,
    )
    op.create_index(
        "ix_operation_plans_workflow_id",
        "operation_plans",
        ["workflow_id"],
        unique=False,
    )
    op.create_index(
        "ix_operation_plans_parent_plan_id",
        "operation_plans",
        ["parent_plan_id"],
        unique=False,
    )

    op.create_table(
        "operation_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("plan_id", sa.String(length=36), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column(
            "operation_type",
            sa.String(length=32),
            server_default="move",
            nullable=False,
        ),
        sa.Column("source_file_id", sa.Integer(), nullable=False),
        sa.Column("source_relative_path", sa.String(), nullable=False),
        sa.Column("target_relative_path", sa.String(), nullable=False),
        sa.Column("source_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("source_mtime_ns", sa.BigInteger(), nullable=False),
        sa.Column("source_hash_algorithm", sa.String(length=16), nullable=True),
        sa.Column("source_sha256", sa.String(length=64), nullable=True),
        sa.Column("reason_kind", sa.String(length=32), nullable=False),
        sa.Column("reason_description", sa.String(length=500), nullable=False),
        sa.Column("reason_match_score", sa.Integer(), nullable=True),
        sa.Column(
            "risks_json",
            sa.Text(),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default="PENDING",
            nullable=False,
        ),
        sa.CheckConstraint(
            "sequence_no >= 1",
            name="ck_operation_items_sequence_positive",
        ),
        sa.CheckConstraint(
            "operation_type IN ('move')",
            name="ck_operation_items_operation_type",
        ),
        sa.CheckConstraint(
            "status IN ('PENDING', 'COMPLETED', 'FAILED', 'UNDONE')",
            name="ck_operation_items_status",
        ),
        sa.CheckConstraint(
            "source_file_id >= 1",
            name="ck_operation_items_source_file_positive",
        ),
        sa.CheckConstraint(
            "source_size_bytes >= 0 AND source_mtime_ns >= 0",
            name="ck_operation_items_source_metadata",
        ),
        sa.CheckConstraint(
            "(source_hash_algorithm IS NULL AND source_sha256 IS NULL) OR "
            "(source_hash_algorithm = 'sha256' AND source_sha256 IS NOT NULL "
            "AND length(source_sha256) = 64)",
            name="ck_operation_items_source_hash_pair",
        ),
        sa.CheckConstraint(
            "reason_kind IN ('matched_candidate', 'manual_selection')",
            name="ck_operation_items_reason_kind",
        ),
        sa.CheckConstraint(
            "length(reason_description) BETWEEN 1 AND 500 "
            "AND reason_description = trim(reason_description)",
            name="ck_operation_items_reason_description",
        ),
        sa.CheckConstraint(
            "(reason_kind = 'matched_candidate' AND reason_match_score IS NOT NULL "
            "AND reason_match_score BETWEEN 0 AND 100) OR "
            "(reason_kind = 'manual_selection' AND reason_match_score IS NULL)",
            name="ck_operation_items_reason_score",
        ),
        sa.ForeignKeyConstraint(
            ["plan_id"],
            ["operation_plans.plan_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "plan_id",
            "sequence_no",
            name="uq_operation_items_plan_sequence",
        ),
    )
    op.create_index(
        "ix_operation_items_plan_id",
        "operation_items",
        ["plan_id"],
        unique=False,
    )


def downgrade() -> None:
    """Remove plan data while leaving workflow checkpoints untouched."""

    op.drop_index("ix_operation_items_plan_id", table_name="operation_items")
    op.drop_table("operation_items")
    op.drop_index(
        "ix_operation_plans_parent_plan_id",
        table_name="operation_plans",
    )
    op.drop_index("ix_operation_plans_workflow_id", table_name="operation_plans")
    op.drop_index("ix_operation_plans_workspace_id", table_name="operation_plans")
    op.drop_table("operation_plans")
