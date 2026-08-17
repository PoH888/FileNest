"""create independent operation status table

Revision ID: h07a01b2c3d4
Revises: g06a01b2c3d4
Create Date: 2026-09-02

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "h07a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "g06a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Persist the current Operation status independently from checkpoints."""

    op.create_table(
        "operation_statuses",
        sa.Column("workflow_id", sa.String(length=36), nullable=False),
        sa.Column("plan_id", sa.String(length=36), nullable=False),
        sa.Column("approval_id", sa.Integer(), nullable=True),
        sa.Column("execution_id", sa.Integer(), nullable=True),
        sa.Column(
            "overall_status",
            sa.String(length=32),
            server_default="PROPOSED",
            nullable=False,
        ),
        sa.Column("revision", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "overall_status IN "
            "('PROPOSED', 'WAITING_APPROVAL', 'APPROVED', 'REJECTED', "
            "'CANCELLED', 'EXECUTING', 'PARTIAL_FAILED', 'COMPLETED', "
            "'UNDOING', 'UNDONE', 'COMPENSATED', 'FAILED')",
            name="ck_operation_statuses_overall_status",
        ),
        sa.CheckConstraint(
            "revision >= 0",
            name="ck_operation_statuses_revision_non_negative",
        ),
        sa.ForeignKeyConstraint(
            ["plan_id"],
            ["operation_plans.plan_id"],
        ),
        sa.ForeignKeyConstraint(
            ["approval_id"],
            ["approval_requests.id"],
        ),
        sa.ForeignKeyConstraint(
            ["execution_id"],
            ["operation_executions.id"],
        ),
        sa.PrimaryKeyConstraint("workflow_id"),
    )
    op.create_index(
        "ix_operation_statuses_plan_id",
        "operation_statuses",
        ["plan_id"],
        unique=False,
    )
    op.create_index(
        "ix_operation_statuses_approval_id",
        "operation_statuses",
        ["approval_id"],
        unique=False,
    )
    op.create_index(
        "ix_operation_statuses_execution_id",
        "operation_statuses",
        ["execution_id"],
        unique=False,
    )


def downgrade() -> None:
    """Remove the current Operation status without touching source records."""

    op.drop_index(
        "ix_operation_statuses_execution_id",
        table_name="operation_statuses",
    )
    op.drop_index(
        "ix_operation_statuses_approval_id",
        table_name="operation_statuses",
    )
    op.drop_index(
        "ix_operation_statuses_plan_id",
        table_name="operation_statuses",
    )
    op.drop_table("operation_statuses")
