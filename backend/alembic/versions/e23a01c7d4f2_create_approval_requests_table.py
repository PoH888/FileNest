"""create approval requests table

Revision ID: e23a01c7d4f2
Revises: c3f4a1b92d6e
Create Date: 2026-08-31

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e23a01c7d4f2"
down_revision: Union[str, Sequence[str], None] = "c3f4a1b92d6e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create current approval state and append-only transition history."""

    op.create_table(
        "approval_requests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("workflow_id", sa.String(length=36), nullable=False),
        sa.Column("plan_id", sa.String(length=36), nullable=False),
        sa.Column(
            "status",
            sa.String(),
            server_default="WAITING_APPROVAL",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('WAITING_APPROVAL', 'APPROVED', 'REJECTED')",
            name="ck_approval_requests_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "workflow_id",
            name="uq_approval_requests_workflow_id",
        ),
    )

    op.create_table(
        "approval_audit_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("approval_request_id", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("previous_status", sa.String(), nullable=False),
        sa.Column("next_status", sa.String(), nullable=False),
        sa.Column("previous_plan_id", sa.String(length=36), nullable=False),
        sa.Column("next_plan_id", sa.String(length=36), nullable=False),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "action IN ('approve', 'edit', 'reject')",
            name="ck_approval_audit_events_action",
        ),
        sa.CheckConstraint(
            "previous_status IN "
            "('WAITING_APPROVAL', 'APPROVED', 'REJECTED')",
            name="ck_approval_audit_events_previous_status",
        ),
        sa.CheckConstraint(
            "next_status IN "
            "('WAITING_APPROVAL', 'APPROVED', 'REJECTED')",
            name="ck_approval_audit_events_next_status",
        ),
        sa.ForeignKeyConstraint(
            ["approval_request_id"],
            ["approval_requests.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_approval_audit_events_approval_request_id",
        "approval_audit_events",
        ["approval_request_id"],
        unique=False,
    )


def downgrade() -> None:
    """Remove approval business state without touching checkpoints."""

    op.drop_index(
        "ix_approval_audit_events_approval_request_id",
        table_name="approval_audit_events",
    )
    op.drop_table("approval_audit_events")
    op.drop_table("approval_requests")
