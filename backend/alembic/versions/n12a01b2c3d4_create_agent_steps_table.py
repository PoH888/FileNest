"""create Agent step records table

Revision ID: n12a01b2c3d4
Revises: m12a01b2c3d4
Create Date: 2026-09-02

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "n12a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "m12a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create durable Agent step storage."""

    op.create_table(
        "agent_steps",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("agent_session_id", sa.Integer(), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("step_type", sa.String(length=64), nullable=False),
        sa.Column("input", sa.Text(), nullable=False),
        sa.Column("output_summary", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default="pending",
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "step_index >= 0",
            name="ck_agent_steps_index_non_negative",
        ),
        sa.CheckConstraint(
            "length(step_type) > 0 AND step_type = trim(step_type)",
            name="ck_agent_steps_type_non_empty",
        ),
        sa.CheckConstraint(
            "length(status) > 0 AND status = trim(status)",
            name="ck_agent_steps_status_non_empty",
        ),
        sa.ForeignKeyConstraint(["agent_session_id"], ["agent_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "agent_session_id",
            "step_index",
            name="uq_agent_steps_session_index",
        ),
    )


def downgrade() -> None:
    """Remove Agent step storage."""

    op.drop_table("agent_steps")
