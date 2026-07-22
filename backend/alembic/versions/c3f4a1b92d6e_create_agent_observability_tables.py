"""create agent observability tables

Revision ID: c3f4a1b92d6e
Revises: 8b872f337530
Create Date: 2026-08-30

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c3f4a1b92d6e"
down_revision: Union[str, Sequence[str], None] = "8b872f337530"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create safe Agent Run and Tool Call lifecycle storage."""

    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.String(),
            server_default="running",
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "model_turns",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'max_steps_reached', "
            "'timed_out', 'cancelled', 'failed')",
            name="ck_agent_runs_status",
        ),
        sa.CheckConstraint(
            "model_turns >= 0",
            name="ck_agent_runs_model_turns_non_negative",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "agent_tool_calls",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("agent_run_id", sa.Integer(), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("model_call_id", sa.String(), nullable=False),
        sa.Column("tool_name", sa.String(), nullable=False),
        sa.Column(
            "status",
            sa.String(),
            server_default="requested",
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.CheckConstraint(
            "sequence_no >= 1",
            name="ck_agent_tool_calls_sequence_positive",
        ),
        sa.CheckConstraint(
            "status IN ('requested', 'succeeded', 'rejected', 'failed')",
            name="ck_agent_tool_calls_status",
        ),
        sa.ForeignKeyConstraint(["agent_run_id"], ["agent_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "agent_run_id",
            "model_call_id",
            name="uq_agent_tool_calls_run_model_call_id",
        ),
        sa.UniqueConstraint(
            "agent_run_id",
            "sequence_no",
            name="uq_agent_tool_calls_run_sequence",
        ),
    )


def downgrade() -> None:
    """Remove Agent observability tables without touching file indexes."""

    op.drop_table("agent_tool_calls")
    op.drop_table("agent_runs")
