"""link Agent runs, sessions, steps, and tool-call records

Revision ID: aa28a01b2c3d4
Revises: z27a01b2c3d4
Create Date: 2026-09-03

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "aa28a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "z27a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Persist the stable ownership graph without deleting audit history."""

    op.create_table(
        "agent_run_sessions",
        sa.Column("agent_run_id", sa.Integer(), nullable=False),
        sa.Column("agent_session_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            ["agent_runs.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["agent_session_id"],
            ["agent_sessions.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("agent_run_id", "agent_session_id"),
    )
    op.create_index(
        "ix_agent_run_sessions_session_id",
        "agent_run_sessions",
        ["agent_session_id"],
    )

    with op.batch_alter_table("agent_tool_calls", recreate="always") as batch_op:
        batch_op.add_column(sa.Column("agent_step_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_agent_tool_calls_agent_step_id",
            "agent_steps",
            ["agent_step_id"],
            ["id"],
            ondelete="RESTRICT",
        )


def downgrade() -> None:
    """Remove lifecycle links while retaining the original tables."""

    op.drop_index(
        "ix_agent_run_sessions_session_id",
        table_name="agent_run_sessions",
    )
    op.drop_table("agent_run_sessions")

    with op.batch_alter_table("agent_tool_calls", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "fk_agent_tool_calls_agent_step_id",
            type_="foreignkey",
        )
        batch_op.drop_column("agent_step_id")
