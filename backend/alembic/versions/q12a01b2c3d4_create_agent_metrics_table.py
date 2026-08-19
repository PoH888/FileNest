"""create Agent metrics table

Revision ID: q12a01b2c3d4
Revises: p12a01b2c3d4
Create Date: 2026-09-02

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "q12a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "p12a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create extensible Agent metric storage."""

    op.create_table(
        "agent_metrics",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("agent_session_id", sa.Integer(), nullable=False),
        sa.Column("agent_step_id", sa.Integer(), nullable=True),
        sa.Column("agent_model_run_id", sa.Integer(), nullable=True),
        sa.Column("metric_name", sa.String(length=128), nullable=False),
        sa.Column("value_json", sa.Text(), nullable=False),
        sa.Column("unit", sa.String(length=32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(metric_name) > 0 AND metric_name = trim(metric_name)",
            name="ck_agent_metrics_name_non_empty",
        ),
        sa.CheckConstraint(
            "length(value_json) > 0",
            name="ck_agent_metrics_value_non_empty",
        ),
        sa.CheckConstraint(
            "unit IS NULL OR (length(unit) > 0 AND unit = trim(unit))",
            name="ck_agent_metrics_unit",
        ),
        sa.ForeignKeyConstraint(["agent_session_id"], ["agent_sessions.id"]),
        sa.ForeignKeyConstraint(["agent_step_id"], ["agent_steps.id"]),
        sa.ForeignKeyConstraint(
            ["agent_model_run_id"],
            ["agent_model_runs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Remove Agent metric storage."""

    op.drop_table("agent_metrics")
