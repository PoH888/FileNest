"""create Agent model run information table

Revision ID: p12a01b2c3d4
Revises: o12a01b2c3d4
Create Date: 2026-09-02

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "p12a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "o12a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create durable Agent model run information storage."""

    op.create_table(
        "agent_model_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("agent_step_id", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("prompt_version", sa.String(length=128), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(model) > 0 AND model = trim(model)",
            name="ck_agent_model_runs_model_non_empty",
        ),
        sa.CheckConstraint(
            "prompt_version IS NULL OR (length(prompt_version) > 0 "
            "AND prompt_version = trim(prompt_version))",
            name="ck_agent_model_runs_prompt_version",
        ),
        sa.CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_agent_model_runs_input_tokens_non_negative",
        ),
        sa.CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_agent_model_runs_output_tokens_non_negative",
        ),
        sa.CheckConstraint(
            "total_tokens IS NULL OR total_tokens >= 0",
            name="ck_agent_model_runs_total_tokens_non_negative",
        ),
        sa.CheckConstraint(
            "(input_tokens IS NULL AND output_tokens IS NULL "
            "AND total_tokens IS NULL) OR "
            "(input_tokens IS NOT NULL AND output_tokens IS NOT NULL "
            "AND total_tokens IS NOT NULL)",
            name="ck_agent_model_runs_token_usage_complete",
        ),
        sa.CheckConstraint(
            "latency_ms >= 0",
            name="ck_agent_model_runs_latency_non_negative",
        ),
        sa.ForeignKeyConstraint(["agent_step_id"], ["agent_steps.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Remove Agent model run information storage."""

    op.drop_table("agent_model_runs")
