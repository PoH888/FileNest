"""add explainable Agent Run metadata fields

Revision ID: z27a01b2c3d4
Revises: y26a01b2c3d4
Create Date: 2026-09-03

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "z27a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "y26a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Store optional, non-sensitive Agent Run metadata for later aggregation."""

    with op.batch_alter_table("agent_runs", recreate="always") as batch_op:
        batch_op.add_column(sa.Column("model_provider", sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column("model_name", sa.String(length=200), nullable=True))
        batch_op.add_column(sa.Column("prompt_version", sa.String(length=128), nullable=True))
        batch_op.add_column(sa.Column("latency_ms", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("input_tokens", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("output_tokens", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "estimated_cost_usd",
                sa.Numeric(precision=20, scale=10),
                nullable=True,
            )
        )
        batch_op.create_check_constraint(
            "ck_agent_runs_model_provider",
            "model_provider IS NULL OR (length(model_provider) > 0 "
            "AND model_provider = trim(model_provider))",
        )
        batch_op.create_check_constraint(
            "ck_agent_runs_model_name",
            "model_name IS NULL OR (length(model_name) > 0 "
            "AND model_name = trim(model_name))",
        )
        batch_op.create_check_constraint(
            "ck_agent_runs_prompt_version",
            "prompt_version IS NULL OR (length(prompt_version) > 0 "
            "AND prompt_version = trim(prompt_version))",
        )
        batch_op.create_check_constraint(
            "ck_agent_runs_latency_non_negative",
            "latency_ms IS NULL OR latency_ms >= 0",
        )
        batch_op.create_check_constraint(
            "ck_agent_runs_input_tokens_non_negative",
            "input_tokens IS NULL OR input_tokens >= 0",
        )
        batch_op.create_check_constraint(
            "ck_agent_runs_output_tokens_non_negative",
            "output_tokens IS NULL OR output_tokens >= 0",
        )
        batch_op.create_check_constraint(
            "ck_agent_runs_token_usage_complete",
            "(input_tokens IS NULL AND output_tokens IS NULL) OR "
            "(input_tokens IS NOT NULL AND output_tokens IS NOT NULL)",
        )
        batch_op.create_check_constraint(
            "ck_agent_runs_estimated_cost_non_negative",
            "estimated_cost_usd IS NULL OR estimated_cost_usd >= 0",
        )


def downgrade() -> None:
    """Remove optional Agent Run metadata fields."""

    with op.batch_alter_table("agent_runs", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_agent_runs_estimated_cost_non_negative", type_="check")
        batch_op.drop_constraint("ck_agent_runs_token_usage_complete", type_="check")
        batch_op.drop_constraint("ck_agent_runs_output_tokens_non_negative", type_="check")
        batch_op.drop_constraint("ck_agent_runs_input_tokens_non_negative", type_="check")
        batch_op.drop_constraint("ck_agent_runs_latency_non_negative", type_="check")
        batch_op.drop_constraint("ck_agent_runs_prompt_version", type_="check")
        batch_op.drop_constraint("ck_agent_runs_model_name", type_="check")
        batch_op.drop_constraint("ck_agent_runs_model_provider", type_="check")
        batch_op.drop_column("estimated_cost_usd")
        batch_op.drop_column("output_tokens")
        batch_op.drop_column("input_tokens")
        batch_op.drop_column("latency_ms")
        batch_op.drop_column("prompt_version")
        batch_op.drop_column("model_name")
        batch_op.drop_column("model_provider")
