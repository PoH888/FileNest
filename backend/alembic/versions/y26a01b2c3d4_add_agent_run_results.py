"""add persisted agent run results and proposal ownership

Revision ID: y26a01b2c3d4
Revises: x25a02b2c3d4
Create Date: 2026-09-03

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "y26a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "x25a02b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Persist user-visible Agent results and link plans to their Run."""

    with op.batch_alter_table("agent_runs", recreate="always") as batch_op:
        batch_op.add_column(sa.Column("final_answer", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("sources_json", sa.Text(), nullable=True))

    with op.batch_alter_table("operation_plans", recreate="always") as batch_op:
        batch_op.add_column(sa.Column("agent_run_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_operation_plans_agent_run_id_agent_runs",
            "agent_runs",
            ["agent_run_id"],
            ["id"],
        )
        batch_op.create_index(
            "ix_operation_plans_agent_run_id",
            ["agent_run_id"],
            unique=False,
        )


def downgrade() -> None:
    """Remove Agent result storage and the optional plan ownership link."""

    with op.batch_alter_table("operation_plans", recreate="always") as batch_op:
        batch_op.drop_index("ix_operation_plans_agent_run_id")
        batch_op.drop_constraint(
            "fk_operation_plans_agent_run_id_agent_runs",
            type_="foreignkey",
        )
        batch_op.drop_column("agent_run_id")

    with op.batch_alter_table("agent_runs", recreate="always") as batch_op:
        batch_op.drop_column("sources_json")
        batch_op.drop_column("final_answer")
