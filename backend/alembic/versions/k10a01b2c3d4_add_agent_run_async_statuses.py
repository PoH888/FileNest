"""allow queued and approval-waiting Agent Run states

Revision ID: k10a01b2c3d4
Revises: j09a01b2c3d4
Create Date: 2026-09-02

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "k10a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "j09a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Allow the asynchronous Agent Run lifecycle states."""

    with op.batch_alter_table("agent_runs", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_agent_runs_status", type_="check")
        batch_op.create_check_constraint(
            "ck_agent_runs_status",
            "status IN ('pending', 'running', 'waiting_approval', "
            "'completed', 'max_steps_reached', 'timed_out', "
            "'cancelled', 'failed')",
        )


def downgrade() -> None:
    """Restore the pre-asynchronous Agent Run status constraint."""

    with op.batch_alter_table("agent_runs", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_agent_runs_status", type_="check")
        batch_op.create_check_constraint(
            "ck_agent_runs_status",
            "status IN ('running', 'completed', 'max_steps_reached', "
            "'timed_out', 'cancelled', 'failed')",
        )
