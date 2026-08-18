"""store the input and validated message context needed by Agent Resume

Revision ID: l11a01b2c3d4
Revises: k10a01b2c3d4
Create Date: 2026-09-02

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "l11a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "k10a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add nullable fields so historical runs remain readable but not resumable."""

    with op.batch_alter_table("agent_runs", recreate="always") as batch_op:
        batch_op.add_column(sa.Column("workspace_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("request_text", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("context_json", sa.Text(), nullable=True))
        batch_op.create_foreign_key(
            "fk_agent_runs_workspace_id_workspaces",
            "workspaces",
            ["workspace_id"],
            ["id"],
        )


def downgrade() -> None:
    """Remove only Resume metadata and preserve the Agent lifecycle records."""

    with op.batch_alter_table("agent_runs", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "fk_agent_runs_workspace_id_workspaces",
            type_="foreignkey",
        )
        batch_op.drop_column("context_json")
        batch_op.drop_column("request_text")
        batch_op.drop_column("workspace_id")
