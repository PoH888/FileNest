"""create Agent intermediate messages table

Revision ID: o12a01b2c3d4
Revises: n12a01b2c3d4
Create Date: 2026-09-02

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "o12a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "n12a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create durable Agent intermediate message storage."""

    op.create_table(
        "agent_messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("agent_step_id", sa.Integer(), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("message_type", sa.String(length=32), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "sequence_no >= 0",
            name="ck_agent_messages_sequence_non_negative",
        ),
        sa.CheckConstraint(
            "message_type IN ('user', 'assistant', 'tool_call', 'tool_result')",
            name="ck_agent_messages_type",
        ),
        sa.CheckConstraint(
            "length(payload_json) > 0",
            name="ck_agent_messages_payload_non_empty",
        ),
        sa.ForeignKeyConstraint(["agent_step_id"], ["agent_steps.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "agent_step_id",
            "sequence_no",
            name="uq_agent_messages_step_sequence",
        ),
    )


def downgrade() -> None:
    """Remove Agent intermediate message storage."""

    op.drop_table("agent_messages")
