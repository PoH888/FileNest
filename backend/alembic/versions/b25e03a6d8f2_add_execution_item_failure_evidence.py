"""add execution item failure evidence

Revision ID: b25e03a6d8f2
Revises: a25e01a7c4d1
Create Date: 2026-08-31

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b25e03a6d8f2"
down_revision: Union[str, Sequence[str], None] = "a25e01a7c4d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Persist stable per-item failure evidence without exception text."""

    op.add_column(
        "operation_execution_items",
        sa.Column("error_code", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "operation_execution_items",
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Remove failure evidence while preserving the execution items."""

    with op.batch_alter_table(
        "operation_execution_items",
        recreate="always",
    ) as batch_op:
        batch_op.drop_column("failed_at")
        batch_op.drop_column("error_code")
