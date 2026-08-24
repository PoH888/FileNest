"""add document ingestion status

Revision ID: w25a01b2c3d4
Revises: v24a01b2c3d4
Create Date: 2026-09-02

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "w25a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "v24a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Persist the ingestion lifecycle state for each document record."""

    with op.batch_alter_table("documents", recreate="always") as batch_op:
        batch_op.add_column(
            sa.Column(
                "ingest_status",
                sa.String(length=20),
                nullable=False,
                server_default=sa.text("'pending'"),
            )
        )
        batch_op.create_check_constraint(
            "ck_documents_ingest_status",
            "ingest_status IN ('pending', 'parsing', 'indexed', 'failed')",
        )

    # Existing rows were written only after a successful parse and index.
    op.execute("UPDATE documents SET ingest_status = 'indexed'")


def downgrade() -> None:
    """Remove persisted document ingestion state."""

    with op.batch_alter_table("documents", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_documents_ingest_status",
            type_="check",
        )
        batch_op.drop_column("ingest_status")
