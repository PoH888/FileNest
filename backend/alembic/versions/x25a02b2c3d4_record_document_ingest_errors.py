"""record document ingestion errors

Revision ID: x25a02b2c3d4
Revises: w25a01b2c3d4
Create Date: 2026-09-02

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "x25a02b2c3d4"
down_revision: Union[str, Sequence[str], None] = "w25a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Allow failed attempts to retain their error without fake source data."""

    with op.batch_alter_table("documents", recreate="always") as batch_op:
        batch_op.alter_column(
            "source_version",
            existing_type=sa.String(length=64),
            nullable=True,
        )
        batch_op.alter_column(
            "source_updated_at",
            existing_type=sa.DateTime(timezone=True),
            nullable=True,
        )
        batch_op.add_column(
            sa.Column("ingest_error", sa.Text(), nullable=True)
        )
        batch_op.create_check_constraint(
            "ck_documents_ingest_error",
            "(ingest_status = 'failed' AND ingest_error IS NOT NULL) "
            "OR (ingest_status <> 'failed' AND ingest_error IS NULL)",
        )


def downgrade() -> None:
    """Remove persisted ingestion errors when no incomplete records remain."""

    connection = op.get_bind()
    failed_count = connection.execute(
        sa.text("SELECT COUNT(*) FROM documents WHERE ingest_status = 'failed'")
    ).scalar_one()
    if failed_count:
        raise RuntimeError(
            "cannot downgrade document ingest errors while failed records exist"
        )

    with op.batch_alter_table("documents", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_documents_ingest_error",
            type_="check",
        )
        batch_op.drop_column("ingest_error")
        batch_op.alter_column(
            "source_version",
            existing_type=sa.String(length=64),
            nullable=False,
        )
        batch_op.alter_column(
            "source_updated_at",
            existing_type=sa.DateTime(timezone=True),
            nullable=False,
        )
