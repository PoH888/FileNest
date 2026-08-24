"""add PDF page provenance storage

Revision ID: u23a01b2c3d4
Revises: t22a02b2c3d4
Create Date: 2026-09-02

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "u23a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "t22a02b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Persist PDF pages and the page range covered by each chunk."""

    op.create_table(
        "document_pages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "page_number >= 1",
            name="ck_document_pages_number_positive",
        ),
        sa.CheckConstraint(
            "start_offset >= 0 AND end_offset >= start_offset",
            name="ck_document_pages_offset_order",
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.document_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "page_number",
            name="uq_document_pages_document_number",
        ),
    )
    op.create_index(
        "ix_document_pages_document_id",
        "document_pages",
        ["document_id"],
        unique=False,
    )

    with op.batch_alter_table("document_chunks", recreate="always") as batch_op:
        batch_op.add_column(sa.Column("page_start", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("page_end", sa.Integer(), nullable=True))
        batch_op.create_check_constraint(
            "ck_document_chunks_page_range",
            "(page_start IS NULL AND page_end IS NULL) "
            "OR (page_start >= 1 AND page_end >= page_start)",
        )


def downgrade() -> None:
    """Remove chunk page ranges and persisted PDF pages."""

    with op.batch_alter_table("document_chunks", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_document_chunks_page_range",
            type_="check",
        )
        batch_op.drop_column("page_end")
        batch_op.drop_column("page_start")

    op.drop_table("document_pages")
