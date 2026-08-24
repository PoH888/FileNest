"""add DOCX structure provenance storage

Revision ID: v24a01b2c3d4
Revises: u23a01b2c3d4
Create Date: 2026-09-02

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "v24a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "u23a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Persist DOCX source positions and each chunk's provenance list."""

    op.create_table(
        "document_positions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("position_index", sa.Integer(), nullable=False),
        sa.Column("element_type", sa.String(length=20), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("section_index", sa.Integer(), nullable=True),
        sa.Column("heading_level", sa.Integer(), nullable=True),
        sa.Column("paragraph_index", sa.Integer(), nullable=True),
        sa.Column("table_index", sa.Integer(), nullable=True),
        sa.Column("row_index", sa.Integer(), nullable=True),
        sa.Column("cell_index", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "position_index >= 0",
            name="ck_document_positions_index_non_negative",
        ),
        sa.CheckConstraint(
            "element_type IN ('paragraph', 'table_cell')",
            name="ck_document_positions_element_type",
        ),
        sa.CheckConstraint(
            "start_offset >= 0 AND end_offset >= start_offset",
            name="ck_document_positions_offset_order",
        ),
        sa.CheckConstraint(
            "section_index IS NULL OR section_index >= 0",
            name="ck_document_positions_section_index",
        ),
        sa.CheckConstraint(
            "heading_level IS NULL OR heading_level >= 1",
            name="ck_document_positions_heading_level",
        ),
        sa.CheckConstraint(
            "paragraph_index IS NULL OR paragraph_index >= 0",
            name="ck_document_positions_paragraph_index",
        ),
        sa.CheckConstraint(
            "table_index IS NULL OR table_index >= 0",
            name="ck_document_positions_table_index",
        ),
        sa.CheckConstraint(
            "row_index IS NULL OR row_index >= 0",
            name="ck_document_positions_row_index",
        ),
        sa.CheckConstraint(
            "cell_index IS NULL OR cell_index >= 0",
            name="ck_document_positions_cell_index",
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.document_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "position_index",
            name="uq_document_positions_document_index",
        ),
    )
    op.create_index(
        "ix_document_positions_document_id",
        "document_positions",
        ["document_id"],
        unique=False,
    )

    with op.batch_alter_table("document_chunks", recreate="always") as batch_op:
        batch_op.add_column(
            sa.Column("source_positions_json", sa.Text(), nullable=True)
        )
        batch_op.create_check_constraint(
            "ck_document_chunks_source_positions_non_empty",
            "source_positions_json IS NULL OR length(source_positions_json) > 0",
        )


def downgrade() -> None:
    """Remove chunk DOCX provenance and persisted DOCX source positions."""

    with op.batch_alter_table("document_chunks", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_document_chunks_source_positions_non_empty",
            type_="check",
        )
        batch_op.drop_column("source_positions_json")

    op.drop_table("document_positions")
