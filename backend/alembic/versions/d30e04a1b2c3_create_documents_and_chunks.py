"""create document and chunk traceability storage

Revision ID: d30e04a1b2c3
Revises: b25e03a6d8f2
Create Date: 2026-08-31

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d30e04a1b2c3"
down_revision: Union[str, Sequence[str], None] = "b25e03a6d8f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create persisted document source metadata and chunk locations."""

    op.create_table(
        "documents",
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("file_entry_id", sa.Integer(), nullable=False),
        sa.Column("source_relative_path", sa.String(), nullable=False),
        sa.Column("source_format", sa.String(length=20), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("source_version", sa.String(length=64), nullable=False),
        sa.Column(
            "source_updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "source_format IN ('markdown', 'text')",
            name="ck_documents_source_format",
        ),
        sa.ForeignKeyConstraint(["file_entry_id"], ["file_entries.id"]),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("document_id"),
        sa.UniqueConstraint(
            "file_entry_id",
            "source_version",
            name="uq_documents_file_entry_source_version",
        ),
    )
    op.create_index(
        "ix_documents_file_entry_id",
        "documents",
        ["file_entry_id"],
        unique=False,
    )
    op.create_index(
        "ix_documents_workspace_id",
        "documents",
        ["workspace_id"],
        unique=False,
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("file_entry_id", sa.Integer(), nullable=False),
        sa.Column("source_relative_path", sa.String(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "chunk_index >= 0",
            name="ck_document_chunks_index_non_negative",
        ),
        sa.CheckConstraint(
            "start_offset >= 0 AND end_offset > start_offset",
            name="ck_document_chunks_offset_order",
        ),
        sa.CheckConstraint(
            "start_line >= 1 AND end_line >= start_line",
            name="ck_document_chunks_line_order",
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.document_id"]),
        sa.ForeignKeyConstraint(["file_entry_id"], ["file_entries.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunks_document_index",
        ),
        sa.UniqueConstraint(
            "chunk_id",
            name="uq_document_chunks_chunk_id",
        ),
    )
    op.create_index(
        "ix_document_chunks_document_id",
        "document_chunks",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        "ix_document_chunks_file_entry_id",
        "document_chunks",
        ["file_entry_id"],
        unique=False,
    )


def downgrade() -> None:
    """Remove chunks before their parent documents."""

    op.drop_table("document_chunks")
    op.drop_table("documents")
