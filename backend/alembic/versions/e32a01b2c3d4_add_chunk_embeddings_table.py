"""create persisted chunk embedding records

Revision ID: e32a01b2c3d4
Revises: d30e04a1b2c3
Create Date: 2026-08-31

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e32a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "d30e04a1b2c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the small SQLite-backed vector storage for document chunks."""

    op.create_table(
        "chunk_embeddings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.String(length=36), nullable=False),
        sa.Column(
            "embedding_model",
            sa.String(length=200),
            nullable=False,
        ),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("vector_json", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "dimension > 0",
            name="ck_chunk_embeddings_dimension_positive",
        ),
        sa.CheckConstraint(
            "length(embedding_model) > 0",
            name="ck_chunk_embeddings_model_non_empty",
        ),
        sa.CheckConstraint(
            "length(vector_json) > 0",
            name="ck_chunk_embeddings_vector_non_empty",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["document_chunks.chunk_id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "chunk_id",
            "embedding_model",
            name="uq_chunk_embeddings_chunk_model",
        ),
    )
    op.create_index(
        "ix_chunk_embeddings_chunk_id",
        "chunk_embeddings",
        ["chunk_id"],
        unique=False,
    )


def downgrade() -> None:
    """Remove vector records before their parent chunk table is removed."""

    op.drop_table("chunk_embeddings")
