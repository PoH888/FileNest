"""记录文档索引完成时间。"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "ad31a01b2c3d4"
down_revision: str | None = "ac30a01b2c3d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("documents", "indexed_at")
