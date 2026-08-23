"""allow PDF as a persisted document format

Revision ID: t22a01b2c3d4
Revises: s20a01b2c3d4
Create Date: 2026-09-02

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "t22a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "s20a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Allow PDF documents while retaining the existing supported formats."""

    with op.batch_alter_table("documents", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_documents_source_format",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_documents_source_format",
            "source_format IN ('markdown', 'text', 'pdf')",
        )


def downgrade() -> None:
    """Restore the pre-T22-01 document format constraint."""

    with op.batch_alter_table("documents", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_documents_source_format",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_documents_source_format",
            "source_format IN ('markdown', 'text')",
        )
