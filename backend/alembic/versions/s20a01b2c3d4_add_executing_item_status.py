"""persist the in-progress state for execution items

Revision ID: s20a01b2c3d4
Revises: r13a01b2c3d4
Create Date: 2026-09-02

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "s20a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "r13a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Allow an item to durably record that its file action has started."""

    with op.batch_alter_table(
        "operation_execution_items",
        recreate="always",
    ) as batch_op:
        batch_op.drop_constraint(
            "ck_operation_execution_items_status",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_operation_execution_items_status",
            "status IN ('PENDING', 'EXECUTING', 'COMPLETED', 'UNDOING', "
            "'UNDONE', 'FAILED')",
        )


def downgrade() -> None:
    """Restore the previous item status constraint."""

    with op.batch_alter_table(
        "operation_execution_items",
        recreate="always",
    ) as batch_op:
        batch_op.drop_constraint(
            "ck_operation_execution_items_status",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_operation_execution_items_status",
            "status IN ('PENDING', 'COMPLETED', 'UNDOING', 'UNDONE', "
            "'FAILED')",
        )
