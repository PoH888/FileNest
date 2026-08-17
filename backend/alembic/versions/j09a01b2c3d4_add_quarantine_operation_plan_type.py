"""allow quarantine operation plans

Revision ID: j09a01b2c3d4
Revises: i08a01b2c3d4
Create Date: 2026-09-02

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "j09a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "i08a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Allow persisted plans and items to describe quarantine destinations."""

    with op.batch_alter_table("operation_plans", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_operation_plans_operation_type",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_operation_plans_operation_type",
            "operation_type IN ('move', 'quarantine')",
        )

    with op.batch_alter_table("operation_items", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_operation_items_operation_type",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_operation_items_operation_type",
            "operation_type IN ('move', 'quarantine')",
        )


def downgrade() -> None:
    """Restore move-only plan constraints."""

    with op.batch_alter_table("operation_items", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_operation_items_operation_type",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_operation_items_operation_type",
            "operation_type IN ('move')",
        )

    with op.batch_alter_table("operation_plans", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_operation_plans_operation_type",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_operation_plans_operation_type",
            "operation_type IN ('move')",
        )
