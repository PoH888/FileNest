"""allow rename operation plans and execution history

Revision ID: r13a01b2c3d4
Revises: q12a01b2c3d4
Create Date: 2026-09-02

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "r13a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "q12a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Allow plans, items, and execution history to describe rename."""

    with op.batch_alter_table("operation_plans", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_operation_plans_operation_type",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_operation_plans_operation_type",
            "operation_type IN ('move', 'quarantine', 'rename')",
        )

    with op.batch_alter_table("operation_items", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_operation_items_operation_type",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_operation_items_operation_type",
            "operation_type IN ('move', 'quarantine', 'rename')",
        )

    with op.batch_alter_table(
        "operation_execution_items",
        recreate="always",
    ) as batch_op:
        batch_op.drop_constraint(
            "ck_operation_execution_items_type",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_operation_execution_items_type",
            "operation_type IN ('move', 'quarantine', 'rename')",
        )


def downgrade() -> None:
    """Restore the pre-rename operation type constraints."""

    with op.batch_alter_table(
        "operation_execution_items",
        recreate="always",
    ) as batch_op:
        batch_op.drop_constraint(
            "ck_operation_execution_items_type",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_operation_execution_items_type",
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

    with op.batch_alter_table("operation_plans", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_operation_plans_operation_type",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_operation_plans_operation_type",
            "operation_type IN ('move', 'quarantine')",
        )
