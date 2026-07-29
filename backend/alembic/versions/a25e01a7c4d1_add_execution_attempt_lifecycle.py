"""add execution attempt lifecycle

Revision ID: a25e01a7c4d1
Revises: f24e05a1b2c3
Create Date: 2026-08-31

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a25e01a7c4d1"
down_revision: Union[str, Sequence[str], None] = "f24e05a1b2c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the first-attempt marker and the partial-completion outcome."""

    with op.batch_alter_table(
        "operation_executions",
        recreate="always",
    ) as batch_op:
        batch_op.add_column(
            sa.Column(
                "attempt",
                sa.Integer(),
                server_default="1",
                nullable=False,
            )
        )
        batch_op.drop_constraint(
            "ck_operation_executions_status",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_operation_executions_status",
            "status IN "
            "('EXECUTING', 'PARTIALLY_COMPLETED', 'COMPLETED', "
            "'UNDOING', 'UNDONE', 'FAILED')",
        )
        batch_op.create_check_constraint(
            "ck_operation_executions_attempt_positive",
            "attempt >= 1",
        )


def downgrade() -> None:
    """Remove retry metadata and conservatively map partial outcomes."""

    # 旧版本不认识部分完成；降级时保守归类为失败，不能伪装成全部成功。
    op.execute(
        "UPDATE operation_executions "
        "SET status = 'FAILED' "
        "WHERE status = 'PARTIALLY_COMPLETED'"
    )

    with op.batch_alter_table(
        "operation_executions",
        recreate="always",
    ) as batch_op:
        batch_op.drop_constraint(
            "ck_operation_executions_attempt_positive",
            type_="check",
        )
        batch_op.drop_constraint(
            "ck_operation_executions_status",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_operation_executions_status",
            "status IN "
            "('EXECUTING', 'COMPLETED', 'UNDOING', 'UNDONE', 'FAILED')",
        )
        batch_op.drop_column("attempt")
