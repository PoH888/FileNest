"""allow cancellation in the approval lifecycle

Revision ID: i08a01b2c3d4
Revises: h07a01b2c3d4
Create Date: 2026-09-02

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "i08a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "h07a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Allow explicit cancellation in plan, approval, and audit records."""

    with op.batch_alter_table("operation_plans", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_operation_plans_status", type_="check")
        batch_op.create_check_constraint(
            "ck_operation_plans_status",
            "status IN ('WAITING_APPROVAL', 'APPROVED', 'REJECTED', "
            "'CANCELLED', 'SUPERSEDED')",
        )

    with op.batch_alter_table("approval_requests", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_approval_requests_status", type_="check")
        batch_op.create_check_constraint(
            "ck_approval_requests_status",
            "status IN ('WAITING_APPROVAL', 'APPROVED', 'REJECTED', 'CANCELLED')",
        )

    with op.batch_alter_table("approval_audit_events", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_approval_audit_events_action", type_="check")
        batch_op.drop_constraint(
            "ck_approval_audit_events_previous_status",
            type_="check",
        )
        batch_op.drop_constraint(
            "ck_approval_audit_events_next_status",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_approval_audit_events_action",
            "action IN ('approve', 'edit', 'reject', 'cancel')",
        )
        batch_op.create_check_constraint(
            "ck_approval_audit_events_previous_status",
            "previous_status IN "
            "('WAITING_APPROVAL', 'APPROVED', 'REJECTED', 'CANCELLED')",
        )
        batch_op.create_check_constraint(
            "ck_approval_audit_events_next_status",
            "next_status IN "
            "('WAITING_APPROVAL', 'APPROVED', 'REJECTED', 'CANCELLED')",
        )


def downgrade() -> None:
    """Restore the pre-cancellation approval constraints."""

    with op.batch_alter_table("approval_audit_events", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_approval_audit_events_action", type_="check")
        batch_op.drop_constraint(
            "ck_approval_audit_events_previous_status",
            type_="check",
        )
        batch_op.drop_constraint(
            "ck_approval_audit_events_next_status",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_approval_audit_events_action",
            "action IN ('approve', 'edit', 'reject')",
        )
        batch_op.create_check_constraint(
            "ck_approval_audit_events_previous_status",
            "previous_status IN ('WAITING_APPROVAL', 'APPROVED', 'REJECTED')",
        )
        batch_op.create_check_constraint(
            "ck_approval_audit_events_next_status",
            "next_status IN ('WAITING_APPROVAL', 'APPROVED', 'REJECTED')",
        )

    with op.batch_alter_table("approval_requests", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_approval_requests_status", type_="check")
        batch_op.create_check_constraint(
            "ck_approval_requests_status",
            "status IN ('WAITING_APPROVAL', 'APPROVED', 'REJECTED')",
        )

    with op.batch_alter_table("operation_plans", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_operation_plans_status", type_="check")
        batch_op.create_check_constraint(
            "ck_operation_plans_status",
            "status IN ('WAITING_APPROVAL', 'APPROVED', 'REJECTED', 'SUPERSEDED')",
        )
