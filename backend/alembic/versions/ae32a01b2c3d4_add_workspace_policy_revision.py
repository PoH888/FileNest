"""add persisted workspace policy and audit records

Revision ID: ae32a01b2c3d4
Revises: ad31a01b2c3d4
Create Date: 2026-09-03

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "ae32a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "ad31a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create policy facts and backfill the compatible default for old workspaces."""

    op.create_table(
        "workspace_policies",
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column(
            "policy_revision",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "read_enabled",
            sa.Boolean(),
            server_default=sa.true(),
            nullable=False,
        ),
        sa.Column(
            "proposal_enabled",
            sa.Boolean(),
            server_default=sa.true(),
            nullable=False,
        ),
        sa.Column(
            "safe_execution_enabled",
            sa.Boolean(),
            server_default=sa.true(),
            nullable=False,
        ),
        sa.Column(
            "user_denylist_json",
            sa.Text(),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "ignore_patterns_json",
            sa.Text(),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "policy_revision >= 0",
            name="ck_workspace_policies_revision_non_negative",
        ),
        sa.CheckConstraint(
            "length(user_denylist_json) > 0",
            name="ck_workspace_policies_denylist_json_present",
        ),
        sa.CheckConstraint(
            "length(ignore_patterns_json) > 0",
            name="ck_workspace_policies_ignore_json_present",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("workspace_id"),
    )

    op.create_table(
        "workspace_policy_audit_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("source", sa.String(length=128), nullable=False),
        sa.Column("previous_revision", sa.Integer(), nullable=False),
        sa.Column("next_revision", sa.Integer(), nullable=False),
        sa.Column("added_rules_json", sa.Text(), nullable=False),
        sa.Column("removed_rules_json", sa.Text(), nullable=False),
        sa.Column("result", sa.String(length=16), nullable=False),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "previous_revision >= 0 AND next_revision >= 0",
            name="ck_workspace_policy_audits_revision_non_negative",
        ),
        sa.CheckConstraint(
            "next_revision >= previous_revision",
            name="ck_workspace_policy_audits_revision_order",
        ),
        sa.CheckConstraint(
            "length(actor) BETWEEN 1 AND 128 AND actor = trim(actor)",
            name="ck_workspace_policy_audits_actor",
        ),
        sa.CheckConstraint(
            "length(source) BETWEEN 1 AND 128 AND source = trim(source)",
            name="ck_workspace_policy_audits_source",
        ),
        sa.CheckConstraint(
            "length(added_rules_json) > 0",
            name="ck_workspace_policy_audits_added_json_present",
        ),
        sa.CheckConstraint(
            "length(removed_rules_json) > 0",
            name="ck_workspace_policy_audits_removed_json_present",
        ),
        sa.CheckConstraint(
            "result IN ('succeeded', 'failed')",
            name="ck_workspace_policy_audits_result",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_workspace_policy_audit_events_workspace_id",
        "workspace_policy_audit_events",
        ["workspace_id"],
        unique=False,
    )

    op.execute(
        sa.text(
            "INSERT INTO workspace_policies "
            "(workspace_id, policy_revision, read_enabled, "
            "proposal_enabled, safe_execution_enabled, "
            "user_denylist_json, ignore_patterns_json) "
            "SELECT id, 0, 1, 1, 1, '[]', '[]' FROM workspaces"
        )
    )


def downgrade() -> None:
    """Remove persisted policy facts while leaving workspace data intact."""

    op.drop_index(
        "ix_workspace_policy_audit_events_workspace_id",
        table_name="workspace_policy_audit_events",
    )
    op.drop_table("workspace_policy_audit_events")
    op.drop_table("workspace_policies")
