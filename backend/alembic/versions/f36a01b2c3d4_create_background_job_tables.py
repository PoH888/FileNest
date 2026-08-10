"""create persisted background job and attempt records

Revision ID: f36a01b2c3d4
Revises: e32a01b2c3d4
Create Date: 2026-09-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f36a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "e32a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create durable Job identity and append-only Attempt identity tables."""

    op.create_table(
        "background_jobs",
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column(
            "schema_version",
            sa.Integer(),
            server_default="1",
            nullable=False,
        ),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column(
            "idempotency_key",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default="pending",
            nullable=False,
        ),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column(
            "revision",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        ),
        sa.Column(
            "cancel_requested_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "finished_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.CheckConstraint(
            "schema_version = 1",
            name="ck_background_jobs_schema_version",
        ),
        sa.CheckConstraint(
            "kind IN ('workspace_scan', 'document_index')",
            name="ck_background_jobs_kind",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'cancel_requested', "
            "'succeeded', 'failed', 'cancelled')",
            name="ck_background_jobs_status",
        ),
        sa.CheckConstraint(
            "length(idempotency_key) BETWEEN 1 AND 128 "
            "AND idempotency_key = trim(idempotency_key)",
            name="ck_background_jobs_idempotency_key",
        ),
        sa.CheckConstraint(
            "max_attempts BETWEEN 1 AND 10",
            name="ck_background_jobs_max_attempts",
        ),
        sa.CheckConstraint(
            "revision >= 0",
            name="ck_background_jobs_revision_non_negative",
        ),
        sa.CheckConstraint(
            "((status IN ('succeeded', 'failed', 'cancelled')) "
            "AND finished_at IS NOT NULL) OR "
            "((status IN ('pending', 'running', 'cancel_requested')) "
            "AND finished_at IS NULL)",
            name="ck_background_jobs_finished_at",
        ),
        sa.CheckConstraint(
            "(status = 'failed' AND error_code IS NOT NULL) OR "
            "(status <> 'failed' AND error_code IS NULL)",
            name="ck_background_jobs_error_code",
        ),
        sa.CheckConstraint(
            "status NOT IN ('cancel_requested', 'cancelled') "
            "OR cancel_requested_at IS NOT NULL",
            name="ck_background_jobs_cancellation",
        ),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("job_id"),
        sa.UniqueConstraint(
            "idempotency_key",
            name="uq_background_jobs_idempotency_key",
        ),
    )
    op.create_index(
        "ix_background_jobs_workspace_id",
        "background_jobs",
        ["workspace_id"],
        unique=False,
    )

    op.create_table(
        "background_job_attempts",
        sa.Column("attempt_id", sa.String(length=36), nullable=False),
        sa.Column(
            "schema_version",
            sa.Integer(),
            server_default="1",
            nullable=False,
        ),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default="running",
            nullable=False,
        ),
        sa.Column(
            "completed_units",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column("total_units", sa.Integer(), nullable=True),
        sa.Column(
            "phase_code",
            sa.String(length=64),
            server_default="starting",
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        ),
        sa.Column(
            "finished_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column(
            "retryable",
            sa.Boolean(),
            server_default="0",
            nullable=False,
        ),
        sa.CheckConstraint(
            "schema_version = 1",
            name="ck_background_job_attempts_schema_version",
        ),
        sa.CheckConstraint(
            "attempt_no >= 1",
            name="ck_background_job_attempts_number_positive",
        ),
        sa.CheckConstraint(
            "status IN ('running', 'succeeded', 'failed', "
            "'cancelled', 'interrupted')",
            name="ck_background_job_attempts_status",
        ),
        sa.CheckConstraint(
            "completed_units >= 0 AND "
            "(total_units IS NULL OR (total_units >= 0 "
            "AND completed_units <= total_units))",
            name="ck_background_job_attempts_progress",
        ),
        sa.CheckConstraint(
            "length(phase_code) BETWEEN 1 AND 64 "
            "AND phase_code = trim(phase_code)",
            name="ck_background_job_attempts_phase_code",
        ),
        sa.CheckConstraint(
            "(status = 'running' AND finished_at IS NULL) OR "
            "(status <> 'running' AND finished_at IS NOT NULL)",
            name="ck_background_job_attempts_finished_at",
        ),
        sa.CheckConstraint(
            "(status IN ('failed', 'interrupted') "
            "AND error_code IS NOT NULL) OR "
            "(status NOT IN ('failed', 'interrupted') "
            "AND error_code IS NULL)",
            name="ck_background_job_attempts_error_code",
        ),
        sa.CheckConstraint(
            "(status = 'interrupted' AND retryable = 1) OR "
            "(status IN ('running', 'succeeded', 'cancelled') "
            "AND retryable = 0) OR status = 'failed'",
            name="ck_background_job_attempts_retryable",
        ),
        sa.ForeignKeyConstraint(["job_id"], ["background_jobs.job_id"]),
        sa.PrimaryKeyConstraint("attempt_id"),
        sa.UniqueConstraint(
            "job_id",
            "attempt_no",
            name="uq_background_job_attempts_job_number",
        ),
    )
    op.create_index(
        "ix_background_job_attempts_job_id",
        "background_job_attempts",
        ["job_id"],
        unique=False,
    )


def downgrade() -> None:
    """Remove Attempts before their parent Jobs."""

    op.drop_table("background_job_attempts")
    op.drop_table("background_jobs")
