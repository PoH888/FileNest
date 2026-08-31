"""persist versioned, rebuildable background job task definitions

Revision ID: ab29a01b2c3d4
Revises: aa28a01b2c3d4
Create Date: 2026-09-03

"""

import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "ab29a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "aa28a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add task version and backfill a safe workspace-only payload."""

    op.add_column(
        "background_jobs",
        sa.Column("task_version", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "background_jobs",
        sa.Column("payload_json", sa.Text(), nullable=True),
    )

    bind = op.get_bind()
    jobs = sa.table(
        "background_jobs",
        sa.column("job_id", sa.String(length=36)),
        sa.column("workspace_id", sa.Integer()),
        sa.column("task_version", sa.String(length=32)),
        sa.column("payload_json", sa.Text()),
    )
    rows = bind.execute(
        sa.select(jobs.c.job_id, jobs.c.workspace_id)
    ).mappings()
    for row in rows:
        bind.execute(
            sa.update(jobs)
            .where(jobs.c.job_id == row["job_id"])
            .values(
                task_version="v1",
                payload_json=json.dumps(
                    {"workspace_id": row["workspace_id"]},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )

    with op.batch_alter_table("background_jobs", recreate="always") as batch_op:
        batch_op.alter_column(
            "task_version",
            existing_type=sa.String(length=32),
            nullable=False,
            server_default="v1",
        )
        batch_op.alter_column(
            "payload_json",
            existing_type=sa.Text(),
            nullable=False,
        )
        batch_op.create_check_constraint(
            "ck_background_jobs_task_version",
            "length(task_version) BETWEEN 1 AND 32 "
            "AND task_version = trim(task_version)",
        )
        batch_op.create_check_constraint(
            "ck_background_jobs_payload_json",
            "length(payload_json) > 0",
        )


def downgrade() -> None:
    """Remove the rebuildable task definition fields."""

    with op.batch_alter_table("background_jobs", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_background_jobs_payload_json",
            type_="check",
        )
        batch_op.drop_constraint(
            "ck_background_jobs_task_version",
            type_="check",
        )
        batch_op.drop_column("payload_json")
        batch_op.drop_column("task_version")
