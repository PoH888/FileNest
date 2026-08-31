"""persist the canonical payload digest used by Job idempotency

Revision ID: ac30a01b2c3d4
Revises: ab29a01b2c3d4
Create Date: 2026-09-03

"""

import hashlib
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "ac30a01b2c3d4"
down_revision: Union[str, Sequence[str], None] = "ab29a01b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Backfill the digest before making it mandatory for new Jobs."""

    op.add_column(
        "background_jobs",
        sa.Column("payload_hash", sa.String(length=64), nullable=True),
    )

    bind = op.get_bind()
    jobs = sa.table(
        "background_jobs",
        sa.column("job_id", sa.String(length=36)),
        sa.column("payload_json", sa.Text()),
        sa.column("payload_hash", sa.String(length=64)),
    )
    rows = list(
        bind.execute(
            sa.select(jobs.c.job_id, jobs.c.payload_json)
        ).mappings()
    )
    for row in rows:
        payload_json = row["payload_json"]
        if not isinstance(payload_json, str) or not payload_json:
            raise RuntimeError("background Job payload_json is invalid")
        bind.execute(
            sa.update(jobs)
            .where(jobs.c.job_id == row["job_id"])
            .values(
                payload_hash=hashlib.sha256(
                    payload_json.encode("utf-8")
                ).hexdigest()
            )
        )

    with op.batch_alter_table("background_jobs", recreate="always") as batch_op:
        batch_op.alter_column(
            "payload_hash",
            existing_type=sa.String(length=64),
            nullable=False,
        )
        batch_op.create_check_constraint(
            "ck_background_jobs_payload_hash",
            "length(payload_hash) = 64 AND payload_hash = lower(payload_hash)",
        )


def downgrade() -> None:
    """Remove the derived digest while retaining the task definition itself."""

    with op.batch_alter_table("background_jobs", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_background_jobs_payload_hash",
            type_="check",
        )
        batch_op.drop_column("payload_hash")
