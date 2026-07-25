from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from backend.app.operation_plan import (
    ContentHash,
    FilePrecondition,
    OperationPlan,
    OperationPlanItem,
    OperationReason,
    OperationRisk,
)


PLAN_ID = UUID("f621e2d9-2e18-4632-a288-154ead632a17")
CREATED_AT = datetime(2026, 8, 30, 8, 0, tzinfo=timezone.utc)


def _operation(**overrides: object) -> OperationPlanItem:
    values: dict[str, object] = {
        "source_file_id": 7,
        "source_relative_path": "inbox/quarterly-report.pdf",
        "target_relative_path": "documents/reports/quarterly-report.pdf",
        "source_precondition": FilePrecondition(
            size_bytes=4096,
            mtime_ns=1_777_777_777_000_000_000,
        ),
        "reason": OperationReason(
            kind="matched_candidate",
            description="候选目录与文件名最匹配",
            match_score=96,
        ),
        "risks": [
            OperationRisk(
                code="source_will_move",
                level="medium",
                description="源路径将在执行后失效",
            )
        ],
    }
    values.update(overrides)
    return OperationPlanItem(**values)


def test_operation_plan_describes_determined_immutable_operations() -> None:
    operation = _operation()
    plan = OperationPlan(
        plan_id=PLAN_ID,
        workspace_id=3,
        created_at=CREATED_AT,
        operations=[operation],
    )

    assert plan.schema_version == 1
    assert plan.operations == (operation,)
    assert plan.operations[0].operation_type == "move"
    assert plan.operations[0].source_precondition.size_bytes == 4096
    assert plan.operations[0].source_precondition.content_hash is None
    assert plan.operations[0].reason.match_score == 96
    assert plan.operations[0].risks[0].code == "source_will_move"
    assert "execute" not in plan.model_dump_json()

    with pytest.raises(ValidationError):
        plan.workspace_id = 4


@pytest.mark.parametrize(
    "path",
    ["../outside.txt", "D:/outside.txt", "folder\\file.txt", " file.txt", "."],
)
def test_operation_item_rejects_unsafe_or_unnormalized_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        _operation(target_relative_path=path)


def test_operation_item_rejects_same_path_and_duplicate_risks() -> None:
    with pytest.raises(ValidationError, match="must be different"):
        _operation(target_relative_path="inbox/quarterly-report.pdf")

    duplicate_risk = OperationRisk(
        code="source_will_move",
        level="low",
        description="重复风险",
    )
    with pytest.raises(ValidationError, match="risk codes must be unique"):
        _operation(risks=[duplicate_risk, duplicate_risk])


def test_file_precondition_records_optional_sha256() -> None:
    content_hash = ContentHash(digest="a" * 64)
    precondition = FilePrecondition(
        size_bytes=8192,
        mtime_ns=1_888_888_888_000_000_000,
        content_hash=content_hash,
    )

    assert precondition.content_hash == content_hash
    assert precondition.content_hash.algorithm == "sha256"


@pytest.mark.parametrize(
    ("field", "value"),
    [("size_bytes", -1), ("mtime_ns", -1)],
)
def test_file_precondition_rejects_negative_metadata(
    field: str,
    value: int,
) -> None:
    values = {"size_bytes": 1, "mtime_ns": 1}
    values[field] = value

    with pytest.raises(ValidationError):
        FilePrecondition(**values)


@pytest.mark.parametrize("digest", ["a" * 63, "A" * 64, "g" * 64])
def test_content_hash_rejects_invalid_sha256_digest(digest: str) -> None:
    with pytest.raises(ValidationError):
        ContentHash(digest=digest)


def test_content_hash_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValidationError):
        ContentHash(algorithm="md5", digest="a" * 64)


def test_operation_item_requires_source_precondition() -> None:
    operation_values = _operation().model_dump()
    operation_values.pop("source_precondition")

    with pytest.raises(ValidationError):
        OperationPlanItem(**operation_values)


def test_reason_requires_score_only_for_ranked_candidate() -> None:
    with pytest.raises(ValidationError, match="requires match_score"):
        OperationReason(
            kind="matched_candidate",
            description="来自候选排序",
        )

    with pytest.raises(ValidationError, match="must not include match_score"):
        OperationReason(
            kind="manual_selection",
            description="由用户明确选择",
            match_score=80,
        )


def test_plan_rejects_duplicate_sources_targets_and_unknown_fields() -> None:
    first = _operation()
    same_source = _operation(
        target_relative_path="archive/quarterly-report.pdf",
    )
    with pytest.raises(ValidationError, match="unique source_file_ids"):
        OperationPlan(
            plan_id=PLAN_ID,
            workspace_id=3,
            created_at=CREATED_AT,
            operations=[first, same_source],
        )

    same_target = _operation(source_file_id=8)
    with pytest.raises(ValidationError, match="unique target_relative_paths"):
        OperationPlan(
            plan_id=PLAN_ID,
            workspace_id=3,
            created_at=CREATED_AT,
            operations=[first, same_target],
        )

    with pytest.raises(ValidationError):
        OperationPlan(
            plan_id=PLAN_ID,
            workspace_id=3,
            created_at=CREATED_AT,
            operations=[first],
            execute=True,
        )


def test_plan_requires_timezone_aware_creation_time() -> None:
    with pytest.raises(ValidationError, match="must include a timezone"):
        OperationPlan(
            plan_id=PLAN_ID,
            workspace_id=3,
            created_at=datetime(2026, 8, 30, 8, 0),
            operations=[_operation()],
        )
