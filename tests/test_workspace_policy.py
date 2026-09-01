from collections.abc import Iterator
from datetime import datetime, timezone
import logging
import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session

from backend.app import services
from backend.app.database import Base
from backend.app.document_indexer import index_workspace_documents
from backend.app.models import (
    ChunkRecord,
    DocumentRecord,
    FileEntry,
    Workspace,
    WorkspacePolicyAuditEvent,
    WorkspacePolicyRecord,
)
from backend.app.operation_plan import (
    FilePrecondition,
    OperationPlan,
    OperationPlanItem,
    OperationReason,
)
from backend.app.read_tools import (
    build_knowledge_search_tool,
    build_list_directory_tool,
    build_search_files_tool,
)
from backend.app import safe_execution
from backend.app.path_policy import (
    WorkspacePolicy,
    WorkspacePolicyPersistenceError,
    parse_workspace_policy,
)
from backend.app.services import (
    WorkspacePolicyError,
    WorkspacePolicyErrorCode,
    create_workspace,
    load_workspace_policy,
    scan_workspace,
    update_workspace_policy,
)
from backend.app.safe_execution import SafeExecutionError, SafeExecutionErrorCode


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'workspace-policy.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)
    yield test_engine
    test_engine.dispose()


def test_create_workspace_persists_backward_compatible_default_policy(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "created-workspace"
    workspace_root.mkdir()

    with Session(engine) as session:
        workspace = create_workspace(
            session,
            "新建策略工作区",
            str(workspace_root),
        )
        record = session.get(WorkspacePolicyRecord, workspace.id)

    assert record is not None
    assert record.policy_revision == 0
    assert record.user_denylist_json == "[]"
    assert record.ignore_patterns_json == "[]"
    with Session(engine) as session:
        assert load_workspace_policy(session, workspace.id) == WorkspacePolicy()


def test_policy_update_uses_revision_and_writes_rule_diff_audit(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "updated-workspace"
    workspace_root.mkdir()
    with Session(engine) as session:
        workspace = create_workspace(
            session,
            "更新策略工作区",
            str(workspace_root),
        )
        updated = update_workspace_policy(
            session,
            workspace.id,
            WorkspacePolicy(
                policy_revision=1,
                proposal_enabled=False,
                user_denylist=("private/data",),
                ignore_patterns=("*.tmp",),
            ),
            actor="admin",
            source="local_api",
        )
        record = session.get(WorkspacePolicyRecord, workspace.id)
        assert record is not None
        audits = session.execute(
            select(WorkspacePolicyAuditEvent)
        ).scalars().all()

    assert updated.policy_revision == 1
    assert record is not None and record.policy_revision == 1
    assert record.proposal_enabled is False
    assert json.loads(record.user_denylist_json) == ["private/data"]
    assert json.loads(record.ignore_patterns_json) == ["*.tmp"]
    assert len(audits) == 1
    assert audits[0].previous_revision == 0
    assert audits[0].next_revision == 1
    assert json.loads(audits[0].added_rules_json) == {
        "user_denylist": ["private/data"],
        "ignore_patterns": ["*.tmp"],
    }
    assert audits[0].result == "succeeded"


def test_policy_update_rejects_stale_revision_without_changing_record(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "stale-policy-workspace"
    workspace_root.mkdir()
    with Session(engine) as session:
        workspace = create_workspace(
            session,
            "过期策略工作区",
            str(workspace_root),
        )
        policy = WorkspacePolicy(policy_revision=1, read_enabled=False)
        update_workspace_policy(
            session,
            workspace.id,
            policy,
            actor="admin",
            source="local_api",
        )

        with pytest.raises(WorkspacePolicyError) as captured:
            update_workspace_policy(
                session,
                workspace.id,
                policy,
                actor="admin",
                source="local_api",
            )
        record = session.get(WorkspacePolicyRecord, workspace.id)

    assert captured.value.code is WorkspacePolicyErrorCode.REVISION_CONFLICT
    assert record is not None and record.policy_revision == 1


def test_policy_audit_failure_rolls_back_permission_change(
    engine: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "audit-failure-workspace"
    workspace_root.mkdir()
    with Session(engine) as session:
        workspace = create_workspace(
            session,
            "审计失败工作区",
            str(workspace_root),
        )

        def fail_audit(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise RuntimeError("audit unavailable")

        monkeypatch.setattr(
            services,
            "add_workspace_policy_audit_event",
            fail_audit,
        )
        with pytest.raises(WorkspacePolicyError) as captured:
            update_workspace_policy(
                session,
                workspace.id,
                WorkspacePolicy(policy_revision=1, read_enabled=False),
                actor="admin",
                source="local_api",
            )
        record = session.get(WorkspacePolicyRecord, workspace.id)

    assert captured.value.code is WorkspacePolicyErrorCode.AUDIT_WRITE_FAILED
    assert record is not None and record.policy_revision == 0
    assert record.read_enabled is True


def test_policy_loading_fails_closed_for_missing_or_invalid_persisted_data(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        workspace = Workspace(
            name="损坏策略工作区",
            root_path="D:/Private/DamagedPolicy",
        )
        session.add(workspace)
        session.flush()
        default_record = session.get(WorkspacePolicyRecord, workspace.id)
        assert default_record is not None
        session.delete(default_record)
        session.flush()
        with pytest.raises(WorkspacePolicyError) as missing:
            load_workspace_policy(session, workspace.id)

        session.add(
            WorkspacePolicyRecord(
                workspace_id=workspace.id,
                policy_revision=0,
                user_denylist_json="{\"not\":\"a-list\"}",
                ignore_patterns_json="[]",
            )
        )
        session.commit()
        with pytest.raises(WorkspacePolicyError) as invalid:
            load_workspace_policy(session, workspace.id)

    assert missing.value.code is WorkspacePolicyErrorCode.INVALID
    assert invalid.value.code is WorkspacePolicyErrorCode.INVALID
    with pytest.raises(WorkspacePolicyPersistenceError):
        parse_workspace_policy(
            policy_revision=0,
            read_enabled=True,
            proposal_enabled=True,
            safe_execution_enabled=True,
            user_denylist_json="not-json",
            ignore_patterns_json="[]",
        )


def test_plan_record_contains_creation_policy_snapshot() -> None:
    from backend.app.organization_planning import build_operation_plan_record

    plan = OperationPlan(
        plan_id=UUID("22222222-2222-4222-8222-222222222222"),
        workspace_id=1,
        created_at=datetime(2026, 9, 3, tzinfo=timezone.utc),
        operations=(
            OperationPlanItem(
                source_file_id=1,
                source_relative_path="inbox/report.txt",
                target_relative_path="archive/report.txt",
                source_precondition=FilePrecondition(
                    size_bytes=1,
                    mtime_ns=1,
                ),
                reason=OperationReason(
                    kind="manual_selection",
                    description="策略快照测试",
                ),
            ),
        ),
    )
    record = build_operation_plan_record(
        plan,
        workflow_id=UUID("33333333-3333-4333-8333-333333333333"),
        policy=WorkspacePolicy(
            policy_revision=4,
            user_denylist=("private",),
            ignore_patterns=("*.tmp",),
        ),
    )

    metadata = json.loads(record.metadata_json)
    assert metadata["workspace_policy"] == {
        "policy_revision": 4,
        "read_enabled": True,
        "proposal_enabled": True,
        "safe_execution_enabled": True,
        "user_denylist_json": '["private"]',
        "ignore_patterns_json": '["*.tmp"]',
    }


def test_safe_execution_fails_closed_when_policy_snapshot_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = OperationPlan.model_construct(
        plan_id=UUID("44444444-4444-4444-8444-444444444444"),
        workspace_id=1,
    )
    snapshot = WorkspacePolicy()
    current = WorkspacePolicy(policy_revision=1)
    monkeypatch.setattr(
        safe_execution,
        "load_operation_plan_policy_snapshot",
        lambda _session, _plan: snapshot,
    )
    monkeypatch.setattr(
        safe_execution,
        "load_workspace_policy",
        lambda _session, _workspace_id: current,
    )

    with pytest.raises(SafeExecutionError) as captured:
        safe_execution._validate_workspace_policy_for_execution(None, plan)  # type: ignore[arg-type]

    assert captured.value.code is SafeExecutionErrorCode.POLICY_CHANGED


def test_safe_execution_fails_closed_when_policy_disables_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = OperationPlan.model_construct(
        plan_id=UUID("55555555-5555-4555-8555-555555555555"),
        workspace_id=1,
    )
    disabled = WorkspacePolicy(safe_execution_enabled=False)
    monkeypatch.setattr(
        safe_execution,
        "load_operation_plan_policy_snapshot",
        lambda _session, _plan: disabled,
    )
    monkeypatch.setattr(
        safe_execution,
        "load_workspace_policy",
        lambda _session, _workspace_id: disabled,
    )

    with pytest.raises(SafeExecutionError) as captured:
        safe_execution._validate_workspace_policy_for_execution(None, plan)  # type: ignore[arg-type]

    assert captured.value.code is SafeExecutionErrorCode.POLICY_DISABLED


def test_scan_and_knowledge_index_use_persisted_policy_and_record_reasons(
    engine: Engine,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    workspace_root = tmp_path / "policy-index-workspace"
    workspace_root.mkdir()
    (workspace_root / "visible.md").write_text("visible", encoding="utf-8")
    (workspace_root / "private.md").write_text("private", encoding="utf-8")
    (workspace_root / "ignored.md").write_text("ignored", encoding="utf-8")

    with Session(engine) as session:
        workspace = create_workspace(
            session,
            "统一索引策略工作区",
            str(workspace_root),
        )
        update_workspace_policy(
            session,
            workspace.id,
            WorkspacePolicy(
                policy_revision=1,
                user_denylist=("private.md",),
                ignore_patterns=("ignored.md",),
            ),
            actor="admin",
            source="test",
        )

        with caplog.at_level(logging.INFO, logger="FileNest"):
            scan_result = scan_workspace(session, workspace.id)
            session.add_all(
                [
                    FileEntry(
                        workspace_id=workspace.id,
                        relative_path="private.md",
                        name="private.md",
                        extension=".md",
                        size_bytes=7,
                        mtime_ns=1,
                    ),
                    FileEntry(
                        workspace_id=workspace.id,
                        relative_path="ignored.md",
                        name="ignored.md",
                        extension=".md",
                        size_bytes=7,
                        mtime_ns=1,
                    ),
                ]
            )
            session.commit()
            index_result = index_workspace_documents(session, workspace.id)

        documents = list(
            session.scalars(
                select(DocumentRecord).where(
                    DocumentRecord.workspace_id == workspace.id,
                )
            ).all()
        )

    assert scan_result.created == 1
    assert index_result.indexed_documents == 1
    assert index_result.skipped_documents == 2
    assert [document.source_relative_path for document in documents] == [
        "visible.md"
    ]
    reasons = {
        (record.relative_path, record.ignored_reason)
        for record in caplog.records
        if hasattr(record, "relative_path")
        and hasattr(record, "ignored_reason")
    }
    assert ("private.md", "path_denylisted") in reasons
    assert ("ignored.md", "workspace_ignore") in reasons


def test_read_tools_share_persisted_denylist_and_ignore_policy(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "policy-read-workspace"
    workspace_root.mkdir()
    for name in ("visible.txt", "private.txt", "ignored.txt"):
        (workspace_root / name).write_text(name, encoding="utf-8")

    with Session(engine) as session:
        workspace = create_workspace(
            session,
            "统一读取策略工作区",
            str(workspace_root),
        )
        update_workspace_policy(
            session,
            workspace.id,
            WorkspacePolicy(
                policy_revision=1,
                user_denylist=("private.txt",),
                ignore_patterns=("ignored.txt",),
            ),
            actor="admin",
            source="test",
        )

        file_entries = [
            FileEntry(
                workspace_id=workspace.id,
                relative_path=name,
                name=name,
                extension=".txt",
                size_bytes=len(name),
                mtime_ns=1,
            )
            for name in ("visible.txt", "private.txt", "ignored.txt")
        ]
        session.add_all(file_entries)
        session.flush()
        for file_entry in file_entries:
            document_id = str(uuid4())
            session.add(
                DocumentRecord(
                    document_id=document_id,
                    workspace_id=workspace.id,
                    file_entry_id=file_entry.id,
                    source_relative_path=file_entry.relative_path,
                    ingest_status="indexed",
                    source_format="text",
                    normalized_text="shared evidence",
                )
            )
            session.add(
                ChunkRecord(
                    chunk_id=str(uuid4()),
                    document_id=document_id,
                    file_entry_id=file_entry.id,
                    source_relative_path=file_entry.relative_path,
                    chunk_index=0,
                    text="shared evidence",
                    start_offset=0,
                    end_offset=14,
                    start_line=1,
                    end_line=1,
                )
            )
        session.commit()

        search_result = build_search_files_tool(session).invoke(
            {"workspace_id": workspace.id, "keyword": ".txt"}
        )
        directory_result = build_list_directory_tool(session).invoke(
            {"workspace_id": workspace.id, "relative_directory": "."}
        )
        knowledge_result = build_knowledge_search_tool(session).invoke(
            {"workspace_id": workspace.id, "query": "evidence"}
        )

    assert search_result.ok is True
    assert [item["relative_path"] for item in search_result.data["items"]] == [
        "visible.txt"
    ]
    assert directory_result.ok is True
    directory_items = {
        item["relative_path"]: item for item in directory_result.data["items"]
    }
    assert directory_items["private.txt"]["ignored_reason"] == (
        "path_denylisted"
    )
    assert directory_items["ignored.txt"]["ignored_reason"] == (
        "workspace_ignore"
    )
    assert knowledge_result.ok is True
    assert [item["source_relative_path"] for item in knowledge_result.data[
        "items"
    ]] == ["visible.txt"]
