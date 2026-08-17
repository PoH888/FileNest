from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import (
    ApprovalRequest,
    FileEntry,
    OperationPlanRecord,
    Workspace,
)
from backend.app.proposal_tools import build_propose_move_tool
from backend.app.workflow_graph import open_checkpointed_workflow_graph


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'propose-move-tool.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _seed_workspace(
    session: Session,
    workspace_root: Path,
    *,
    name: str,
) -> tuple[Workspace, FileEntry]:
    source_path = workspace_root / "inbox" / "quarterly-report.txt"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"proposal tool")
    (workspace_root / "reports" / "quarterly").mkdir(parents=True)

    workspace = Workspace(name=name, root_path=str(workspace_root))
    session.add(workspace)
    session.flush()
    metadata = source_path.stat()
    file_entry = FileEntry(
        workspace_id=workspace.id,
        relative_path="inbox/quarterly-report.txt",
        name="quarterly-report.txt",
        extension=".txt",
        size_bytes=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
    )
    session.add(file_entry)
    session.commit()
    return workspace, file_entry


def _disk_snapshot(workspace_root: Path) -> dict[str, tuple[str, bytes | None]]:
    return {
        path.relative_to(workspace_root).as_posix(): (
            "file" if path.is_file() else "directory",
            path.read_bytes() if path.is_file() else None,
        )
        for path in workspace_root.rglob("*")
    }


def test_propose_move_creates_waiting_plan_without_disk_write(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    checkpoint_path = tmp_path / "workflow-checkpoints.sqlite"

    with Session(engine) as session:
        workspace, file_entry = _seed_workspace(
            session,
            workspace_root,
            name="移动 Proposal 工作区",
        )
        before = _disk_snapshot(workspace_root)
        with open_checkpointed_workflow_graph(checkpoint_path) as graph:
            tool = build_propose_move_tool(
                session,
                graph,
                workflow_id_factory=lambda: UUID(
                    "11111111-1111-4111-8111-111111111111"
                ),
                plan_id_factory=lambda: UUID(
                    "22222222-2222-4222-8222-222222222222"
                ),
            )
            result = tool.invoke(
                {
                    "workspace_id": workspace.id,
                    "source_file_id": file_entry.id,
                    "destination": "reports/quarterly",
                }
            )

        assert result.model_dump() == {
            "ok": True,
            "data": {
                "plan_id": "22222222-2222-4222-8222-222222222222",
            },
            "error": None,
        }
        plan = session.get(
            OperationPlanRecord,
            "22222222-2222-4222-8222-222222222222",
        )
        approval = session.scalar(select(ApprovalRequest))
        assert plan is not None
        assert plan.status == "WAITING_APPROVAL"
        assert plan.workspace_id == workspace.id
        assert len(plan.items) == 1
        assert plan.items[0].source_file_id == file_entry.id
        assert plan.items[0].target_relative_path == (
            "reports/quarterly/quarterly-report.txt"
        )
        assert approval is not None
        assert approval.status == "WAITING_APPROVAL"
        assert _disk_snapshot(workspace_root) == before


def test_propose_move_rejects_file_outside_requested_workspace(
    engine: Engine,
    tmp_path: Path,
) -> None:
    with Session(engine) as session:
        source_workspace, file_entry = _seed_workspace(
            session,
            tmp_path / "source-workspace",
            name="源工作区",
        )
        other_workspace, _ = _seed_workspace(
            session,
            tmp_path / "other-workspace",
            name="其他工作区",
        )
        source_before = _disk_snapshot(Path(source_workspace.root_path))
        other_before = _disk_snapshot(Path(other_workspace.root_path))
        checkpoint_path = tmp_path / "scope-checkpoints.sqlite"

        with open_checkpointed_workflow_graph(checkpoint_path) as graph:
            tool = build_propose_move_tool(session, graph)
            result = tool.invoke(
                {
                    "workspace_id": other_workspace.id,
                    "source_file_id": file_entry.id,
                    "destination": "reports/quarterly",
                }
            )

        assert result.ok is False
        assert result.error is not None
        assert result.error.code == "file_not_found"
        assert session.scalar(select(OperationPlanRecord)) is None
        assert session.scalar(select(ApprovalRequest)) is None
        assert _disk_snapshot(Path(source_workspace.root_path)) == source_before
        assert _disk_snapshot(Path(other_workspace.root_path)) == other_before
