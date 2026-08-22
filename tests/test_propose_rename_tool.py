from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import ApprovalRequest, FileEntry, OperationPlanRecord, Workspace
from backend.app.proposal_tools import build_propose_rename_tool
from backend.app.workflow_graph import open_checkpointed_workflow_graph


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'propose-rename-tool.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _seed_workspace(
    session: Session,
    workspace_root: Path,
) -> tuple[Workspace, FileEntry]:
    source_path = workspace_root / "inbox" / "quarterly-report.txt"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"rename proposal")

    workspace = Workspace(name="重命名 Proposal 工作区", root_path=str(workspace_root))
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


def test_propose_rename_creates_proposal_without_renaming_file(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    source_path = workspace_root / "inbox" / "quarterly-report.txt"
    target_path = workspace_root / "inbox" / "quarterly-report-final.txt"
    checkpoint_path = tmp_path / "workflow-checkpoints.sqlite"
    workflow_id = UUID("66666666-6666-4666-8666-666666666666")
    plan_id = UUID("77777777-7777-4777-8777-777777777777")

    with Session(engine) as session:
        workspace, file_entry = _seed_workspace(session, workspace_root)
        source_before = source_path.read_bytes()
        with open_checkpointed_workflow_graph(checkpoint_path) as graph:
            tool = build_propose_rename_tool(
                session,
                graph,
                workflow_id_factory=lambda: workflow_id,
                plan_id_factory=lambda: plan_id,
            )

            result = tool.invoke(
                {
                    "workspace_id": workspace.id,
                    "source_file_id": file_entry.id,
                    "new_name": "quarterly-report-final.txt",
                }
            )

        assert result.model_dump() == {
            "ok": True,
            "data": {
                "plan_id": str(plan_id),
            },
            "error": None,
        }
        plan = session.get(OperationPlanRecord, str(plan_id))
        approval = session.scalar(select(ApprovalRequest))
        assert plan is not None
        assert plan.operation_type == "rename"
        assert plan.status == "WAITING_APPROVAL"
        assert plan.items[0].operation_type == "rename"
        assert plan.items[0].target_relative_path == (
            "inbox/quarterly-report-final.txt"
        )
        assert approval is not None
        assert approval.workflow_id == str(workflow_id)
        assert approval.status == "WAITING_APPROVAL"
        assert source_path.exists()
        assert source_path.read_bytes() == source_before
        assert not target_path.exists()


def test_propose_rename_rejects_path_in_new_name(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    source_path = workspace_root / "inbox" / "quarterly-report.txt"
    checkpoint_path = tmp_path / "workflow-checkpoints.sqlite"

    with Session(engine) as session:
        workspace, file_entry = _seed_workspace(session, workspace_root)
        with open_checkpointed_workflow_graph(checkpoint_path) as graph:
            tool = build_propose_rename_tool(session, graph)

            result = tool.invoke(
                {
                    "workspace_id": workspace.id,
                    "source_file_id": file_entry.id,
                    "new_name": "../outside.txt",
                }
            )

        assert result.ok is False
        assert result.error is not None
        assert result.error.code == "invalid_arguments"
        assert source_path.exists()
        assert not (tmp_path / "outside.txt").exists()
