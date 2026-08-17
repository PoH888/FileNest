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
from backend.app.proposal_tools import build_propose_quarantine_tool
from backend.app.workflow_graph import open_checkpointed_workflow_graph


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'propose-quarantine-tool.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _seed_workspace(
    session: Session,
    workspace_root: Path,
) -> tuple[Workspace, FileEntry]:
    source_path = workspace_root / "inbox" / "report.pdf"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"quarantine proposal")

    workspace = Workspace(
        name="隔离 Proposal 工作区",
        root_path=str(workspace_root),
    )
    session.add(workspace)
    session.flush()
    metadata = source_path.stat()
    file_entry = FileEntry(
        workspace_id=workspace.id,
        relative_path="inbox/report.pdf",
        name="report.pdf",
        extension=".pdf",
        size_bytes=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
    )
    session.add(file_entry)
    session.commit()
    return workspace, file_entry


def test_propose_quarantine_creates_waiting_plan_without_moving_file(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    quarantine_root = tmp_path / "application-quarantine"
    source_path = workspace_root / "inbox" / "report.pdf"
    checkpoint_path = tmp_path / "workflow-checkpoints.sqlite"
    workflow_id = UUID("44444444-4444-4444-8444-444444444444")
    plan_id = UUID("55555555-5555-4555-8555-555555555555")

    with Session(engine) as session:
        workspace, file_entry = _seed_workspace(session, workspace_root)
        with open_checkpointed_workflow_graph(checkpoint_path) as graph:
            tool = build_propose_quarantine_tool(
                session,
                graph,
                quarantine_root=quarantine_root,
                workflow_id_factory=lambda: workflow_id,
                plan_id_factory=lambda: plan_id,
            )
            result = tool.invoke(
                {
                    "workspace_id": workspace.id,
                    "source_file_id": file_entry.id,
                }
            )

        destination = (
            f"workspace-{workspace.id}/{plan_id}/{file_entry.id}/report.pdf"
        )
        assert result.model_dump() == {
            "ok": True,
            "data": {
                "plan_id": str(plan_id),
                "quarantine_destination": destination,
            },
            "error": None,
        }
        plan = session.get(OperationPlanRecord, str(plan_id))
        approval = session.scalar(select(ApprovalRequest))
        assert plan is not None
        assert plan.operation_type == "quarantine"
        assert plan.status == "WAITING_APPROVAL"
        assert len(plan.items) == 1
        assert plan.items[0].operation_type == "quarantine"
        assert plan.items[0].target_relative_path == destination
        assert approval is not None
        assert approval.status == "WAITING_APPROVAL"
        assert source_path.read_bytes() == b"quarantine proposal"
        assert not quarantine_root.exists()


def test_propose_quarantine_rejects_overlapping_quarantine_root(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"

    with Session(engine) as session:
        workspace, file_entry = _seed_workspace(session, workspace_root)
        checkpoint_path = tmp_path / "overlap-checkpoints.sqlite"

        with open_checkpointed_workflow_graph(checkpoint_path) as graph:
            tool = build_propose_quarantine_tool(
                session,
                graph,
                quarantine_root=workspace_root / "quarantine",
            )
            result = tool.invoke(
                {
                    "workspace_id": workspace.id,
                    "source_file_id": file_entry.id,
                }
            )

        assert result.ok is False
        assert result.error is not None
        assert result.error.code == "quarantine_roots_overlap"
        assert session.scalar(select(OperationPlanRecord)) is None
        assert session.scalar(select(ApprovalRequest)) is None
        assert (workspace_root / "inbox" / "report.pdf").exists()
        assert not (workspace_root / "quarantine").exists()
