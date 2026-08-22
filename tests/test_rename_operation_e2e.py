from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

import backend.app.safe_execution as safe_execution_module
from backend.app.database import Base
from backend.app.models import (
    FileEntry,
    OperationExecutionItem,
    Workspace,
)
from backend.app.proposal_tools import build_propose_rename_tool
from backend.app.repositories import (
    find_operation_execution_items,
    get_file_entry_by_id,
    get_operation_execution_by_workflow_id,
)
from backend.app.safe_execution import (
    SafeExecutionRequest,
    execute_safe_operation_plan,
    undo_safe_operation_execution,
)
from backend.app.services import (
    approve_operation_plan,
    get_operation_plan,
)
from backend.app.workflow_graph import open_checkpointed_workflow_graph


WORKFLOW_ID = UUID("88888888-8888-4888-8888-888888888888")
PLAN_ID = UUID("99999999-9999-4999-8999-999999999999")


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'rename-operation.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _seed_workspace(
    session: Session,
    workspace_root: Path,
) -> tuple[Workspace, FileEntry]:
    source_path = workspace_root / "inbox" / "report.txt"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"rename operation")

    workspace = Workspace(
        name="Rename E2E 工作区",
        root_path=str(workspace_root),
    )
    session.add(workspace)
    session.flush()
    metadata = source_path.stat()
    file_entry = FileEntry(
        workspace_id=workspace.id,
        relative_path="inbox/report.txt",
        name="report.txt",
        extension=".txt",
        size_bytes=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
    )
    session.add(file_entry)
    session.commit()
    return workspace, file_entry


def _approved_rename_request(
    session: Session,
    tmp_path: Path,
) -> tuple[Path, SafeExecutionRequest]:
    workspace_root = tmp_path / "workspace"
    checkpoint_path = tmp_path / "workflow-checkpoints.sqlite"
    workspace, file_entry = _seed_workspace(session, workspace_root)
    with open_checkpointed_workflow_graph(checkpoint_path) as graph:
        tool = build_propose_rename_tool(
            session,
            graph,
            workflow_id_factory=lambda: WORKFLOW_ID,
            plan_id_factory=lambda: PLAN_ID,
        )
        result = tool.invoke(
            {
                "workspace_id": workspace.id,
                "source_file_id": file_entry.id,
                "new_name": "report-final.txt",
            }
        )

    assert result.ok is True
    approve_operation_plan(session, WORKFLOW_ID, PLAN_ID)
    plan = get_operation_plan(session, PLAN_ID, workflow_id=WORKFLOW_ID)
    assert plan is not None
    return workspace_root, SafeExecutionRequest(
        workflow_id=WORKFLOW_ID,
        plan=plan,
    )


def test_rename_propose_approve_execute_history_and_undo(
    engine: Engine,
    tmp_path: Path,
) -> None:
    with Session(engine) as session:
        workspace_root, request = _approved_rename_request(session, tmp_path)

        execute_result = execute_safe_operation_plan(
            session,
            request,
            now=request.plan.created_at,
        )
        execution = get_operation_execution_by_workflow_id(
            session,
            str(WORKFLOW_ID),
        )
        assert execution is not None
        execution_item = find_operation_execution_items(
            session,
            execution.id,
        )[0]
        assert isinstance(execution_item, OperationExecutionItem)
        assert execute_result.status == "COMPLETED"
        assert execution_item.operation_type == "rename"
        assert execution_item.source_file_id == request.plan.operations[0].source_file_id
        assert execution_item.before_relative_path == "inbox/report.txt"
        assert execution_item.after_relative_path == "inbox/report-final.txt"
        assert execution_item.status == "COMPLETED"
        assert not (workspace_root / "inbox" / "report.txt").exists()
        assert (
            workspace_root / "inbox" / "report-final.txt"
        ).read_bytes() == b"rename operation"

        undo_result = undo_safe_operation_execution(
            session,
            WORKFLOW_ID,
            now=request.plan.created_at,
        )

        assert undo_result.status == "UNDONE"
        assert (workspace_root / "inbox" / "report.txt").read_bytes() == (
            b"rename operation"
        )
        assert not (workspace_root / "inbox" / "report-final.txt").exists()


def test_rename_execution_records_toctou_target_conflict(
    engine: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(engine) as session:
        workspace_root, request = _approved_rename_request(session, tmp_path)
        original_move = safe_execution_module.SafeFileMover.move

        def occupy_target_then_move(
            mover: safe_execution_module.SafeFileMover,
            source_path: Path,
            target_path: Path,
        ) -> Path:
            (workspace_root / target_path).write_bytes(b"racing target")
            return original_move(mover, source_path, target_path)

        monkeypatch.setattr(
            safe_execution_module.SafeFileMover,
            "move",
            occupy_target_then_move,
        )

        result = execute_safe_operation_plan(
            session,
            request,
            now=request.plan.created_at,
        )

        execution = get_operation_execution_by_workflow_id(
            session,
            str(WORKFLOW_ID),
        )
        assert execution is not None
        execution_item = find_operation_execution_items(
            session,
            execution.id,
        )[0]
        file_entry = get_file_entry_by_id(
            session,
            request.plan.workspace_id,
            request.plan.operations[0].source_file_id,
        )
        assert result.status == "FAILED"
        assert result.items[0].error_code == "safe_move_target_conflict"
        assert execution_item.operation_type == "rename"
        assert execution_item.status == "FAILED"
        assert file_entry is not None
        assert file_entry.relative_path == "inbox/report.txt"
        assert (workspace_root / "inbox" / "report.txt").read_bytes() == (
            b"rename operation"
        )
        assert (
            workspace_root / "inbox" / "report-final.txt"
        ).read_bytes() == b"racing target"
