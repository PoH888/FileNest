from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from functools import partial
from pathlib import Path
import sys
from uuid import uuid4

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database import Base
from backend.app.document_chunker import chunk_document
from backend.app.document_contracts import Document
from backend.app.mcp_server import FileNestMCPServer
from backend.app.models import (
    ApprovalRequest,
    ChunkRecord,
    DocumentRecord,
    FileEntry,
    OperationExecution,
    Workspace,
)
from backend.app.safe_execution import (
    SafeExecutionRequest,
    execute_safe_operation_plan,
)
from backend.app.services import (
    OperationPlanApprovalError,
    OperationPlanApprovalErrorCode,
    validate_operation_plan,
)
from backend.app.workflow import WorkflowState
from backend.app.workflow_graph import open_checkpointed_workflow_graph


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'mcp-server.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _seed_knowledge_index(session: Session) -> int:
    workspace = Workspace(
        name="MCP 搜索工作区",
        root_path="D:/Private/MCPWorkspace",
    )
    session.add(workspace)
    session.flush()

    relative_path = "guides/approval.md"
    text = "第一行：审批流程\n第二行：批准后才能移动文件。"
    file_entry = FileEntry(
        workspace_id=workspace.id,
        relative_path=relative_path,
        name=Path(relative_path).name,
        extension=Path(relative_path).suffix,
        size_bytes=len(text.encode("utf-8")),
        mtime_ns=1_800_000_000_000_000_000,
    )
    session.add(file_entry)
    session.flush()

    document = Document(
        document_id=uuid4(),
        workspace_id=workspace.id,
        file_entry_id=file_entry.id,
        source_relative_path=relative_path,
        source_format="markdown",
        normalized_text=text,
        source_version=f"{1:064x}",
        source_updated_at="2026-09-01T00:00:00+00:00",
    )
    session.add(DocumentRecord.from_contract(document))
    session.add_all(
        ChunkRecord.from_contract(chunk)
        for chunk in chunk_document(document)
    )
    session.commit()
    return workspace.id


def _server(engine: Engine) -> FileNestMCPServer:
    return FileNestMCPServer(sessionmaker(bind=engine, expire_on_commit=False))


def _proposal_server(
    engine: Engine,
    checkpoint_path: Path,
) -> FileNestMCPServer:
    return FileNestMCPServer(
        sessionmaker(bind=engine, expire_on_commit=False),
        workflow_graph_factory=lambda session: open_checkpointed_workflow_graph(
            checkpoint_path,
            operation_plan_validator=partial(validate_operation_plan, session),
        ),
    )


def _seed_operation_workspace(
    session: Session,
    workspace_root: Path,
) -> tuple[int, int]:
    source_path = workspace_root / "inbox" / "report.txt"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("proposal source", encoding="utf-8")
    (workspace_root / "archive").mkdir(parents=True)

    workspace = Workspace(name="MCP 提案工作区", root_path=str(workspace_root))
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
    return workspace.id, file_entry.id


def _disk_snapshot(
    workspace_root: Path,
) -> dict[str, tuple[str, bytes | None]]:
    return {
        path.relative_to(workspace_root).as_posix(): (
            "file" if path.is_file() else "directory",
            path.read_bytes() if path.is_file() else None,
        )
        for path in workspace_root.rglob("*")
    }


def test_mcp_server_exposes_search_as_a_read_only_tool(engine: Engine) -> None:
    with Session(engine) as session:
        workspace_id = _seed_knowledge_index(session)

    server = _server(engine)
    listed = asyncio.run(server.list_tools())
    result = asyncio.run(
        server.call_tool(
            "search_files",
            {"workspace_id": workspace_id, "keyword": "approval"},
        )
    )

    assert [tool.name for tool in listed.tools] == [
        "search_files",
        "knowledge_search",
        "create_operation_proposal",
    ]
    assert all(
        tool.annotations is not None
        and tool.annotations.read_only_hint is True
        and tool.annotations.destructive_hint is False
        for tool in listed.tools[:2]
    )
    proposal_tool = listed.tools[2]
    assert proposal_tool.annotations is not None
    assert proposal_tool.annotations.read_only_hint is False
    assert proposal_tool.annotations.destructive_hint is False
    assert proposal_tool.input_schema["additionalProperties"] is False
    assert result.is_error is False
    assert result.structured_content["ok"] is True
    assert result.structured_content["data"]["items"][0]["relative_path"] == (
        "guides/approval.md"
    )
    assert "D:/Private" not in result.model_dump_json()


def test_mcp_server_exposes_traceable_knowledge_search(engine: Engine) -> None:
    with Session(engine) as session:
        workspace_id = _seed_knowledge_index(session)

    result = asyncio.run(
        _server(engine).call_tool(
            "knowledge_search",
            {"workspace_id": workspace_id, "query": "审批"},
        )
    )

    item = result.structured_content["data"]["items"][0]
    assert result.is_error is False
    assert item["source_relative_path"] == "guides/approval.md"
    assert item["start_line"] == 1
    assert item["end_line"] == 2
    assert "D:/Private" not in result.model_dump_json()


def test_mcp_server_rejects_unregistered_write_tool(engine: Engine) -> None:
    result = asyncio.run(
        _server(engine).call_tool(
            "delete_file",
            {"path": "guides/approval.md"},
        )
    )

    assert result.is_error is True
    assert result.structured_content == {
        "ok": False,
        "data": None,
        "error": {
            "code": "unknown_tool",
            "message": "请求的工具未注册",
            "details": {},
        },
    }


def test_mcp_server_creates_waiting_proposal_without_disk_mutation(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "proposal-workspace"
    checkpoint_path = tmp_path / "proposal-checkpoints.sqlite"
    with Session(engine) as session:
        workspace_id, file_id = _seed_operation_workspace(
            session,
            workspace_root,
        )

    before = _disk_snapshot(workspace_root)
    result = asyncio.run(
        _proposal_server(engine, checkpoint_path).call_tool(
            "create_operation_proposal",
            {
                "workspace_id": workspace_id,
                "target_directories": ["archive"],
                "selections": [
                    {
                        "source_file_id": file_id,
                        "target_directory": "archive",
                    }
                ],
            },
        )
    )

    assert result.is_error is False
    assert result.structured_content["ok"] is True
    proposal = result.structured_content["data"]
    assert proposal["approval_status"] == "WAITING_APPROVAL"
    assert proposal["workflow"]["status"] == "waiting"
    assert proposal["workflow"]["wait_reason_code"] == (
        "human_approval_required"
    )
    assert (
        proposal["workflow"]["operation_plan"]["operations"][0][
            "target_relative_path"
        ]
        == "archive/report.txt"
    )
    assert str(workspace_root) not in result.model_dump_json()
    assert _disk_snapshot(workspace_root) == before

    with Session(engine) as session:
        approval = session.scalar(select(ApprovalRequest))
        assert approval is not None
        assert approval.id == proposal["approval_id"]
        assert approval.status == "WAITING_APPROVAL"


def test_mcp_server_rejects_invalid_proposal_without_persisting(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "invalid-proposal-workspace"
    with Session(engine) as session:
        workspace_id, file_id = _seed_operation_workspace(
            session,
            workspace_root,
        )
    before = _disk_snapshot(workspace_root)

    result = asyncio.run(
        _server(engine).call_tool(
            "create_operation_proposal",
            {
                "workspace_id": workspace_id,
                "target_directories": ["archive"],
                "selections": [
                    {
                        "source_file_id": file_id,
                        "target_directory": "archive",
                    }
                ],
                "unexpected": True,
            },
        )
    )

    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "invalid_arguments"
    assert _disk_snapshot(workspace_root) == before
    with Session(engine) as session:
        assert session.scalar(select(ApprovalRequest)) is None


def test_mcp_proposal_cannot_bypass_path_policy(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "policy-workspace"
    outside_path = tmp_path / "outside.txt"
    outside_path.write_text("outside evidence", encoding="utf-8")
    with Session(engine) as session:
        workspace_id, file_id = _seed_operation_workspace(
            session,
            workspace_root,
        )
        file_entry = session.get(FileEntry, file_id)
        assert file_entry is not None
        file_entry.relative_path = "../outside.txt"
        session.commit()

    workspace_before = _disk_snapshot(workspace_root)
    outside_before = outside_path.read_bytes()
    result = asyncio.run(
        _proposal_server(
            engine,
            tmp_path / "policy-checkpoints.sqlite",
        ).call_tool(
            "create_operation_proposal",
            {
                "workspace_id": workspace_id,
                "target_directories": ["archive"],
                "selections": [
                    {
                        "source_file_id": file_id,
                        "target_directory": "archive",
                    }
                ],
            },
        )
    )

    assert result.is_error is True
    assert result.structured_content["error"]["code"] == (
        "organization_plan_unavailable"
    )
    assert str(outside_path) not in result.model_dump_json()
    assert _disk_snapshot(workspace_root) == workspace_before
    assert outside_path.read_bytes() == outside_before
    with Session(engine) as session:
        assert session.scalar(select(ApprovalRequest)) is None


def test_mcp_proposal_cannot_bypass_approval_or_safe_executor(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "unapproved-execution-workspace"
    with Session(engine) as session:
        workspace_id, file_id = _seed_operation_workspace(
            session,
            workspace_root,
        )
    before = _disk_snapshot(workspace_root)

    result = asyncio.run(
        _proposal_server(
            engine,
            tmp_path / "unapproved-execution-checkpoints.sqlite",
        ).call_tool(
            "create_operation_proposal",
            {
                "workspace_id": workspace_id,
                "target_directories": ["archive"],
                "selections": [
                    {
                        "source_file_id": file_id,
                        "target_directory": "archive",
                    }
                ],
            },
        )
    )
    assert result.is_error is False
    workflow = WorkflowState.model_validate(
        result.structured_content["data"]["workflow"]
    )

    with Session(engine) as session:
        with pytest.raises(OperationPlanApprovalError) as captured_error:
            execute_safe_operation_plan(
                session,
                SafeExecutionRequest(
                    workflow_id=workflow.workflow_id,
                    plan=workflow.operation_plan,
                ),
            )

        assert captured_error.value.code == (
            OperationPlanApprovalErrorCode.NOT_APPROVED
        )
        assert session.scalar(select(OperationExecution)) is None
    assert _disk_snapshot(workspace_root) == before


async def _run_stdio_client_round_trip(
    database_path: Path,
    workspace_id: int,
) -> tuple[list[str], object, object]:
    project_root = Path(__file__).resolve().parents[1]
    environment = {
        name: os.environ[name]
        for name in ("PATH", "PATHEXT", "SystemRoot", "TEMP", "TMP")
        if name in os.environ
    }
    environment["FILENEST_DATABASE_URL"] = (
        f"sqlite:///{database_path.as_posix()}"
    )
    environment["PYTHONPATH"] = str(project_root)
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "backend.app.mcp_server"],
        env=environment,
        cwd=project_root,
    )

    async with stdio_client(server_parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as client:
            await client.initialize()
            listed = await client.list_tools()
            search_result = await client.call_tool(
                "search_files",
                {"workspace_id": workspace_id, "keyword": "approval"},
            )
            rejected_result = await client.call_tool(
                "delete_file",
                {"path": "guides/approval.md"},
            )

    return [tool.name for tool in listed.tools], search_result, rejected_result


def test_mcp_stdio_client_round_trip_preserves_server_boundary(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "mcp-stdio.db"
    test_engine = create_engine(
        f"sqlite:///{database_path.as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)
    try:
        with Session(test_engine) as session:
            workspace_id = _seed_knowledge_index(session)

        listed_names, search_result, rejected_result = asyncio.run(
            _run_stdio_client_round_trip(database_path, workspace_id)
        )

        assert listed_names == [
            "search_files",
            "knowledge_search",
            "create_operation_proposal",
        ]
        assert search_result.is_error is False
        assert search_result.structured_content["ok"] is True
        assert rejected_result.is_error is True
        assert rejected_result.structured_content["error"]["code"] == (
            "unknown_tool"
        )
    finally:
        test_engine.dispose()
