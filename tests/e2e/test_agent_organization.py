from collections.abc import Iterator
from pathlib import Path
from time import monotonic, sleep

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_api import ReadOnlyAgentRunExecutor, get_agent_run_executor
from backend.app.database import Base, get_session
from backend.app.fake_model_client import FakeModelClient
from backend.app.main import app
from backend.app.model_client import ModelMessage, ModelResponse, ModelToolCall
from backend.app.models import (
    ApprovalRequest,
    FileEntry,
    OperationPlanRecord,
    Workspace,
)
from backend.app.tool_contracts import ToolResult


@pytest.fixture
def agent_organization_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, sessionmaker[Session]]]:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'agent-organization.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    def override_get_session() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    with TestClient(app) as client:
        yield client, session_factory

    app.dependency_overrides.clear()
    engine.dispose()


def _seed_workspace(
    session_factory: sessionmaker[Session],
    workspace_root: Path,
) -> tuple[int, int, Path, Path, bytes]:
    source_path = workspace_root / "downloads" / "calculus-course.pdf"
    target_path = (
        workspace_root / "subjects" / "mathematics" / source_path.name
    )
    source_path.parent.mkdir(parents=True)
    target_path.parent.mkdir(parents=True)
    source_content = b"agent organization e2e"
    source_path.write_bytes(source_content)

    with session_factory() as session:
        workspace = Workspace(
            name="Agent 整理 E2E 工作区",
            root_path=str(workspace_root),
        )
        session.add(workspace)
        session.flush()
        metadata = source_path.stat()
        file_entry = FileEntry(
            workspace_id=workspace.id,
            relative_path="downloads/calculus-course.pdf",
            name="calculus-course.pdf",
            extension=".pdf",
            size_bytes=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
        )
        session.add(file_entry)
        session.commit()
        return (
            workspace.id,
            file_entry.id,
            source_path,
            target_path,
            source_content,
        )


def _tool_call_response(
    *,
    call_id: str,
    name: str,
    arguments: dict[str, object],
) -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(
            role="assistant",
            tool_calls=(
                ModelToolCall(
                    id=call_id,
                    name=name,
                    arguments=arguments,
                ),
            ),
        ),
        finish_reason="tool_calls",
    )


def _final_response(content: str) -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(role="assistant", content=content),
        finish_reason="stop",
    )


def _run_move_proposal(
    client: TestClient,
    session_factory: sessionmaker[Session],
    *,
    workspace_id: int,
    file_id: int,
    request_text: str,
    model_client: FakeModelClient,
) -> tuple[str, str]:
    app.dependency_overrides[get_agent_run_executor] = lambda: (
        ReadOnlyAgentRunExecutor(lambda: model_client)
    )

    response = client.post(
        "/api/v1/agent-runs",
        json={
            "workspace_id": workspace_id,
            "request_text": request_text,
        },
    )

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    deadline = monotonic() + 5
    while True:
        state_response = client.get(f"/api/v1/agent-runs/{run_id}")
        assert state_response.status_code == 200
        if state_response.json()["status"] not in {
            "pending",
            "running",
            "waiting_approval",
        }:
            break
        if monotonic() >= deadline:
            pytest.fail("Agent Run did not reach a terminal state")
        sleep(0.01)

    search_result = ToolResult.model_validate_json(
        model_client.calls[1].messages[-1].content
    )
    proposal_result = ToolResult.model_validate_json(
        model_client.calls[2].messages[-1].content
    )
    assert search_result.ok is True
    assert proposal_result.ok is True
    assert proposal_result.data is not None
    plan_id = str(proposal_result.data["plan_id"])
    with session_factory() as session:
        plan = session.get(OperationPlanRecord, plan_id)
        approval = session.scalar(
            select(ApprovalRequest).where(ApprovalRequest.plan_id == plan_id)
        )
        assert plan is not None
        assert plan.workspace_id == workspace_id
        assert plan.items[0].source_file_id == file_id
        assert plan.status == "WAITING_APPROVAL"
        assert approval is not None
        assert approval.status == "WAITING_APPROVAL"
        return plan_id, approval.workflow_id


def _move_model_client(workspace_id: int, file_id: int) -> FakeModelClient:
    return FakeModelClient(
        [
            _tool_call_response(
                call_id="call_search_course",
                name="search_files",
                arguments={
                    "workspace_id": workspace_id,
                    "keyword": "course",
                },
            ),
            _tool_call_response(
                call_id="call_propose_move",
                name="propose_move",
                arguments={
                    "workspace_id": workspace_id,
                    "source_file_id": file_id,
                    "destination": "subjects/mathematics",
                },
            ),
            _final_response("已生成待审批的课程整理计划。"),
        ]
    )


def test_agent_proposal_approval_execution_updates_disk_state(
    agent_organization_client: tuple[TestClient, sessionmaker[Session]],
    tmp_path: Path,
) -> None:
    client, session_factory = agent_organization_client
    workspace_id, file_id, source_path, target_path, source_content = (
        _seed_workspace(session_factory, tmp_path / "success-workspace")
    )
    plan_id, workflow_id = _run_move_proposal(
        client,
        session_factory,
        workspace_id=workspace_id,
        file_id=file_id,
        request_text="把下载目录里的课程 PDF 按科目整理",
        model_client=_move_model_client(workspace_id, file_id),
    )

    assert source_path.exists()
    assert not target_path.exists()

    approve_response = client.post(
        f"/api/v1/workflows/{workflow_id}/decisions",
        json={
            "action": "approve",
            "expected_plan_id": plan_id,
        },
    )
    assert approve_response.status_code == 200
    assert approve_response.json()["approval_status"] == "APPROVED"

    execute_response = client.post(
        f"/api/v1/workflows/{workflow_id}/execute"
    )
    assert execute_response.status_code == 200
    assert execute_response.json()["status"] == "COMPLETED"
    assert not source_path.exists()
    assert target_path.read_bytes() == source_content


def test_agent_proposal_cannot_execute_before_approval(
    agent_organization_client: tuple[TestClient, sessionmaker[Session]],
    tmp_path: Path,
) -> None:
    client, session_factory = agent_organization_client
    workspace_id, file_id, source_path, target_path, source_content = (
        _seed_workspace(session_factory, tmp_path / "unapproved-workspace")
    )
    plan_id, workflow_id = _run_move_proposal(
        client,
        session_factory,
        workspace_id=workspace_id,
        file_id=file_id,
        request_text="整理课程 PDF",
        model_client=_move_model_client(workspace_id, file_id),
    )

    execute_response = client.post(
        f"/api/v1/workflows/{workflow_id}/execute"
    )

    assert execute_response.status_code == 409
    assert execute_response.json() == {
        "detail": {
            "code": "organization_workflow_not_ready",
            "message": "工作流尚未获得批准，不能执行文件操作。",
        }
    }
    assert source_path.exists()
    assert source_path.read_bytes() == source_content
    assert not target_path.exists()
    with session_factory() as session:
        plan = session.get(OperationPlanRecord, plan_id)
        assert plan is not None
        assert plan.status == "WAITING_APPROVAL"
