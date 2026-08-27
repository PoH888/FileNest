from collections.abc import Iterator
from pathlib import Path
from threading import Event
from time import monotonic, sleep

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

import backend.app.safe_execution as safe_execution_module
from backend.app.agent_api import (
    AgentRunResponse,
    ReadOnlyAgentRunExecutor,
    get_agent_run_executor,
)
from backend.app.agent_observability import SqlAlchemyAgentRunRecorder
from backend.app.agent_recovery import recover_unfinished_agent_runs
from backend.app.database import Base, get_session
from backend.app.fake_model_client import FakeModelClient
from backend.app.main import app
from backend.app.model_client import ModelMessage, ModelResponse, ModelToolCall
from backend.app.models import (
    AgentRun,
    ApprovalRequest,
    FileEntry,
    OperationExecution,
    OperationExecutionItem,
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


def _wait_for_agent_run_status(
    client: TestClient,
    run_id: int,
    expected_status: str,
) -> dict[str, object]:
    deadline = monotonic() + 5
    while True:
        response = client.get(f"/api/v1/agent-runs/{run_id}")
        assert response.status_code == 200
        state = response.json()
        if state["status"] == expected_status:
            return state
        if monotonic() >= deadline:
            pytest.fail(
                f"Agent Run did not reach status {expected_status!r}"
            )
        sleep(0.01)


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


def _run_rename_proposal(
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
        assert plan.operation_type == "rename"
        assert plan.workspace_id == workspace_id
        assert plan.items[0].source_file_id == file_id
        assert plan.items[0].target_relative_path == (
            "downloads/calculus-course-final.pdf"
        )
        assert plan.status == "WAITING_APPROVAL"
        assert approval is not None
        assert approval.status == "WAITING_APPROVAL"
        return plan_id, approval.workflow_id


def _rename_model_client(workspace_id: int, file_id: int) -> FakeModelClient:
    return FakeModelClient(
        [
            _tool_call_response(
                call_id="call_search_course_for_rename",
                name="search_files",
                arguments={
                    "workspace_id": workspace_id,
                    "keyword": "course",
                },
            ),
            _tool_call_response(
                call_id="call_propose_rename",
                name="propose_rename",
                arguments={
                    "workspace_id": workspace_id,
                    "source_file_id": file_id,
                    "new_name": "calculus-course-final.pdf",
                },
            ),
            _final_response("已生成待审批的课程重命名计划。"),
        ]
    )


def _run_quarantine_proposal(
    client: TestClient,
    session_factory: sessionmaker[Session],
    *,
    workspace_id: int,
    file_id: int,
    request_text: str,
    model_client: FakeModelClient,
    quarantine_root: Path,
) -> tuple[str, str, str]:
    app.dependency_overrides[get_agent_run_executor] = lambda: (
        ReadOnlyAgentRunExecutor(
            lambda: model_client,
            quarantine_root=quarantine_root,
        )
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
    quarantine_destination = str(
        proposal_result.data["quarantine_destination"]
    )
    with session_factory() as session:
        plan = session.get(OperationPlanRecord, plan_id)
        approval = session.scalar(
            select(ApprovalRequest).where(ApprovalRequest.plan_id == plan_id)
        )
        assert plan is not None
        assert plan.operation_type == "quarantine"
        assert plan.workspace_id == workspace_id
        assert plan.items[0].source_file_id == file_id
        assert plan.items[0].target_relative_path == quarantine_destination
        assert plan.status == "WAITING_APPROVAL"
        assert approval is not None
        assert approval.status == "WAITING_APPROVAL"
        return plan_id, approval.workflow_id, quarantine_destination


def _quarantine_model_client(
    workspace_id: int,
    file_id: int,
) -> FakeModelClient:
    return FakeModelClient(
        [
            _tool_call_response(
                call_id="call_search_course_for_quarantine",
                name="search_files",
                arguments={
                    "workspace_id": workspace_id,
                    "keyword": "course",
                },
            ),
            _tool_call_response(
                call_id="call_propose_quarantine",
                name="propose_quarantine",
                arguments={
                    "workspace_id": workspace_id,
                    "source_file_id": file_id,
                },
            ),
            _final_response("已生成待审批的课程隔离计划。"),
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

    execution_data = execute_response.json()
    assert execution_data["workflow_id"] == workflow_id
    assert execution_data["plan_id"] == plan_id
    assert execution_data["items"] == [
        {
            "sequence_no": 1,
            "source_file_id": file_id,
            "status": "COMPLETED",
            "before_relative_path": "downloads/calculus-course.pdf",
            "after_relative_path": (
                "subjects/mathematics/calculus-course.pdf"
            ),
            "error_code": None,
        }
    ]
    with session_factory() as session:
        execution = session.get(
            OperationExecution,
            execution_data["execution_id"],
        )
        assert execution is not None
        assert execution.workflow_id == workflow_id
        assert execution.plan_id == plan_id
        assert execution.status == "COMPLETED"
        execution_item = session.scalar(
            select(OperationExecutionItem).where(
                OperationExecutionItem.execution_id == execution.id,
                OperationExecutionItem.sequence_no == 1,
            )
        )
        assert execution_item is not None
        assert execution_item.status == "COMPLETED"

    undo_response = client.post(
        f"/api/v1/workflows/{workflow_id}/undo"
    )

    assert undo_response.status_code == 200
    assert undo_response.json()["status"] == "UNDONE"
    assert undo_response.json()["items"][0]["status"] == "UNDONE"
    assert source_path.read_bytes() == source_content
    assert not target_path.exists()
    with session_factory() as session:
        execution = session.get(
            OperationExecution,
            execution_data["execution_id"],
        )
        assert execution is not None
        assert execution.status == "UNDONE"
        execution_item = session.scalar(
            select(OperationExecutionItem).where(
                OperationExecutionItem.execution_id == execution.id,
                OperationExecutionItem.sequence_no == 1,
            )
        )
        assert execution_item is not None
        assert execution_item.status == "UNDONE"
        assert execution_item.undone_at is not None


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
        execution = session.scalar(
            select(OperationExecution).where(
                OperationExecution.workflow_id == workflow_id,
            )
        )
        assert execution is None


def test_agent_rename_proposal_approval_execution_updates_disk_state(
    agent_organization_client: tuple[TestClient, sessionmaker[Session]],
    tmp_path: Path,
) -> None:
    client, session_factory = agent_organization_client
    workspace_id, file_id, source_path, _, source_content = _seed_workspace(
        session_factory,
        tmp_path / "rename-success-workspace",
    )
    target_path = source_path.with_name("calculus-course-final.pdf")
    plan_id, workflow_id = _run_rename_proposal(
        client,
        session_factory,
        workspace_id=workspace_id,
        file_id=file_id,
        request_text="把课程 PDF 重命名为最终版文件名",
        model_client=_rename_model_client(workspace_id, file_id),
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

    with session_factory() as session:
        execution = session.scalar(
            select(OperationExecution).where(
                OperationExecution.workflow_id == workflow_id,
            )
        )
        assert execution is not None
        assert execution.status == "COMPLETED"
        execution_item = session.scalar(
            select(OperationExecutionItem).where(
                OperationExecutionItem.execution_id == execution.id,
                OperationExecutionItem.sequence_no == 1,
            )
        )
        assert execution_item is not None
        assert execution_item.operation_type == "rename"
        assert execution_item.status == "COMPLETED"

    undo_response = client.post(
        f"/api/v1/workflows/{workflow_id}/undo"
    )

    assert undo_response.status_code == 200
    assert undo_response.json()["status"] == "UNDONE"
    assert source_path.read_bytes() == source_content
    assert not target_path.exists()
    with session_factory() as session:
        execution = session.scalar(
            select(OperationExecution).where(
                OperationExecution.workflow_id == workflow_id,
            )
        )
        assert execution is not None
        assert execution.status == "UNDONE"


def test_agent_rename_cannot_execute_before_approval(
    agent_organization_client: tuple[TestClient, sessionmaker[Session]],
    tmp_path: Path,
) -> None:
    client, session_factory = agent_organization_client
    workspace_id, file_id, source_path, _, source_content = _seed_workspace(
        session_factory,
        tmp_path / "rename-unapproved-workspace",
    )
    target_path = source_path.with_name("calculus-course-final.pdf")
    plan_id, workflow_id = _run_rename_proposal(
        client,
        session_factory,
        workspace_id=workspace_id,
        file_id=file_id,
        request_text="整理课程 PDF 的文件名",
        model_client=_rename_model_client(workspace_id, file_id),
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
        execution = session.scalar(
            select(OperationExecution).where(
                OperationExecution.workflow_id == workflow_id,
            )
        )
        assert execution is None


def test_agent_quarantine_proposal_approval_execution_updates_disk_state(
    agent_organization_client: tuple[TestClient, sessionmaker[Session]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session_factory = agent_organization_client
    workspace_id, file_id, source_path, _, source_content = _seed_workspace(
        session_factory,
        tmp_path / "quarantine-success-workspace",
    )
    quarantine_root = tmp_path / "quarantine"
    monkeypatch.setattr(
        safe_execution_module,
        "resolve_quarantine_root",
        lambda: quarantine_root,
    )
    plan_id, workflow_id, quarantine_destination = _run_quarantine_proposal(
        client,
        session_factory,
        workspace_id=workspace_id,
        file_id=file_id,
        request_text="把课程 PDF 放入隔离区",
        model_client=_quarantine_model_client(workspace_id, file_id),
        quarantine_root=quarantine_root,
    )
    quarantine_path = quarantine_root / Path(quarantine_destination)

    assert source_path.exists()
    assert not quarantine_path.exists()

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
    assert quarantine_path.read_bytes() == source_content

    with session_factory() as session:
        execution = session.scalar(
            select(OperationExecution).where(
                OperationExecution.workflow_id == workflow_id,
            )
        )
        assert execution is not None
        assert execution.status == "COMPLETED"
        execution_item = session.scalar(
            select(OperationExecutionItem).where(
                OperationExecutionItem.execution_id == execution.id,
                OperationExecutionItem.sequence_no == 1,
            )
        )
        assert execution_item is not None
        assert execution_item.operation_type == "quarantine"
        assert execution_item.before_location == "workspace"
        assert execution_item.after_location == "quarantine"
        assert execution_item.status == "COMPLETED"

    undo_response = client.post(
        f"/api/v1/workflows/{workflow_id}/undo"
    )

    assert undo_response.status_code == 200
    assert undo_response.json()["status"] == "UNDONE"
    assert source_path.read_bytes() == source_content
    assert not quarantine_path.exists()
    with session_factory() as session:
        execution = session.scalar(
            select(OperationExecution).where(
                OperationExecution.workflow_id == workflow_id,
            )
        )
        assert execution is not None
        assert execution.status == "UNDONE"


def test_agent_quarantine_cannot_execute_before_approval(
    agent_organization_client: tuple[TestClient, sessionmaker[Session]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session_factory = agent_organization_client
    workspace_id, file_id, source_path, _, source_content = _seed_workspace(
        session_factory,
        tmp_path / "quarantine-unapproved-workspace",
    )
    quarantine_root = tmp_path / "quarantine"
    monkeypatch.setattr(
        safe_execution_module,
        "resolve_quarantine_root",
        lambda: quarantine_root,
    )
    plan_id, workflow_id, quarantine_destination = _run_quarantine_proposal(
        client,
        session_factory,
        workspace_id=workspace_id,
        file_id=file_id,
        request_text="隔离课程 PDF",
        model_client=_quarantine_model_client(workspace_id, file_id),
        quarantine_root=quarantine_root,
    )
    quarantine_path = quarantine_root / Path(quarantine_destination)

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
    assert not quarantine_path.exists()
    with session_factory() as session:
        plan = session.get(OperationPlanRecord, plan_id)
        assert plan is not None
        assert plan.status == "WAITING_APPROVAL"
        execution = session.scalar(
            select(OperationExecution).where(
                OperationExecution.workflow_id == workflow_id,
            )
        )
        assert execution is None


def test_agent_reject_keeps_filesystem_unchanged(
    agent_organization_client: tuple[TestClient, sessionmaker[Session]],
    tmp_path: Path,
) -> None:
    client, session_factory = agent_organization_client
    workspace_id, file_id, source_path, target_path, source_content = (
        _seed_workspace(session_factory, tmp_path / "reject-workspace")
    )
    plan_id, workflow_id = _run_move_proposal(
        client,
        session_factory,
        workspace_id=workspace_id,
        file_id=file_id,
        request_text="整理课程 PDF",
        model_client=_move_model_client(workspace_id, file_id),
    )

    reject_response = client.post(
        f"/api/v1/workflows/{workflow_id}/decisions",
        json={
            "action": "reject",
            "expected_plan_id": plan_id,
        },
    )

    assert reject_response.status_code == 200
    assert reject_response.json()["approval_status"] == "REJECTED"
    assert source_path.exists()
    assert source_path.read_bytes() == source_content
    assert not target_path.exists()

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
        assert plan.status == "REJECTED"
        approval = session.scalar(
            select(ApprovalRequest).where(ApprovalRequest.plan_id == plan_id)
        )
        assert approval is not None
        assert approval.status == "REJECTED"
        execution = session.scalar(
            select(OperationExecution).where(
                OperationExecution.workflow_id == workflow_id,
            )
        )
        assert execution is None


def test_agent_running_cancel_reaches_cancelled(
    agent_organization_client: tuple[TestClient, sessionmaker[Session]],
    tmp_path: Path,
) -> None:
    client, session_factory = agent_organization_client
    workspace_id, _, _, _, _ = _seed_workspace(
        session_factory,
        tmp_path / "cancel-workspace",
    )
    started = Event()

    class CancellableExecutor:
        def run(
            self,
            session: Session,
            *,
            workspace_id: int,
            request_text: str,
            run_id: int | None = None,
            cancel_event: Event | None = None,
        ) -> AgentRunResponse:
            assert workspace_id > 0
            assert request_text == "取消请求"
            assert run_id is not None
            assert cancel_event is not None
            started.set()
            assert cancel_event.wait(timeout=5)
            SqlAlchemyAgentRunRecorder(session).finish_run(
                agent_run_id=run_id,
                status="cancelled",
                model_turns=0,
                error_code=None,
            )
            return AgentRunResponse(
                run_id=run_id,
                status="cancelled",
            )

    app.dependency_overrides[get_agent_run_executor] = CancellableExecutor
    response = client.post(
        "/api/v1/agent-runs",
        json={
            "workspace_id": workspace_id,
            "request_text": "取消请求",
        },
    )

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert started.wait(timeout=2)

    cancel_response = client.post(f"/api/v1/agent-runs/{run_id}/cancel")

    assert cancel_response.status_code == 200
    assert cancel_response.json()["run_id"] == run_id
    assert _wait_for_agent_run_status(client, run_id, "cancelled")[
        "status"
    ] == "cancelled"
    with session_factory() as session:
        persisted_run = session.get(AgentRun, run_id)
        assert persisted_run is not None
        assert persisted_run.status == "cancelled"

    repeated_cancel_response = client.post(
        f"/api/v1/agent-runs/{run_id}/cancel"
    )
    assert repeated_cancel_response.status_code == 200
    assert repeated_cancel_response.json()["status"] == "cancelled"


def test_service_restart_recovers_interrupted_agent_run(
    agent_organization_client: tuple[TestClient, sessionmaker[Session]],
    tmp_path: Path,
) -> None:
    client, session_factory = agent_organization_client
    workspace_id, _, _, _, _ = _seed_workspace(
        session_factory,
        tmp_path / "restart-workspace",
    )
    request_text = "重启后恢复请求"
    initial_messages = (
        ModelMessage(role="system", content="系统上下文"),
        ModelMessage(role="user", content=request_text),
    )
    with session_factory() as session:
        recorder = SqlAlchemyAgentRunRecorder(session)
        run_id = recorder.start_pending_run(
            workspace_id=workspace_id,
            request_text=request_text,
            messages=initial_messages,
        )
        recorder.start_existing_run(run_id)

    with session_factory() as session:
        assert recover_unfinished_agent_runs(session) == (run_id,)
    with session_factory() as session:
        interrupted_run = session.get(AgentRun, run_id)
        assert interrupted_run is not None
        assert interrupted_run.status == "failed"
        assert interrupted_run.error_code == "worker_interrupted"

    class ResumeExecutor:
        def run(
            self,
            session: Session,
            *,
            workspace_id: int,
            request_text: str,
            run_id: int | None = None,
            cancel_event: Event | None = None,
        ) -> AgentRunResponse:
            assert workspace_id > 0
            assert request_text == "重启后恢复请求"
            assert run_id is not None
            assert cancel_event is not None
            SqlAlchemyAgentRunRecorder(session).finish_run(
                agent_run_id=run_id,
                status="completed",
                model_turns=1,
                error_code=None,
            )
            return AgentRunResponse(
                run_id=run_id,
                status="completed",
            )

    app.dependency_overrides[get_agent_run_executor] = ResumeExecutor
    response = client.post(f"/api/v1/agent-runs/{run_id}/resume")

    assert response.status_code == 202
    assert response.json() == {"run_id": run_id}
    assert _wait_for_agent_run_status(client, run_id, "completed")[
        "status"
    ] == "completed"
    with session_factory() as session:
        recovered_run = session.get(AgentRun, run_id)
        assert recovered_run is not None
        assert recovered_run.status == "completed"
        assert recovered_run.error_code is None
