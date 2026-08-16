from collections.abc import Iterator
from functools import partial
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database import Base, get_session
from backend.app.main import app
from backend.app.models import FileEntry, OperationPlanRecord, Workspace
from backend.app.services import get_operation_plan, validate_operation_plan
from backend.app.workflow_graph import open_checkpointed_workflow_graph
from backend.app.workflow_runtime import get_workflow_graph


WORKFLOW_ID = UUID("11111111-1111-4111-8111-111111111111")


@pytest.fixture
def organization_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, sessionmaker[Session], Path]]:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'organization-api.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    checkpoint_path = tmp_path / "workflow-checkpoints.sqlite"

    def override_get_session() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    def override_get_workflow_graph() -> Iterator[object]:
        with session_factory() as session:
            with open_checkpointed_workflow_graph(
                checkpoint_path,
                operation_plan_validator=partial(validate_operation_plan, session),
            ) as graph:
                yield graph

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_workflow_graph] = override_get_workflow_graph
    with TestClient(app) as client:
        yield client, session_factory, tmp_path

    app.dependency_overrides.clear()
    engine.dispose()


def _seed_workspace(
    session_factory: sessionmaker[Session],
    workspace_root: Path,
) -> tuple[int, int]:
    source_path = workspace_root / "inbox" / "quarterly-report.txt"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"organization api")
    (workspace_root / "reports" / "quarterly").mkdir(parents=True)

    with session_factory() as session:
        workspace = Workspace(name="计划 API", root_path=str(workspace_root))
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
        return workspace.id, file_entry.id


def test_create_and_get_organization_workflow(
    organization_client: tuple[TestClient, sessionmaker[Session], Path],
) -> None:
    client, session_factory, tmp_path = organization_client
    workspace_id, file_id = _seed_workspace(
        session_factory,
        tmp_path / "workspace",
    )

    create_response = client.post(
        "/api/v1/workflows",
        json={
            "workspace_id": workspace_id,
            "target_directories": ["reports/quarterly"],
            "selections": [
                {
                    "source_file_id": file_id,
                    "target_directory": "reports/quarterly",
                }
            ],
        },
    )

    assert create_response.status_code == 201
    created = create_response.json()
    workflow_id = created["workflow_id"]
    assert created["status"] == "waiting"
    assert created["approval_status"] == "WAITING_APPROVAL"
    assert created["operation_plan"]["workspace_id"] == workspace_id
    assert created["operation_plan"]["operations"][0][
        "target_relative_path"
    ] == "reports/quarterly/quarterly-report.txt"

    get_response = client.get(f"/api/v1/workflows/{workflow_id}")

    assert get_response.status_code == 200
    assert get_response.json() == created


def test_get_workflow_reads_complete_plan_from_business_database(
    organization_client: tuple[TestClient, sessionmaker[Session], Path],
) -> None:
    client, session_factory, tmp_path = organization_client
    workspace_id, file_id = _seed_workspace(
        session_factory,
        tmp_path / "business-plan-source-workspace",
    )
    created = client.post(
        "/api/v1/workflows",
        json={
            "workspace_id": workspace_id,
            "target_directories": ["reports/quarterly"],
            "selections": [
                {
                    "source_file_id": file_id,
                    "target_directory": "reports/quarterly",
                }
            ],
        },
    ).json()

    with session_factory() as session:
        persisted = session.get(
            OperationPlanRecord,
            created["operation_plan"]["plan_id"],
        )
        assert persisted is not None
        persisted.items[0].target_relative_path = (
            "reports/quarterly/from-business-db.txt"
        )
        session.commit()

    response = client.get(f"/api/v1/workflows/{created['workflow_id']}")

    assert response.status_code == 200
    assert response.json()["operation_plan"]["operations"][0][
        "target_relative_path"
    ] == "reports/quarterly/from-business-db.txt"


def test_get_workflow_rejects_missing_business_plan(
    organization_client: tuple[TestClient, sessionmaker[Session], Path],
) -> None:
    client, session_factory, tmp_path = organization_client
    workspace_id, file_id = _seed_workspace(
        session_factory,
        tmp_path / "missing-business-plan-workspace",
    )
    created = client.post(
        "/api/v1/workflows",
        json={
            "workspace_id": workspace_id,
            "target_directories": ["reports/quarterly"],
            "selections": [
                {
                    "source_file_id": file_id,
                    "target_directory": "reports/quarterly",
                }
            ],
        },
    ).json()

    with session_factory() as session:
        persisted = session.get(
            OperationPlanRecord,
            created["operation_plan"]["plan_id"],
        )
        assert persisted is not None
        session.delete(persisted)
        session.commit()

    response = client.get(f"/api/v1/workflows/{created['workflow_id']}")

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "workflow_state_conflict"


def test_business_plan_survives_deleted_checkpoint(
    organization_client: tuple[TestClient, sessionmaker[Session], Path],
) -> None:
    client, session_factory, tmp_path = organization_client
    workspace_id, file_id = _seed_workspace(
        session_factory,
        tmp_path / "checkpoint-loss-workspace",
    )
    created = client.post(
        "/api/v1/workflows",
        json={
            "workspace_id": workspace_id,
            "target_directories": ["reports/quarterly"],
            "selections": [
                {
                    "source_file_id": file_id,
                    "target_directory": "reports/quarterly",
                }
            ],
        },
    ).json()

    checkpoint_path = tmp_path / "workflow-checkpoints.sqlite"
    assert checkpoint_path.is_file()
    checkpoint_path.unlink()

    with session_factory() as session:
        restored = get_operation_plan(
            session,
            created["operation_plan"]["plan_id"],
            workflow_id=created["workflow_id"],
        )

    assert restored is not None
    assert restored.plan_id == UUID(created["operation_plan"]["plan_id"])
    assert restored.operations[0].target_relative_path == (
        "reports/quarterly/quarterly-report.txt"
    )


def test_get_unknown_organization_workflow_returns_safe_404(
    organization_client: tuple[TestClient, sessionmaker[Session], Path],
) -> None:
    client, _, _ = organization_client

    response = client.get(f"/api/v1/workflows/{WORKFLOW_ID}")

    assert response.status_code == 404
    assert response.json() == {
        "detail": {
            "code": "workflow_not_found",
            "message": "工作流不存在。",
        }
    }


def test_approve_organization_workflow_via_api(
    organization_client: tuple[TestClient, sessionmaker[Session], Path],
) -> None:
    client, session_factory, tmp_path = organization_client
    workspace_id, file_id = _seed_workspace(
        session_factory,
        tmp_path / "approval-workspace",
    )
    created = client.post(
        "/api/v1/workflows",
        json={
            "workspace_id": workspace_id,
            "target_directories": ["reports/quarterly"],
            "selections": [
                {
                    "source_file_id": file_id,
                    "target_directory": "reports/quarterly",
                }
            ],
        },
    ).json()

    response = client.post(
        f"/api/v1/workflows/{created['workflow_id']}/decisions",
        json={
            "action": "approve",
            "expected_plan_id": created["operation_plan"]["plan_id"],
        },
    )

    assert response.status_code == 200
    decided = response.json()
    assert decided["status"] == "ready"
    assert decided["approval_status"] == "APPROVED"
    assert decided["operation_plan"] == created["operation_plan"]


def test_decision_api_rejects_stale_plan_id(
    organization_client: tuple[TestClient, sessionmaker[Session], Path],
) -> None:
    client, session_factory, tmp_path = organization_client
    workspace_id, file_id = _seed_workspace(
        session_factory,
        tmp_path / "stale-plan-workspace",
    )
    created = client.post(
        "/api/v1/workflows",
        json={
            "workspace_id": workspace_id,
            "target_directories": ["reports/quarterly"],
            "selections": [
                {
                    "source_file_id": file_id,
                    "target_directory": "reports/quarterly",
                }
            ],
        },
    ).json()

    response = client.post(
        f"/api/v1/workflows/{created['workflow_id']}/decisions",
        json={
            "action": "approve",
            "expected_plan_id": str(WORKFLOW_ID),
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "code": "organization_workflow_plan_mismatch",
            "message": "审批决定与当前工作流状态冲突。",
        }
    }
    unchanged = client.get(
        f"/api/v1/workflows/{created['workflow_id']}"
    ).json()
    assert unchanged["status"] == "waiting"
    assert unchanged["approval_status"] == "WAITING_APPROVAL"


def test_edit_organization_workflow_via_api(
    organization_client: tuple[TestClient, sessionmaker[Session], Path],
) -> None:
    client, session_factory, tmp_path = organization_client
    workspace_root = tmp_path / "edit-api-workspace"
    workspace_id, file_id = _seed_workspace(session_factory, workspace_root)
    (workspace_root / "archive").mkdir()

    created = client.post(
        "/api/v1/workflows",
        json={
            "workspace_id": workspace_id,
            "target_directories": ["reports/quarterly"],
            "selections": [
                {
                    "source_file_id": file_id,
                    "target_directory": "reports/quarterly",
                }
            ],
        },
    ).json()

    response = client.post(
        f"/api/v1/workflows/{created['workflow_id']}/decisions",
        json={
            "action": "edit",
            "expected_plan_id": created["operation_plan"]["plan_id"],
            "changes": [
                {
                    "source_file_id": file_id,
                    "target_directory": "archive",
                }
            ],
        },
    )

    assert response.status_code == 200
    edited = response.json()
    assert edited["status"] == "waiting"
    assert edited["approval_status"] == "WAITING_APPROVAL"
    assert edited["operation_plan"]["plan_id"] != created["operation_plan"][
        "plan_id"
    ]
    assert edited["operation_plan"]["operations"][0]["target_relative_path"] == (
        "archive/quarterly-report.txt"
    )


def test_edit_api_rejects_stale_plan_id(
    organization_client: tuple[TestClient, sessionmaker[Session], Path],
) -> None:
    client, session_factory, tmp_path = organization_client
    workspace_root = tmp_path / "stale-edit-api-workspace"
    workspace_id, file_id = _seed_workspace(session_factory, workspace_root)
    (workspace_root / "archive").mkdir()
    created = client.post(
        "/api/v1/workflows",
        json={
            "workspace_id": workspace_id,
            "target_directories": ["reports/quarterly"],
            "selections": [
                {
                    "source_file_id": file_id,
                    "target_directory": "reports/quarterly",
                }
            ],
        },
    ).json()

    response = client.post(
        f"/api/v1/workflows/{created['workflow_id']}/decisions",
        json={
            "action": "edit",
            "expected_plan_id": str(WORKFLOW_ID),
            "changes": [
                {
                    "source_file_id": file_id,
                    "target_directory": "archive",
                }
            ],
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "code": "organization_workflow_plan_mismatch",
            "message": "审批决定与当前工作流状态冲突。",
        }
    }


def test_execute_and_undo_organization_workflow_via_api(
    organization_client: tuple[TestClient, sessionmaker[Session], Path],
) -> None:
    client, session_factory, tmp_path = organization_client
    workspace_root = tmp_path / "execute-undo-api-workspace"
    workspace_id, file_id = _seed_workspace(session_factory, workspace_root)
    source_path = workspace_root / "inbox" / "quarterly-report.txt"
    target_path = workspace_root / "reports" / "quarterly" / "quarterly-report.txt"
    source_content = source_path.read_bytes()

    create_response = client.post(
        "/api/v1/workflows",
        json={
            "workspace_id": workspace_id,
            "target_directories": ["reports/quarterly"],
            "selections": [
                {
                    "source_file_id": file_id,
                    "target_directory": "reports/quarterly",
                }
            ],
        },
    )
    assert create_response.status_code == 201
    created = create_response.json()

    approve_response = client.post(
        f"/api/v1/workflows/{created['workflow_id']}/decisions",
        json={
            "action": "approve",
            "expected_plan_id": created["operation_plan"]["plan_id"],
        },
    )
    assert approve_response.status_code == 200
    assert approve_response.json()["approval_status"] == "APPROVED"

    execute_response = client.post(
        f"/api/v1/workflows/{created['workflow_id']}/execute"
    )
    assert execute_response.status_code == 200
    executed = execute_response.json()
    assert executed["status"] == "COMPLETED"
    assert executed["items"][0]["before_relative_path"] == (
        "inbox/quarterly-report.txt"
    )
    assert executed["items"][0]["after_relative_path"] == (
        "reports/quarterly/quarterly-report.txt"
    )
    assert not source_path.exists()
    assert target_path.read_bytes() == source_content

    undo_response = client.post(
        f"/api/v1/workflows/{created['workflow_id']}/undo"
    )
    assert undo_response.status_code == 200
    undone = undo_response.json()
    assert undone["status"] == "UNDONE"
    assert source_path.read_bytes() == source_content
    assert not target_path.exists()


def test_execute_api_rejects_unapproved_workflow_without_disk_change(
    organization_client: tuple[TestClient, sessionmaker[Session], Path],
) -> None:
    client, session_factory, tmp_path = organization_client
    workspace_root = tmp_path / "unapproved-execute-api-workspace"
    workspace_id, file_id = _seed_workspace(session_factory, workspace_root)
    source_path = workspace_root / "inbox" / "quarterly-report.txt"
    source_content = source_path.read_bytes()

    created = client.post(
        "/api/v1/workflows",
        json={
            "workspace_id": workspace_id,
            "target_directories": ["reports/quarterly"],
            "selections": [
                {
                    "source_file_id": file_id,
                    "target_directory": "reports/quarterly",
                }
            ],
        },
    ).json()

    response = client.post(
        f"/api/v1/workflows/{created['workflow_id']}/execute"
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "code": "organization_workflow_not_ready",
            "message": "工作流尚未获得批准，不能执行文件操作。",
        }
    }
    assert source_path.read_bytes() == source_content


def test_minimal_ui_wires_execution_and_undo_path() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert 'id="execute-plan"' in html
    assert 'id="undo-plan"' in html
    assert "/execute`" in html
    assert "/undo`" in html
    assert 'executePlanButton.addEventListener("click"' in html
    assert 'undoPlanButton.addEventListener("click"' in html
    assert "showExecutionAction();" in html
    assert "showUndoAction();" in html
