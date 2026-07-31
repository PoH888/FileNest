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
from backend.app.models import FileEntry, Workspace
from backend.app.services import validate_operation_plan
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
