import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database import Base, get_session
from backend.app.main import app
from backend.app.models import Workspace


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    database_path = tmp_path / "filenest-test.db"
    test_engine = create_engine(
        f"sqlite:///{database_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )

    Base.metadata.create_all(bind=test_engine)

    TestSessionFactory = sessionmaker(
        bind=test_engine,
        expire_on_commit=False,
    )

    def override_get_session() -> Iterator[Session]:
        with TestSessionFactory() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    test_engine.dispose()


def test_created_workspace_is_available_to_later_request(
    client: TestClient,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    payload = {
        "name": "测试工作区",
        "root_path": str(workspace_root),
    }
    expected_workspace = {
        "id": 1,
        "name": "测试工作区",
        "root_path": str(workspace_root.resolve()),
    }

    create_response = client.post(
        "/api/v1/workspaces",
        json=payload,
    )

    assert create_response.status_code == 201
    assert create_response.json() == expected_workspace

    list_response = client.get("/api/v1/workspaces")

    assert list_response.status_code == 200
    assert list_response.json() == [expected_workspace]


def test_duplicate_root_path_returns_conflict(
    client: TestClient,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "duplicate"
    workspace_root.mkdir()
    payload = {
        "name": "重复路径测试",
        "root_path": str(workspace_root),
    }

    first_response = client.post(
        "/api/v1/workspaces",
        json=payload,
    )
    duplicate_response = client.post(
        "/api/v1/workspaces",
        json=payload,
    )

    assert first_response.status_code == 201
    assert duplicate_response.status_code == 409
    assert duplicate_response.json() == {
        "detail": {
            "code": "workspace_path_conflict",
            "message": "工作区路径已存在。",
        }
    }

    list_response = client.get("/api/v1/workspaces")

    assert list_response.status_code == 200
    assert len(list_response.json()) == 1


def test_workspace_create_rejects_missing_root(
    client: TestClient,
    tmp_path: Path,
) -> None:
    response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "不存在目录",
            "root_path": str(tmp_path / "missing"),
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "workspace_root_not_found"


def test_workspace_create_rejects_file_root(
    client: TestClient,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "not-a-directory.txt"
    file_path.write_text("not a directory", encoding="utf-8")

    response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "输入文件",
            "root_path": str(file_path),
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "workspace_root_not_directory"


def test_workspace_create_normalizes_parent_and_trailing_separator(
    client: TestClient,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "normalized"
    workspace_root.mkdir()
    unresolved_path = workspace_root.parent / "missing-parent" / ".." / "normalized"

    response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "规范化路径",
            "root_path": f"{unresolved_path}{os.sep}",
        },
    )

    assert response.status_code == 201
    assert response.json()["root_path"] == str(workspace_root.resolve())

    duplicate_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "同一路径",
            "root_path": str(workspace_root),
        },
    )
    assert duplicate_response.status_code == 409
    assert duplicate_response.json()["detail"]["code"] == "workspace_path_conflict"


@pytest.mark.skipif(os.name != "nt", reason="Windows 大小写路径语义专属测试")
def test_workspace_create_treats_windows_case_variants_as_duplicate(
    client: TestClient,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "CaseSensitiveName"
    workspace_root.mkdir()

    first_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "大小写路径",
            "root_path": str(workspace_root),
        },
    )
    duplicate_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "大小写等价路径",
            "root_path": str(workspace_root).lower(),
        },
    )

    assert first_response.status_code == 201
    assert duplicate_response.status_code == 409
    assert duplicate_response.json()["detail"]["code"] == "workspace_path_conflict"


def test_workspace_create_rejects_sensitive_root(
    client: TestClient,
    tmp_path: Path,
) -> None:
    sensitive_root = tmp_path / ".git"
    sensitive_root.mkdir()

    response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "敏感目录",
            "root_path": str(sensitive_root),
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "sensitive_path"


def test_workspace_persists_after_database_reopens(tmp_path: Path) -> None:
    database_path = tmp_path / "reopen-test.db"
    database_url = f"sqlite:///{database_path.as_posix()}"

    first_engine = create_engine(database_url)
    try:
        Base.metadata.create_all(bind=first_engine)
        FirstSessionFactory = sessionmaker(
            bind=first_engine,
            expire_on_commit=False,
        )

        with FirstSessionFactory() as session:
            session.add(
                Workspace(
                    name="重新连接测试",
                    root_path="D:/Test/Reopen",
                )
            )
            session.commit()
    finally:
        first_engine.dispose()

    assert database_path.exists()

    second_engine = create_engine(database_url)
    try:
        SecondSessionFactory = sessionmaker(
            bind=second_engine,
            expire_on_commit=False,
        )

        with SecondSessionFactory() as session:
            workspace = session.get(Workspace, 1)

            assert workspace is not None
            assert workspace.name == "重新连接测试"
            assert workspace.root_path == "D:/Test/Reopen"
    finally:
        second_engine.dispose()
