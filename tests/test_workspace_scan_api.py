from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database import Base, get_session
from backend.app.main import app
from backend.app.models import FileEntry
from backend.app.repositories import find_file_entries


@pytest.fixture
def scan_client(tmp_path: Path) -> Iterator[tuple[TestClient, Engine]]:
    database_path = tmp_path / "scan-api.db"
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
        yield test_client, test_engine

    app.dependency_overrides.clear()
    test_engine.dispose()


def test_scan_workspace_api_indexes_files(
    scan_client: tuple[TestClient, Engine],
    tmp_path: Path,
) -> None:
    client, engine = scan_client
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "report.txt").write_text("report", encoding="utf-8")
    documents = workspace_root / "Documents"
    documents.mkdir()
    (documents / "notes.md").write_text("notes", encoding="utf-8")

    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "扫描 API 测试",
            "root_path": str(workspace_root),
        },
    )
    workspace_id = create_response.json()["id"]

    scan_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/scan",
    )

    assert scan_response.status_code == 200
    assert scan_response.json() == {
        "created": 2,
        "updated": 0,
        "deleted": 0,
        "unchanged": 0,
    }

    with Session(engine) as session:
        assert [
            entry.relative_path
            for entry in find_file_entries(session, workspace_id)
        ] == ["Documents/notes.md", "report.txt"]


def test_repeated_scan_tracks_real_file_changes(
    scan_client: tuple[TestClient, Engine],
    tmp_path: Path,
) -> None:
    client, engine = scan_client
    workspace_root = tmp_path / "changing-workspace"
    workspace_root.mkdir()

    keep_file = workspace_root / "keep.txt"
    changed_file = workspace_root / "changed.txt"
    deleted_file = workspace_root / "deleted.txt"
    keep_file.write_text("keep", encoding="utf-8")
    changed_file.write_text("old", encoding="utf-8")
    deleted_file.write_text("delete", encoding="utf-8")

    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "连续扫描测试",
            "root_path": str(workspace_root),
        },
    )
    workspace_id = create_response.json()["id"]
    scan_url = f"/api/v1/workspaces/{workspace_id}/scan"

    first_response = client.post(scan_url)

    assert first_response.status_code == 200
    assert first_response.json() == {
        "created": 3,
        "updated": 0,
        "deleted": 0,
        "unchanged": 0,
    }

    with Session(engine) as session:
        initial_entries = {
            entry.relative_path: entry
            for entry in find_file_entries(session, workspace_id)
        }
        changed_entry_id = initial_entries["changed.txt"].id
        keep_entry_id = initial_entries["keep.txt"].id

    repeated_response = client.post(scan_url)

    assert repeated_response.status_code == 200
    assert repeated_response.json() == {
        "created": 0,
        "updated": 0,
        "deleted": 0,
        "unchanged": 3,
    }

    changed_file.write_text("new content is longer", encoding="utf-8")
    new_file = workspace_root / "new.pdf"
    new_file.write_bytes(b"new file")
    deleted_file.unlink()

    changed_response = client.post(scan_url)

    assert changed_response.status_code == 200
    assert changed_response.json() == {
        "created": 1,
        "updated": 1,
        "deleted": 1,
        "unchanged": 1,
    }

    with Session(engine) as session:
        final_entries = {
            entry.relative_path: entry
            for entry in find_file_entries(session, workspace_id)
        }

        assert list(final_entries) == [
            "changed.txt",
            "keep.txt",
            "new.pdf",
        ]
        assert final_entries["changed.txt"].id == changed_entry_id
        assert final_entries["keep.txt"].id == keep_entry_id
        assert final_entries["changed.txt"].size_bytes == changed_file.stat().st_size
        assert final_entries["changed.txt"].mtime_ns == changed_file.stat().st_mtime_ns
        assert final_entries["new.pdf"].size_bytes == new_file.stat().st_size


def test_scan_workspace_api_returns_not_found(
    scan_client: tuple[TestClient, Engine],
) -> None:
    client, _ = scan_client

    response = client.post("/api/v1/workspaces/999/scan")

    assert response.status_code == 404
    assert response.json() == {
        "detail": {
            "code": "workspace_not_found",
            "message": "工作区不存在。",
        }
    }


def test_unavailable_workspace_does_not_delete_existing_index(
    scan_client: tuple[TestClient, Engine],
    tmp_path: Path,
) -> None:
    client, engine = scan_client
    missing_root = tmp_path / "missing-workspace"
    missing_root.mkdir()
    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "失效工作区",
            "root_path": str(missing_root),
        },
    )
    workspace_id = create_response.json()["id"]
    missing_root.rmdir()

    with Session(engine) as session:
        session.add(
            FileEntry(
                workspace_id=workspace_id,
                relative_path="preserved.txt",
                name="preserved.txt",
                extension=".txt",
                size_bytes=10,
                mtime_ns=100,
            )
        )
        session.commit()

    response = client.post(
        f"/api/v1/workspaces/{workspace_id}/scan",
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "code": "workspace_scan_unavailable",
            "message": "工作区目录当前不可扫描。",
        }
    }

    with Session(engine) as session:
        assert [
            entry.relative_path
            for entry in find_file_entries(session, workspace_id)
        ] == ["preserved.txt"]
