from collections.abc import Iterator
from pathlib import Path
from time import monotonic, sleep

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


def _wait_for_job(
    client: TestClient,
    job_id: str,
    expected_status: str,
    workspace_id: int,
) -> dict[str, object]:
    deadline = monotonic() + 2
    while monotonic() < deadline:
        response = client.get(
            f"/api/v1/jobs/{job_id}",
            params={"workspace_id": workspace_id},
        )
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] == expected_status:
            return payload
        sleep(0.005)
    pytest.fail(f"job did not reach status {expected_status!r}")


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

    assert scan_response.status_code == 202
    job_id = scan_response.json()["job_id"]
    completed = _wait_for_job(client, job_id, "completed", workspace_id)
    assert completed["job_id"] == job_id
    assert completed["error_code"] is None

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

    assert first_response.status_code == 202
    _wait_for_job(
        client,
        first_response.json()["job_id"],
        "completed",
        workspace_id,
    )

    with Session(engine) as session:
        initial_entries = {
            entry.relative_path: entry
            for entry in find_file_entries(session, workspace_id)
        }
        changed_entry_id = initial_entries["changed.txt"].id
        keep_entry_id = initial_entries["keep.txt"].id

    repeated_response = client.post(
        scan_url,
        headers={"Idempotency-Key": "changing-scan-002"},
    )

    assert repeated_response.status_code == 202
    _wait_for_job(
        client,
        repeated_response.json()["job_id"],
        "completed",
        workspace_id,
    )

    changed_file.write_text("new content is longer", encoding="utf-8")
    new_file = workspace_root / "new.pdf"
    new_file.write_bytes(b"new file")
    deleted_file.unlink()

    changed_response = client.post(
        scan_url,
        headers={"Idempotency-Key": "changing-scan-003"},
    )

    assert changed_response.status_code == 202
    _wait_for_job(
        client,
        changed_response.json()["job_id"],
        "completed",
        workspace_id,
    )

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

    assert response.status_code == 202
    failed = _wait_for_job(
        client,
        response.json()["job_id"],
        "failed",
        workspace_id,
    )
    assert failed["error_code"] == "workspace_scan_unavailable"

    with Session(engine) as session:
        assert [
            entry.relative_path
            for entry in find_file_entries(session, workspace_id)
        ] == ["preserved.txt"]


def test_job_status_returns_not_found(scan_client: tuple[TestClient, Engine]) -> None:
    client, _ = scan_client

    response = client.get(
        "/api/v1/jobs/00000000-0000-0000-0000-000000000000",
        params={"workspace_id": 1},
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": {
            "code": "job_not_found",
            "message": "Job 不存在。",
        }
    }
