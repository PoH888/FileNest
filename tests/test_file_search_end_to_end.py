from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database import Base, get_session
from backend.app.main import app


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    database_path = tmp_path / "file-search-e2e.db"
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


def test_create_scan_search_and_read_file_detail(
    client: TestClient,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "search-workspace"
    reports = workspace_root / "Reports"
    reports.mkdir(parents=True)
    report_content = "FileNest quarterly report"
    report_path = reports / "Quarterly Summary.PDF"
    report_path.write_text(report_content, encoding="utf-8")
    (reports / "quarterly-notes.txt").write_text("notes", encoding="utf-8")
    (workspace_root / "unrelated.pdf").write_text("other", encoding="utf-8")

    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "端到端搜索工作区",
            "root_path": str(workspace_root),
        },
    )
    assert create_response.status_code == 201
    workspace_id = create_response.json()["id"]

    scan_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/scan",
    )
    assert scan_response.status_code == 202
    scan_job = _wait_for_job(
        client,
        scan_response.json()["job_id"],
        "completed",
        workspace_id,
    )
    assert scan_job["error_code"] is None

    search_response = client.get(
        f"/api/v1/workspaces/{workspace_id}/files",
        params={
            "keyword": "quarterly",
            "extension": "pdf",
        },
    )
    assert search_response.status_code == 200
    search_payload = search_response.json()
    assert search_payload["total"] == 1
    assert search_payload["page"] == 1
    assert search_payload["page_size"] == 50

    item = search_payload["items"][0]
    assert item["relative_path"] == "Reports/Quarterly Summary.PDF"
    assert item["name"] == "Quarterly Summary.PDF"
    assert item["extension"] == ".pdf"
    assert item["size_bytes"] == len(report_content.encode("utf-8"))
    assert datetime.fromisoformat(item["modified_at"]).utcoffset() is not None
    assert "root_path" not in item

    detail_response = client.get(
        f"/api/v1/workspaces/{workspace_id}/files/{item['id']}"
    )
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail == {
        **item,
        "workspace_id": workspace_id,
    }
    assert "root_path" not in detail
