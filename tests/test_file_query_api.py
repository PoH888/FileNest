from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database import Base, get_session
from backend.app.main import app
from backend.app.models import FileEntry


@pytest.fixture
def file_client(tmp_path: Path) -> Iterator[tuple[TestClient, Engine]]:
    database_path = tmp_path / "file-query-api.db"
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


def _create_workspace(client: TestClient, name: str = "文件 API 工作区") -> int:
    response = client.post(
        "/api/v1/workspaces",
        json={
            "name": name,
            "root_path": f"D:/Test/{name}",
        },
    )
    assert response.status_code == 201
    return response.json()["id"]


def _epoch_ns(year: int, month: int, day: int) -> int:
    value = datetime(year, month, day, tzinfo=timezone.utc)
    return int(value.timestamp()) * 1_000_000_000


def _add_file_entries(engine: Engine, workspace_id: int) -> int:
    with Session(engine) as session:
        entries = [
            FileEntry(
                workspace_id=workspace_id,
                relative_path="Reports/a-old.pdf",
                name="a-old.pdf",
                extension=".pdf",
                size_bytes=10,
                mtime_ns=_epoch_ns(2026, 8, 1),
            ),
            FileEntry(
                workspace_id=workspace_id,
                relative_path="Reports/b-middle.pdf",
                name="b-middle.pdf",
                extension=".pdf",
                size_bytes=20,
                mtime_ns=_epoch_ns(2026, 8, 15),
            ),
            FileEntry(
                workspace_id=workspace_id,
                relative_path="Reports/c-new.pdf",
                name="c-new.pdf",
                extension=".pdf",
                size_bytes=30,
                mtime_ns=_epoch_ns(2026, 8, 20),
            ),
            FileEntry(
                workspace_id=workspace_id,
                relative_path="notes/todo.txt",
                name="todo.txt",
                extension=".txt",
                size_bytes=40,
                mtime_ns=_epoch_ns(2026, 8, 15),
            ),
        ]
        session.add_all(entries)
        session.commit()
        return entries[0].id


def test_file_list_api_returns_safe_paginated_response(
    file_client: tuple[TestClient, Engine],
) -> None:
    client, engine = file_client
    workspace_id = _create_workspace(client)
    _add_file_entries(engine, workspace_id)

    response = client.get(f"/api/v1/workspaces/{workspace_id}/files")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 4
    assert payload["page"] == 1
    assert payload["page_size"] == 50
    assert payload["items"][0] == {
        "id": 1,
        "relative_path": "Reports/a-old.pdf",
        "name": "a-old.pdf",
        "extension": ".pdf",
        "size_bytes": 10,
        "modified_at": "2026-08-01T00:00:00Z",
    }
    assert all("root_path" not in item for item in payload["items"])


def test_file_search_api_filters_sorts_and_paginates(
    file_client: tuple[TestClient, Engine],
) -> None:
    client, engine = file_client
    workspace_id = _create_workspace(client)
    _add_file_entries(engine, workspace_id)

    response = client.get(
        f"/api/v1/workspaces/{workspace_id}/files",
        params={
            "keyword": "reports",
            "extension": "PDF",
            "modified_from": "2026-08-15T00:00:00Z",
            "modified_to": "2026-08-20T00:00:00Z",
            "sort_by": "modified_at",
            "sort_order": "desc",
            "page": 2,
            "page_size": 1,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {
                "id": 2,
                "relative_path": "Reports/b-middle.pdf",
                "name": "b-middle.pdf",
                "extension": ".pdf",
                "size_bytes": 20,
                "modified_at": "2026-08-15T00:00:00Z",
            }
        ],
        "total": 2,
        "page": 2,
        "page_size": 1,
    }


def test_file_list_api_returns_workspace_not_found(
    file_client: tuple[TestClient, Engine],
) -> None:
    client, _ = file_client

    response = client.get("/api/v1/workspaces/999/files")

    assert response.status_code == 404
    assert response.json() == {
        "detail": {
            "code": "workspace_not_found",
            "message": "工作区不存在。",
        }
    }


def test_file_list_api_rejects_invalid_query(
    file_client: tuple[TestClient, Engine],
) -> None:
    client, _ = file_client
    workspace_id = _create_workspace(client)

    response = client.get(
        f"/api/v1/workspaces/{workspace_id}/files",
        params={"page": 0},
    )

    assert response.status_code == 422


def test_file_detail_api_returns_safe_index_metadata(
    file_client: tuple[TestClient, Engine],
) -> None:
    client, engine = file_client
    workspace_id = _create_workspace(client)
    file_id = _add_file_entries(engine, workspace_id)

    response = client.get(
        f"/api/v1/workspaces/{workspace_id}/files/{file_id}"
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": file_id,
        "relative_path": "Reports/a-old.pdf",
        "name": "a-old.pdf",
        "extension": ".pdf",
        "size_bytes": 10,
        "modified_at": "2026-08-01T00:00:00Z",
        "workspace_id": workspace_id,
    }
    assert "root_path" not in response.json()


def test_file_detail_api_hides_cross_workspace_file(
    file_client: tuple[TestClient, Engine],
) -> None:
    client, engine = file_client
    first_workspace_id = _create_workspace(client, "第一个详情工作区")
    second_workspace_id = _create_workspace(client, "第二个详情工作区")
    file_id = _add_file_entries(engine, first_workspace_id)

    response = client.get(
        f"/api/v1/workspaces/{second_workspace_id}/files/{file_id}"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": {
            "code": "file_not_found",
            "message": "文件索引不存在。",
        }
    }


def test_file_detail_api_returns_file_not_found(
    file_client: tuple[TestClient, Engine],
) -> None:
    client, _ = file_client
    workspace_id = _create_workspace(client)

    response = client.get(
        f"/api/v1/workspaces/{workspace_id}/files/999"
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "file_not_found"


def test_file_detail_api_returns_workspace_not_found(
    file_client: tuple[TestClient, Engine],
) -> None:
    client, _ = file_client

    response = client.get("/api/v1/workspaces/999/files/1")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "workspace_not_found"


def test_file_detail_api_rejects_non_integer_file_id(
    file_client: tuple[TestClient, Engine],
) -> None:
    client, _ = file_client
    workspace_id = _create_workspace(client)

    response = client.get(
        f"/api/v1/workspaces/{workspace_id}/files/not-an-integer"
    )

    assert response.status_code == 422
