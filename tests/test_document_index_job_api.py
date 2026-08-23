from collections.abc import Iterator
from pathlib import Path
from time import monotonic, sleep

import pytest
from docx import Document as WordDocument
from pypdf import PdfWriter
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database import Base, get_session
from backend.app.main import app
from backend.app.models import ChunkRecord, DocumentRecord, FileEntry


@pytest.fixture
def document_index_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, Engine]]:
    database_path = tmp_path / "document-index-api.db"
    test_engine = create_engine(
        f"sqlite:///{database_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=test_engine)
    test_session_factory = sessionmaker(
        bind=test_engine,
        expire_on_commit=False,
    )

    def override_get_session() -> Iterator[Session]:
        with test_session_factory() as session:
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
) -> dict[str, object]:
    deadline = monotonic() + 2
    while monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] == expected_status:
            return payload
        sleep(0.005)
    pytest.fail(f"job did not reach status {expected_status!r}")


def test_document_index_job_parses_chunks_and_persists(
    document_index_client: tuple[TestClient, Engine],
    tmp_path: Path,
) -> None:
    client, engine = document_index_client
    workspace_root = tmp_path / "document-workspace"
    workspace_root.mkdir()
    source_file = workspace_root / "notes.md"
    source_file.write_text("标题\n正文", encoding="utf-8")

    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "文档索引 API 测试",
            "root_path": str(workspace_root),
        },
    )
    workspace_id = create_response.json()["id"]

    scan_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/scan",
    )
    assert scan_response.status_code == 202
    _wait_for_job(client, scan_response.json()["job_id"], "completed")

    index_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/index",
    )
    assert index_response.status_code == 202
    index_job = _wait_for_job(
        client,
        index_response.json()["job_id"],
        "completed",
    )
    assert index_job["error_code"] is None

    with Session(engine) as session:
        documents = list(
            session.scalars(
                select(DocumentRecord).where(
                    DocumentRecord.workspace_id == workspace_id,
                )
            ).all()
        )
        chunks = list(
            session.scalars(
                select(ChunkRecord).where(
                    ChunkRecord.file_entry_id == documents[0].file_entry_id,
                )
            ).all()
        )

    assert len(documents) == 1
    assert documents[0].source_relative_path == "notes.md"
    assert documents[0].source_format == "markdown"
    assert [chunk.text for chunk in chunks] == ["标题\n正文"]


def test_document_index_job_supports_pdf(
    document_index_client: tuple[TestClient, Engine],
    tmp_path: Path,
) -> None:
    client, engine = document_index_client
    workspace_root = tmp_path / "pdf-document-workspace"
    workspace_root.mkdir()
    pdf_file = workspace_root / "summary.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_file.open("wb") as output:
        writer.write(output)

    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "PDF 文档索引测试",
            "root_path": str(workspace_root),
        },
    )
    workspace_id = create_response.json()["id"]

    scan_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/scan",
    )
    assert scan_response.status_code == 202
    _wait_for_job(client, scan_response.json()["job_id"], "completed")

    index_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/index",
    )
    assert index_response.status_code == 202
    index_job = _wait_for_job(
        client,
        index_response.json()["job_id"],
        "completed",
    )
    assert index_job["error_code"] is None

    with Session(engine) as session:
        documents = list(
            session.scalars(
                select(DocumentRecord).where(
                    DocumentRecord.workspace_id == workspace_id,
                )
            ).all()
        )

    assert len(documents) == 1
    assert documents[0].source_relative_path == "summary.pdf"
    assert documents[0].source_format == "pdf"


def test_document_index_job_rejects_malformed_pdf(
    document_index_client: tuple[TestClient, Engine],
    tmp_path: Path,
) -> None:
    client, engine = document_index_client
    workspace_root = tmp_path / "malformed-pdf-workspace"
    workspace_root.mkdir()
    (workspace_root / "broken.pdf").write_bytes(
        b"%PDF-1.7\nnot a valid PDF"
    )

    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "损坏 PDF 索引测试",
            "root_path": str(workspace_root),
        },
    )
    workspace_id = create_response.json()["id"]

    scan_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/scan",
    )
    assert scan_response.status_code == 202
    _wait_for_job(client, scan_response.json()["job_id"], "completed")

    index_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/index",
    )
    assert index_response.status_code == 202
    failed = _wait_for_job(client, index_response.json()["job_id"], "failed")
    assert failed["error_code"] == "document_index_failed"

    with Session(engine) as session:
        assert session.scalars(select(DocumentRecord)).first() is None


def test_document_index_job_supports_docx(
    document_index_client: tuple[TestClient, Engine],
    tmp_path: Path,
) -> None:
    client, engine = document_index_client
    workspace_root = tmp_path / "docx-document-workspace"
    workspace_root.mkdir()
    docx_file = workspace_root / "summary.docx"
    word_document = WordDocument()
    word_document.add_paragraph("DOCX text")
    word_document.save(docx_file)

    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "DOCX 文档索引测试",
            "root_path": str(workspace_root),
        },
    )
    workspace_id = create_response.json()["id"]

    scan_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/scan",
    )
    assert scan_response.status_code == 202
    _wait_for_job(client, scan_response.json()["job_id"], "completed")

    index_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/index",
    )
    assert index_response.status_code == 202
    index_job = _wait_for_job(
        client,
        index_response.json()["job_id"],
        "completed",
    )
    assert index_job["error_code"] is None

    with Session(engine) as session:
        documents = list(
            session.scalars(
                select(DocumentRecord).where(
                    DocumentRecord.workspace_id == workspace_id,
                )
            ).all()
        )

    assert len(documents) == 1
    assert documents[0].source_relative_path == "summary.docx"
    assert documents[0].source_format == "docx"
    assert documents[0].normalized_text == "DOCX text"


def test_document_index_job_rejects_malformed_docx(
    document_index_client: tuple[TestClient, Engine],
    tmp_path: Path,
) -> None:
    client, engine = document_index_client
    workspace_root = tmp_path / "malformed-docx-workspace"
    workspace_root.mkdir()
    (workspace_root / "broken.docx").write_bytes(b"not a valid DOCX package")

    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "损坏 DOCX 索引测试",
            "root_path": str(workspace_root),
        },
    )
    workspace_id = create_response.json()["id"]

    scan_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/scan",
    )
    assert scan_response.status_code == 202
    _wait_for_job(client, scan_response.json()["job_id"], "completed")

    index_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/index",
    )
    assert index_response.status_code == 202
    failed = _wait_for_job(client, index_response.json()["job_id"], "failed")
    assert failed["error_code"] == "document_index_failed"

    with Session(engine) as session:
        assert session.scalars(select(DocumentRecord)).first() is None


def test_document_index_job_failure_rolls_back_partial_index(
    document_index_client: tuple[TestClient, Engine],
    tmp_path: Path,
) -> None:
    client, engine = document_index_client
    workspace_root = tmp_path / "broken-document-workspace"
    workspace_root.mkdir()

    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "文档索引失败测试",
            "root_path": str(workspace_root),
        },
    )
    workspace_id = create_response.json()["id"]

    with Session(engine) as session:
        session.add(
            FileEntry(
                workspace_id=workspace_id,
                relative_path="missing.md",
                name="missing.md",
                extension=".md",
                size_bytes=1,
                mtime_ns=1,
            )
        )
        session.commit()

    response = client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/index",
    )
    assert response.status_code == 202
    failed = _wait_for_job(client, response.json()["job_id"], "failed")
    assert failed["error_code"] == "document_index_failed"

    with Session(engine) as session:
        assert session.scalars(select(DocumentRecord)).first() is None
