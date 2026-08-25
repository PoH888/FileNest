from collections.abc import Iterator
from pathlib import Path
from datetime import datetime, timezone
from time import monotonic, sleep
from uuid import uuid4

import pytest
from docx import Document as WordDocument
from pypdf import PdfWriter
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database import Base, get_session
from backend.app.main import app
from backend.app.models import (
    ChunkEmbeddingRecord,
    ChunkRecord,
    DocumentRecord,
    FileEntry,
)


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
    assert documents[0].ingest_status == "indexed"
    assert [chunk.text for chunk in chunks] == ["标题\n正文"]


def test_document_index_job_supports_txt(
    document_index_client: tuple[TestClient, Engine],
    tmp_path: Path,
) -> None:
    client, engine = document_index_client
    workspace_root = tmp_path / "txt-document-workspace"
    workspace_root.mkdir()
    (workspace_root / "notes.txt").write_text("纯文本内容", encoding="utf-8")

    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "TXT 文档索引测试",
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
        document = session.scalars(select(DocumentRecord)).one()

    assert document.source_relative_path == "notes.txt"
    assert document.source_format == "text"
    assert document.normalized_text == "纯文本内容"


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
        failed_document = session.scalars(select(DocumentRecord)).one()

    assert failed_document.ingest_status == "failed"
    assert failed_document.ingest_error is not None
    assert "readable PDF" in failed_document.ingest_error


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
        failed_document = session.scalars(select(DocumentRecord)).one()

    assert failed_document.ingest_status == "failed"
    assert failed_document.ingest_error is not None
    assert "readable DOCX" in failed_document.ingest_error


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
        failed_document = session.scalars(select(DocumentRecord)).one()

    assert failed_document.ingest_status == "failed"
    assert failed_document.ingest_error is not None
    assert failed_document.ingest_error.strip()


def test_knowledge_index_endpoint_creates_background_job(
    document_index_client: tuple[TestClient, Engine],
    tmp_path: Path,
) -> None:
    client, _engine = document_index_client
    workspace_root = tmp_path / "knowledge-index-workspace"
    workspace_root.mkdir()

    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "Knowledge 索引 API 测试",
            "root_path": str(workspace_root),
        },
    )
    workspace_id = create_response.json()["id"]

    response = client.post(
        "/api/v1/knowledge/index",
        json={"workspace_id": workspace_id},
    )

    assert response.status_code == 202
    job = _wait_for_job(client, response.json()["job_id"], "completed")
    assert job["error_code"] is None


def test_knowledge_index_endpoint_rejects_unknown_workspace(
    document_index_client: tuple[TestClient, Engine],
) -> None:
    client, _engine = document_index_client

    response = client.post(
        "/api/v1/knowledge/index",
        json={"workspace_id": 999999},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "workspace_not_found"


def test_knowledge_documents_endpoint_lists_indexed_documents(
    document_index_client: tuple[TestClient, Engine],
    tmp_path: Path,
) -> None:
    client, engine = document_index_client
    workspace_root = tmp_path / "knowledge-documents-workspace"
    workspace_root.mkdir()

    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "Knowledge 文档列表测试",
            "root_path": str(workspace_root),
        },
    )
    workspace_id = create_response.json()["id"]

    with Session(engine) as session:
        file_entry = FileEntry(
            workspace_id=workspace_id,
            relative_path="notes.md",
            name="notes.md",
            extension=".md",
            size_bytes=5,
            mtime_ns=1,
        )
        session.add(file_entry)
        session.flush()
        session.add(
            DocumentRecord(
                document_id=str(uuid4()),
                workspace_id=workspace_id,
                file_entry_id=file_entry.id,
                source_relative_path="notes.md",
                ingest_status="indexed",
                source_format="markdown",
                normalized_text="正文",
            )
        )
        session.commit()

    response = client.get(
        "/api/v1/knowledge/documents",
        params={"workspace_id": workspace_id},
    )

    assert response.status_code == 200
    assert response.json()[0]["workspace_id"] == workspace_id
    assert response.json()[0]["source_relative_path"] == "notes.md"


def test_knowledge_documents_endpoint_returns_empty_for_unknown_workspace(
    document_index_client: tuple[TestClient, Engine],
) -> None:
    client, _engine = document_index_client

    response = client.get(
        "/api/v1/knowledge/documents",
        params={"workspace_id": 999999},
    )

    assert response.status_code == 200
    assert response.json() == []


def test_knowledge_document_endpoint_returns_metadata_status_and_provenance(
    document_index_client: tuple[TestClient, Engine],
    tmp_path: Path,
) -> None:
    client, engine = document_index_client
    workspace_root = tmp_path / "knowledge-document-detail-workspace"
    workspace_root.mkdir()

    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "Knowledge 文档详情测试",
            "root_path": str(workspace_root),
        },
    )
    workspace_id = create_response.json()["id"]

    with Session(engine) as session:
        file_entry = FileEntry(
            workspace_id=workspace_id,
            relative_path="manual.md",
            name="manual.md",
            extension=".md",
            size_bytes=5,
            mtime_ns=1,
        )
        session.add(file_entry)
        session.flush()
        document = DocumentRecord(
            document_id=str(uuid4()),
            workspace_id=workspace_id,
            file_entry_id=file_entry.id,
            source_relative_path="manual.md",
            ingest_status="indexed",
            source_format="markdown",
            normalized_text="正文",
            source_version="a" * 64,
            source_updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        session.add(document)
        session.commit()
        document_id = document.document_id

    response = client.get(f"/api/v1/knowledge/documents/{document_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["source_relative_path"] == "manual.md"
    assert payload["ingest_status"] == "indexed"
    assert payload["provenance"]["source_version"] == "a" * 64
    assert "root_path" not in str(payload)


def test_knowledge_document_endpoint_rejects_unknown_document(
    document_index_client: tuple[TestClient, Engine],
) -> None:
    client, _engine = document_index_client
    document_id = uuid4()

    response = client.get(f"/api/v1/knowledge/documents/{document_id}")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "document_not_found"


def test_knowledge_search_endpoint_returns_chunks_documents_relevance_and_provenance(
    document_index_client: tuple[TestClient, Engine],
    tmp_path: Path,
) -> None:
    client, engine = document_index_client
    workspace_root = tmp_path / "knowledge-search-workspace"
    workspace_root.mkdir()

    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "Knowledge 搜索测试",
            "root_path": str(workspace_root),
        },
    )
    workspace_id = create_response.json()["id"]

    with Session(engine) as session:
        file_entry = FileEntry(
            workspace_id=workspace_id,
            relative_path="guide.md",
            name="guide.md",
            extension=".md",
            size_bytes=20,
            mtime_ns=1,
        )
        session.add(file_entry)
        session.flush()
        document = DocumentRecord(
            document_id=str(uuid4()),
            workspace_id=workspace_id,
            file_entry_id=file_entry.id,
            source_relative_path="guide.md",
            ingest_status="indexed",
            source_format="markdown",
            normalized_text="Knowledge guide",
        )
        session.add(document)
        session.flush()
        session.add(
            ChunkRecord(
                chunk_id=str(uuid4()),
                document_id=document.document_id,
                file_entry_id=file_entry.id,
                source_relative_path="guide.md",
                chunk_index=0,
                text="Knowledge guide",
                start_offset=0,
                end_offset=len("Knowledge guide"),
                start_line=1,
                end_line=1,
            )
        )
        session.commit()

    response = client.post(
        "/api/v1/knowledge/search",
        json={
            "workspace_id": workspace_id,
            "query": " knowledge ",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "knowledge"
    assert payload["chunks"][0]["text"] == "Knowledge guide"
    assert payload["documents"][0]["source_relative_path"] == "guide.md"
    assert payload["relevance"][0]["score"] == 1
    assert payload["provenance"][0]["source_relative_path"] == "guide.md"


def test_knowledge_search_endpoint_rejects_unknown_workspace(
    document_index_client: tuple[TestClient, Engine],
) -> None:
    client, _engine = document_index_client

    response = client.post(
        "/api/v1/knowledge/search",
        json={
            "workspace_id": 999999,
            "query": "anything",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "workspace_not_found"


def test_knowledge_document_delete_removes_index_and_preserves_source_file(
    document_index_client: tuple[TestClient, Engine],
    tmp_path: Path,
) -> None:
    client, engine = document_index_client
    workspace_root = tmp_path / "knowledge-delete-workspace"
    workspace_root.mkdir()
    source_file = workspace_root / "keep.md"
    source_file.write_text("保留原始文件", encoding="utf-8")

    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "Knowledge 删除测试",
            "root_path": str(workspace_root),
        },
    )
    workspace_id = create_response.json()["id"]

    with Session(engine) as session:
        file_entry = FileEntry(
            workspace_id=workspace_id,
            relative_path="keep.md",
            name="keep.md",
            extension=".md",
            size_bytes=18,
            mtime_ns=1,
        )
        session.add(file_entry)
        session.flush()
        document = DocumentRecord(
            document_id=str(uuid4()),
            workspace_id=workspace_id,
            file_entry_id=file_entry.id,
            source_relative_path="keep.md",
            ingest_status="indexed",
            source_format="markdown",
            normalized_text="保留原始文件",
        )
        session.add(document)
        session.flush()
        chunk = ChunkRecord(
            chunk_id=str(uuid4()),
            document_id=document.document_id,
            file_entry_id=file_entry.id,
            source_relative_path="keep.md",
            chunk_index=0,
            text="保留原始文件",
            start_offset=0,
            end_offset=6,
            start_line=1,
            end_line=1,
        )
        session.add(chunk)
        session.flush()
        session.add(
            ChunkEmbeddingRecord.from_vector(
                chunk_id=chunk.chunk_id,
                embedding_model="test-model",
                vector=[1.0, 0.0],
            )
        )
        session.commit()
        document_id = document.document_id

    response = client.delete(f"/api/v1/knowledge/documents/{document_id}")

    assert response.status_code == 204
    assert source_file.exists()
    with Session(engine) as session:
        assert session.get(DocumentRecord, document_id) is None
        assert session.scalars(select(ChunkRecord)).all() == []
        assert session.scalars(select(ChunkEmbeddingRecord)).all() == []


def test_knowledge_document_delete_rejects_unknown_document(
    document_index_client: tuple[TestClient, Engine],
) -> None:
    client, _engine = document_index_client

    response = client.delete(f"/api/v1/knowledge/documents/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "document_not_found"
