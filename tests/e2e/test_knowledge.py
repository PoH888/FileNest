from collections.abc import Iterator
from pathlib import Path
from time import monotonic, sleep

import pytest
from docx import Document as WordDocument
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_api import (
    NO_EVIDENCE_REFUSAL,
    ReadOnlyAgentRunExecutor,
)
from backend.app.database import Base, get_session
from backend.app.fake_model_client import FakeModelClient
from backend.app.main import app
from backend.app.model_client import ModelMessage, ModelResponse, ModelToolCall
from backend.app.tool_contracts import ToolResult


@pytest.fixture
def knowledge_e2e_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, Engine, sessionmaker[Session]]]:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'knowledge-e2e.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    def override_get_session() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    with TestClient(app) as client:
        yield client, engine, session_factory

    app.dependency_overrides.clear()
    engine.dispose()


def _wait_for_job(
    client: TestClient,
    job_id: str,
    expected_status: str,
    workspace_id: int,
) -> dict[str, object]:
    deadline = monotonic() + 5
    while monotonic() < deadline:
        response = client.get(
            f"/api/v1/jobs/{job_id}",
            params={"workspace_id": workspace_id},
        )
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] == expected_status:
            return payload
        sleep(0.01)
    pytest.fail(f"job did not reach status {expected_status!r}")


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


def _write_text_pdf(path: Path, text: str) -> None:
    escaped_text = (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .encode("ascii")
    )
    content = b"BT /F1 12 Tf 72 720 Td (" + escaped_text + b") Tj ET"
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length "
        + str(len(content)).encode("ascii")
        + b" >>\nstream\n"
        + content
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    )
    parts = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets = [0]
    for object_number, obj in enumerate(objects, start=1):
        offsets.append(sum(len(part) for part in parts))
        parts.extend(
            (
                f"{object_number} 0 obj\n".encode("ascii"),
                obj,
                b"\nendobj\n",
            )
        )

    xref_offset = sum(len(part) for part in parts)
    parts.append(b"xref\n0 6\n0000000000 65535 f \n")
    parts.extend(
        f"{offset:010d} 00000 n \n".encode("ascii")
        for offset in offsets[1:]
    )
    parts.extend(
        (
            b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n",
            str(xref_offset).encode("ascii"),
            b"\n%%EOF\n",
        )
    )
    path.write_bytes(b"".join(parts))


def _create_four_documents(workspace_root: Path, token: str) -> None:
    workspace_root.mkdir()
    (workspace_root / "manual.md").write_text(
        f"Markdown evidence: {token}",
        encoding="utf-8",
    )
    (workspace_root / "notes.txt").write_text(
        f"Text evidence: {token}",
        encoding="utf-8",
    )
    docx = WordDocument()
    docx.add_paragraph(f"DOCX evidence: {token}")
    docx.save(workspace_root / "guide.docx")
    _write_text_pdf(workspace_root / "report.pdf", f"PDF evidence: {token}")


def test_four_document_formats_index_retrieve_answer_with_citations(
    knowledge_e2e_client: tuple[TestClient, Engine, sessionmaker[Session]],
    tmp_path: Path,
) -> None:
    client, _engine, session_factory = knowledge_e2e_client
    token = "phase38-knowledge-token"
    workspace_root = tmp_path / "four-format-workspace"
    _create_four_documents(workspace_root, token)

    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "Phase 38 Knowledge E2E",
            "root_path": str(workspace_root),
        },
    )
    assert create_response.status_code == 201
    workspace_id = create_response.json()["id"]

    scan_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/scan",
    )
    assert scan_response.status_code == 202
    assert _wait_for_job(
        client,
        scan_response.json()["job_id"],
        "completed",
        workspace_id,
    )["error_code"] is None

    index_response = client.post(
        "/api/v1/knowledge/index",
        json={"workspace_id": workspace_id},
    )
    assert index_response.status_code == 202
    assert _wait_for_job(
        client,
        index_response.json()["job_id"],
        "completed",
        workspace_id,
    )["error_code"] is None

    documents_response = client.get(
        "/api/v1/knowledge/documents",
        params={"workspace_id": workspace_id},
    )
    assert documents_response.status_code == 200
    documents = documents_response.json()
    assert len(documents) == 4
    assert {
        document["source_format"] for document in documents
    } == {"pdf", "docx", "markdown", "text"}

    search_response = client.post(
        "/api/v1/knowledge/search",
        json={
            "workspace_id": workspace_id,
            "query": token,
            "top_k": 10,
        },
    )
    assert search_response.status_code == 200
    search_payload = search_response.json()
    assert search_payload["total"] == 4
    assert len(search_payload["chunks"]) == 4
    assert {
        document["source_relative_path"]
        for document in search_payload["documents"]
    } == {"guide.docx", "manual.md", "notes.txt", "report.pdf"}
    assert len(search_payload["provenance"]) == 4

    model_client = FakeModelClient(
        [
            _tool_call_response(
                call_id="call_four_format_knowledge_search",
                name="knowledge_search",
                arguments={
                    "workspace_id": workspace_id,
                    "query": token,
                    "top_k": 10,
                },
            ),
            _final_response("四种文档均提供了同一条证据。"),
        ]
    )
    with session_factory() as session:
        agent_response = ReadOnlyAgentRunExecutor(
            lambda: model_client,
        ).run(
            session,
            workspace_id=workspace_id,
            request_text="请根据四种文档回答并给出引用。",
        )

    assert agent_response.status == "completed"
    assert agent_response.final_answer == "四种文档均提供了同一条证据。"
    assert {
        source.relative_path for source in agent_response.sources
    } == {"guide.docx", "manual.md", "notes.txt", "report.pdf"}
    pdf_sources = [
        source
        for source in agent_response.sources
        if source.relative_path == "report.pdf"
    ]
    assert len(pdf_sources) == 1
    assert (pdf_sources[0].page_start, pdf_sources[0].page_end) == (1, 1)
    tool_message = model_client.calls[1].messages[-1].content
    tool_result = ToolResult.model_validate_json(tool_message)
    assert tool_result.ok is True
    assert isinstance(tool_result.data, dict)
    assert tool_result.data["total"] == 4


def test_knowledge_answer_refuses_without_retrieved_evidence(
    knowledge_e2e_client: tuple[TestClient, Engine, sessionmaker[Session]],
    tmp_path: Path,
) -> None:
    client, _engine, session_factory = knowledge_e2e_client
    workspace_root = tmp_path / "empty-knowledge-workspace"
    workspace_root.mkdir()
    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "name": "Phase 38 Empty Knowledge E2E",
            "root_path": str(workspace_root),
        },
    )
    assert create_response.status_code == 201
    workspace_id = create_response.json()["id"]

    model_client = FakeModelClient(
        [
            _tool_call_response(
                call_id="call_empty_knowledge_search",
                name="knowledge_search",
                arguments={
                    "workspace_id": workspace_id,
                    "query": "missing-evidence-token",
                },
            ),
            _final_response("根据常识回答。"),
        ]
    )
    with session_factory() as session:
        agent_response = ReadOnlyAgentRunExecutor(
            lambda: model_client,
        ).run(
            session,
            workspace_id=workspace_id,
            request_text="没有文档证据时请回答。",
        )

    assert agent_response.status == "completed"
    assert agent_response.final_answer == NO_EVIDENCE_REFUSAL
    assert agent_response.sources == ()
