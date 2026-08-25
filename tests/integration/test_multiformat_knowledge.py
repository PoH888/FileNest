from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from uuid import UUID
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from fastapi import HTTPException
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app import agent_api
from backend.app.agent_api import NO_EVIDENCE_REFUSAL, ReadOnlyAgentRunExecutor
from backend.app.document_chunker import chunk_document
from backend.app.database import Base
from backend.app.document_indexer import (
    DocumentIndexWorkspaceNotFoundError,
    index_workspace_documents,
)
from backend.app.document_parser import DocumentParseError, load_document
from backend.app.filesystem_adapter import FileSystemAdapter
from backend.app.fake_model_client import FakeModelClient
from backend.app.knowledge_api import KnowledgeSearchRequest, search_knowledge
from backend.app.model_client import ModelMessage, ModelResponse, ModelToolCall
from backend.app.models import ChunkRecord, DocumentRecord, FileEntry, Workspace
from backend.app.tool_contracts import ToolResult


KNOWLEDGE_ANCHOR = "filenest-multiformat-anchor-27"
SUPPORTED_FORMATS = (".pdf", ".docx", ".md", ".txt")
SAMPLE_TEXT = f"FileNest knowledge anchor: {KNOWLEDGE_ANCHOR}."
DOCUMENT_ID = UUID("8a96e8c4-6ebc-4b5f-83b8-b27c681e8d95")
PARSE_CASES = (
    (".pdf", "pdf"),
    (".docx", "docx"),
    (".md", "markdown"),
    (".txt", "text"),
)


def _write_txt(path: Path) -> None:
    path.write_text(SAMPLE_TEXT + "\n", encoding="utf-8")


def _write_markdown(path: Path) -> None:
    path.write_text(
        f"# FileNest Knowledge Sample\n\n{SAMPLE_TEXT}\n",
        encoding="utf-8",
    )


def _write_pdf(path: Path) -> None:
    escaped_text = (
        SAMPLE_TEXT.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]

    document = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode("ascii"))
        document.extend(body)
        document.extend(b"\nendobj\n")

    xref_offset = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(document)


def _write_docx(path: Path) -> None:
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>{SAMPLE_TEXT}</w:t></w:r></w:p>
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>
""".encode("utf-8")
    content_types = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    relationships = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("word/document.xml", document_xml)


_WRITERS: dict[str, Callable[[Path], None]] = {
    ".pdf": _write_pdf,
    ".docx": _write_docx,
    ".md": _write_markdown,
    ".txt": _write_txt,
}


def _write_sample(path: Path) -> Path:
    suffix = path.suffix.lower()
    try:
        writer = _WRITERS[suffix]
    except KeyError as exc:
        raise ValueError(f"Unsupported knowledge format: {suffix}") from exc
    writer(path)
    return path


@pytest.fixture
def knowledge_samples(tmp_path: Path) -> dict[str, Path]:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    return {
        suffix: _write_sample(workspace_root / f"knowledge_sample{suffix}")
        for suffix in SUPPORTED_FORMATS
    }


@pytest.mark.parametrize("suffix", SUPPORTED_FORMATS)
def test_multiformat_samples_are_materialized(
    knowledge_samples: dict[str, Path],
    suffix: str,
) -> None:
    sample = knowledge_samples[suffix]

    assert sample.is_file()
    assert sample.suffix == suffix
    assert sample.stat().st_size > 0


def test_sample_builder_rejects_unsupported_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported knowledge format"):
        _write_sample(tmp_path / "knowledge_sample.csv")


@pytest.mark.parametrize(("suffix", "source_format"), PARSE_CASES)
def test_multiformat_documents_parse_into_normalized_documents(
    knowledge_samples: dict[str, Path],
    suffix: str,
    source_format: str,
) -> None:
    sample = knowledge_samples[suffix]
    adapter = FileSystemAdapter(sample.parent)

    document = load_document(
        adapter,
        workspace_id=27,
        file_entry_id=1,
        source_relative_path=sample.name,
        document_id=DOCUMENT_ID,
    )

    assert document.document_id == DOCUMENT_ID
    assert document.source_format == source_format
    assert document.source_relative_path == sample.name
    assert KNOWLEDGE_ANCHOR in document.normalized_text
    assert document.source_version == sha256(sample.read_bytes()).hexdigest()
    if suffix == ".pdf":
        assert len(document.pages) == 1
    elif suffix == ".docx":
        assert document.source_positions


def test_multiformat_parser_rejects_malformed_known_format(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    malformed_file = workspace_root / "broken.docx"
    workspace_root.mkdir()
    malformed_file.write_bytes(b"not a valid DOCX package")
    adapter = FileSystemAdapter(workspace_root)

    with pytest.raises(DocumentParseError, match="readable DOCX"):
        load_document(
            adapter,
            workspace_id=27,
            file_entry_id=1,
            source_relative_path=malformed_file.name,
            document_id=DOCUMENT_ID,
        )


@pytest.mark.parametrize("suffix", SUPPORTED_FORMATS)
def test_multiformat_documents_are_chunked_with_traceable_ranges(
    knowledge_samples: dict[str, Path],
    suffix: str,
) -> None:
    sample = knowledge_samples[suffix]
    adapter = FileSystemAdapter(sample.parent)
    document = load_document(
        adapter,
        workspace_id=27,
        file_entry_id=1,
        source_relative_path=sample.name,
        document_id=DOCUMENT_ID,
    )

    chunks = chunk_document(document, max_chars=32)

    assert chunks
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert "".join(chunk.text for chunk in chunks) == document.normalized_text
    assert all(
        chunk.document_id == document.document_id
        and chunk.file_entry_id == document.file_entry_id
        and chunk.source_relative_path == document.source_relative_path
        and document.normalized_text[chunk.start_offset : chunk.end_offset]
        == chunk.text
        for chunk in chunks
    )
    if suffix == ".pdf":
        assert all(chunk.page_start == 1 and chunk.page_end == 1 for chunk in chunks)
    elif suffix == ".docx":
        assert all(chunk.source_positions for chunk in chunks)


def test_multiformat_chunking_rejects_invalid_chunk_limit(
    knowledge_samples: dict[str, Path],
) -> None:
    sample = knowledge_samples[".txt"]
    document = load_document(
        FileSystemAdapter(sample.parent),
        workspace_id=27,
        file_entry_id=1,
        source_relative_path=sample.name,
        document_id=DOCUMENT_ID,
    )

    with pytest.raises(ValueError, match="positive integer"):
        chunk_document(document, max_chars=0)


def test_multiformat_documents_and_chunks_persist_with_source_metadata(
    knowledge_samples: dict[str, Path],
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "knowledge-persistence.db"
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    Base.metadata.create_all(bind=engine)
    expected_document_ids: dict[str, str] = {}
    try:
        with Session(engine) as session:
            workspace_root = next(iter(knowledge_samples.values())).parent
            workspace = Workspace(
                name="多格式 Knowledge 持久化测试",
                root_path=str(workspace_root),
            )
            session.add(workspace)
            session.flush()

            for index, (suffix, source_format) in enumerate(PARSE_CASES, start=1):
                sample = knowledge_samples[suffix]
                file_entry = FileEntry(
                    workspace_id=workspace.id,
                    relative_path=sample.name,
                    name=sample.name,
                    extension=suffix,
                    size_bytes=sample.stat().st_size,
                    mtime_ns=1_800_000_000_000_000_000 + index,
                )
                session.add(file_entry)
                session.flush()

                document_id = UUID(int=index)
                document = load_document(
                    FileSystemAdapter(sample.parent),
                    workspace_id=workspace.id,
                    file_entry_id=file_entry.id,
                    source_relative_path=sample.name,
                    document_id=document_id,
                )
                chunks = chunk_document(document, max_chars=32)
                session.add(DocumentRecord.from_contract(document))
                session.add_all(ChunkRecord.from_contract(chunk) for chunk in chunks)
                expected_document_ids[source_format] = str(document_id)

            session.commit()

        with Session(engine) as session:
            restored_documents = {
                document.source_format: document
                for document in session.scalars(select(DocumentRecord))
            }
            restored_chunks = list(
                session.scalars(
                    select(ChunkRecord).order_by(
                        ChunkRecord.document_id,
                        ChunkRecord.chunk_index,
                    )
                )
            )

        assert set(restored_documents) == {source_format for _, source_format in PARSE_CASES}
        chunks_by_document: dict[str, list[ChunkRecord]] = {}
        for chunk in restored_chunks:
            chunks_by_document.setdefault(chunk.document_id, []).append(chunk)

        for suffix, source_format in PARSE_CASES:
            document = restored_documents[source_format]
            chunks = chunks_by_document[expected_document_ids[source_format]]
            assert document.source_relative_path == knowledge_samples[suffix].name
            assert KNOWLEDGE_ANCHOR in document.normalized_text
            assert chunks
            assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
            assert "".join(chunk.text for chunk in chunks) == document.normalized_text
            assert all(
                chunk.file_entry_id == document.file_entry_id
                and chunk.source_relative_path == document.source_relative_path
                for chunk in chunks
            )
            if source_format == "pdf":
                assert all(chunk.page_start == 1 and chunk.page_end == 1 for chunk in chunks)
            elif source_format == "docx":
                assert all(chunk.source_positions_json for chunk in chunks)
    finally:
        engine.dispose()


def test_multiformat_persistence_rejects_duplicate_chunk_index(
    knowledge_samples: dict[str, Path],
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "knowledge-persistence-failure.db"
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    Base.metadata.create_all(bind=engine)
    try:
        sample = knowledge_samples[".txt"]
        with Session(engine) as session:
            workspace = Workspace(
                name="多格式 Knowledge 持久化失败测试",
                root_path=str(sample.parent),
            )
            session.add(workspace)
            session.flush()
            file_entry = FileEntry(
                workspace_id=workspace.id,
                relative_path=sample.name,
                name=sample.name,
                extension=".txt",
                size_bytes=sample.stat().st_size,
                mtime_ns=1_800_000_000_000_000_001,
            )
            session.add(file_entry)
            session.flush()
            document = load_document(
                FileSystemAdapter(sample.parent),
                workspace_id=workspace.id,
                file_entry_id=file_entry.id,
                source_relative_path=sample.name,
                document_id=UUID(int=27),
            )
            chunk = chunk_document(document, max_chars=32)[0]
            session.add(DocumentRecord.from_contract(document))
            session.add(ChunkRecord.from_contract(chunk))
            session.commit()

            session.add(ChunkRecord.from_contract(chunk))
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()

            saved_chunks = list(
                session.scalars(
                    select(ChunkRecord).where(
                        ChunkRecord.document_id == str(document.document_id)
                    )
                )
            )
            assert len(saved_chunks) == 1
    finally:
        engine.dispose()


def test_document_indexer_indexes_all_multiformat_sources_and_skips_same_versions(
    knowledge_samples: dict[str, Path],
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "knowledge-index.db"
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    Base.metadata.create_all(bind=engine)
    try:
        with Session(engine) as session:
            workspace_root = next(iter(knowledge_samples.values())).parent
            workspace = Workspace(
                name="多格式 Knowledge 索引测试",
                root_path=str(workspace_root),
            )
            session.add(workspace)
            session.flush()
            workspace_id = workspace.id
            for index, suffix in enumerate(SUPPORTED_FORMATS, start=1):
                sample = knowledge_samples[suffix]
                session.add(
                    FileEntry(
                        workspace_id=workspace_id,
                        relative_path=sample.name,
                        name=sample.name,
                        extension=suffix,
                        size_bytes=sample.stat().st_size,
                        mtime_ns=1_800_000_000_000_000_000 + index,
                    )
                )
            session.commit()

        with Session(engine) as session:
            first_result = index_workspace_documents(session, workspace_id)
        with Session(engine) as session:
            second_result = index_workspace_documents(session, workspace_id)

        assert first_result.indexed_documents == len(SUPPORTED_FORMATS)
        assert first_result.indexed_chunks == len(SUPPORTED_FORMATS)
        assert first_result.skipped_documents == 0
        assert second_result.indexed_documents == 0
        assert second_result.indexed_chunks == 0
        assert second_result.skipped_documents == len(SUPPORTED_FORMATS)

        with Session(engine) as session:
            documents = list(
                session.scalars(
                    select(DocumentRecord).order_by(DocumentRecord.source_relative_path)
                )
            )
            chunks = list(session.scalars(select(ChunkRecord)))

        assert [document.source_relative_path for document in documents] == [
            f"knowledge_sample{suffix}" for suffix in (".docx", ".md", ".pdf", ".txt")
        ]
        assert {document.source_format for document in documents} == {
            source_format for _, source_format in PARSE_CASES
        }
        assert all(
            document.ingest_status == "indexed"
            and KNOWLEDGE_ANCHOR in document.normalized_text
            for document in documents
        )
        assert len(chunks) == len(SUPPORTED_FORMATS)
    finally:
        engine.dispose()


def test_document_indexer_rejects_unknown_workspace(tmp_path: Path) -> None:
    database_path = tmp_path / "knowledge-index-failure.db"
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    Base.metadata.create_all(bind=engine)
    try:
        with Session(engine) as session:
            with pytest.raises(DocumentIndexWorkspaceNotFoundError):
                index_workspace_documents(session, workspace_id=999999)
    finally:
        engine.dispose()


def test_knowledge_search_retrieves_all_multiformat_chunks(
    knowledge_samples: dict[str, Path],
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "knowledge-search.db"
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    Base.metadata.create_all(bind=engine)
    try:
        with Session(engine) as session:
            workspace_root = next(iter(knowledge_samples.values())).parent
            workspace = Workspace(
                name="多格式 Knowledge 检索测试",
                root_path=str(workspace_root),
            )
            session.add(workspace)
            session.flush()
            workspace_id = workspace.id
            for index, suffix in enumerate(SUPPORTED_FORMATS, start=1):
                sample = knowledge_samples[suffix]
                session.add(
                    FileEntry(
                        workspace_id=workspace_id,
                        relative_path=sample.name,
                        name=sample.name,
                        extension=suffix,
                        size_bytes=sample.stat().st_size,
                        mtime_ns=1_800_000_000_000_000_000 + index,
                    )
                )
            session.commit()

        with Session(engine) as session:
            index_result = index_workspace_documents(session, workspace_id)
            response = search_knowledge(
                KnowledgeSearchRequest(
                    workspace_id=workspace_id,
                    query=KNOWLEDGE_ANCHOR,
                    top_k=10,
                ),
                session,
            )

        assert index_result.indexed_documents == len(SUPPORTED_FORMATS)
        assert response.query == KNOWLEDGE_ANCHOR
        assert response.total == len(SUPPORTED_FORMATS)
        assert response.has_more is False
        assert len(response.chunks) == len(SUPPORTED_FORMATS)
        assert len(response.documents) == len(SUPPORTED_FORMATS)
        assert len(response.relevance) == len(SUPPORTED_FORMATS)
        assert len(response.provenance) == len(SUPPORTED_FORMATS)
        assert {document.source_format for document in response.documents} == {
            source_format for _, source_format in PARSE_CASES
        }
        assert all(KNOWLEDGE_ANCHOR in chunk.text for chunk in response.chunks)
        assert all(
            provenance.start_offset < provenance.end_offset
            and provenance.start_line <= provenance.end_line
            for provenance in response.provenance
        )
    finally:
        engine.dispose()


def test_knowledge_search_rejects_unknown_workspace(tmp_path: Path) -> None:
    database_path = tmp_path / "knowledge-search-failure.db"
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    Base.metadata.create_all(bind=engine)
    try:
        with Session(engine) as session:
            with pytest.raises(HTTPException) as error:
                search_knowledge(
                    KnowledgeSearchRequest(
                        workspace_id=999999,
                        query=KNOWLEDGE_ANCHOR,
                    ),
                    session,
                )
        assert error.value.status_code == 404
    finally:
        engine.dispose()


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


def _prepare_indexed_workspace(
    engine: Engine,
    knowledge_samples: dict[str, Path],
) -> int:
    with Session(engine) as session:
        workspace_root = next(iter(knowledge_samples.values())).parent
        workspace = Workspace(
            name="多格式 Knowledge 回答测试",
            root_path=str(workspace_root),
        )
        session.add(workspace)
        session.flush()
        workspace_id = workspace.id
        for index, suffix in enumerate(SUPPORTED_FORMATS, start=1):
            sample = knowledge_samples[suffix]
            session.add(
                FileEntry(
                    workspace_id=workspace_id,
                    relative_path=sample.name,
                    name=sample.name,
                    extension=suffix,
                    size_bytes=sample.stat().st_size,
                    mtime_ns=1_800_000_000_000_000_000 + index,
                )
            )
        session.commit()

    with Session(engine) as session:
        result = index_workspace_documents(session, workspace_id)
    assert result.indexed_documents == len(SUPPORTED_FORMATS)
    return workspace_id


def test_multiformat_evidence_drives_agent_answer(
    knowledge_samples: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "knowledge-answer.db"
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(agent_api, "WORKFLOW_CHECKPOINT_PATH", tmp_path / "workflow.db")
    try:
        workspace_id = _prepare_indexed_workspace(engine, knowledge_samples)
        model_client = FakeModelClient(
            [
                _tool_call_response(
                    call_id="call_multiformat_knowledge",
                    name="knowledge_search",
                    arguments={
                        "workspace_id": workspace_id,
                        "query": KNOWLEDGE_ANCHOR,
                    },
                ),
                _final_response("四种格式文档都包含该知识锚点。"),
            ]
        )

        with Session(engine) as session:
            response = ReadOnlyAgentRunExecutor(lambda: model_client).run(
                session,
                workspace_id=workspace_id,
                request_text="请根据知识库回答这个问题。",
            )

        assert response.status == "completed"
        assert response.final_answer == "四种格式文档都包含该知识锚点。"
        assert len(response.sources) == len(SUPPORTED_FORMATS)
        assert KNOWLEDGE_ANCHOR in model_client.calls[1].messages[-1].content
    finally:
        engine.dispose()


def test_multiformat_agent_refuses_answer_without_knowledge_evidence(
    knowledge_samples: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "knowledge-answer-failure.db"
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(
        agent_api,
        "WORKFLOW_CHECKPOINT_PATH",
        tmp_path / "workflow-failure.db",
    )
    try:
        workspace_id = _prepare_indexed_workspace(engine, knowledge_samples)
        model_client = FakeModelClient(
            [
                _tool_call_response(
                    call_id="call_missing_knowledge",
                    name="knowledge_search",
                    arguments={
                        "workspace_id": workspace_id,
                        "query": "missing-knowledge-anchor",
                    },
                ),
                _final_response("根据常识回答，但没有文档依据。"),
            ]
        )

        with Session(engine) as session:
            response = ReadOnlyAgentRunExecutor(lambda: model_client).run(
                session,
                workspace_id=workspace_id,
                request_text="请回答没有文档依据的问题。",
            )

        assert response.status == "completed"
        assert response.final_answer == NO_EVIDENCE_REFUSAL
        assert response.sources == ()
    finally:
        engine.dispose()


def test_multiformat_agent_sources_preserve_file_locations_and_pdf_pages(
    knowledge_samples: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "knowledge-citation.db"
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(agent_api, "WORKFLOW_CHECKPOINT_PATH", tmp_path / "workflow.db")
    try:
        workspace_id = _prepare_indexed_workspace(engine, knowledge_samples)
        model_client = FakeModelClient(
            [
                _tool_call_response(
                    call_id="call_multiformat_citation",
                    name="knowledge_search",
                    arguments={
                        "workspace_id": workspace_id,
                        "query": KNOWLEDGE_ANCHOR,
                    },
                ),
                _final_response("答案包含四种格式的来源。"),
            ]
        )

        with Session(engine) as session:
            response = ReadOnlyAgentRunExecutor(lambda: model_client).run(
                session,
                workspace_id=workspace_id,
                request_text="请保留回答的文件位置和页码。",
            )

        sources_by_path = {source.relative_path: source for source in response.sources}
        expected_paths = {
            f"knowledge_sample{suffix}" for suffix in SUPPORTED_FORMATS
        }
        assert set(sources_by_path) == expected_paths
        assert all(
            source.file_id > 0
            and source.name == Path(source.relative_path).name
            and source.start_line is not None
            and source.end_line is not None
            and source.start_offset is not None
            and source.end_offset is not None
            for source in response.sources
        )
        assert sources_by_path["knowledge_sample.pdf"].page_start == 1
        assert sources_by_path["knowledge_sample.pdf"].page_end == 1
        assert all(
            sources_by_path[f"knowledge_sample{suffix}"].page_start is None
            and sources_by_path[f"knowledge_sample{suffix}"].page_end is None
            for suffix in (".docx", ".md", ".txt")
        )
    finally:
        engine.dispose()


def test_agent_citation_drops_incomplete_knowledge_source() -> None:
    tool_result = ToolResult.success(
        {
            "items": [
                {
                    "file_id": 1,
                    "name": "broken.pdf",
                    "source_relative_path": "broken.pdf",
                    "start_line": 1,
                    "end_line": 1,
                    "start_offset": 4,
                    "end_offset": 4,
                    "page_start": 1,
                    "page_end": 1,
                }
            ]
        }
    )
    messages = (
        ModelMessage(
            role="assistant",
            tool_calls=(
                ModelToolCall(
                    id="call_broken_citation",
                    name="knowledge_search",
                    arguments={},
                ),
            ),
        ),
        ModelMessage(
            role="tool",
            tool_call_id="call_broken_citation",
            content=tool_result.model_dump_json(),
        ),
    )

    assert agent_api._source_references(messages, workspace_id=27) == ()


def test_multiformat_knowledge_e2e_reaches_answer_with_citations(
    knowledge_samples: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "multiformat-knowledge-e2e.db"
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(
        agent_api,
        "WORKFLOW_CHECKPOINT_PATH",
        tmp_path / "workflow-e2e.db",
    )
    try:
        workspace_id = _prepare_indexed_workspace(engine, knowledge_samples)
        model_client = FakeModelClient(
            [
                _tool_call_response(
                    call_id="call_multiformat_e2e_search",
                    name="knowledge_search",
                    arguments={
                        "workspace_id": workspace_id,
                        "query": KNOWLEDGE_ANCHOR,
                    },
                ),
                _final_response("完整链路确认了四种格式的知识证据。"),
            ]
        )

        with Session(engine) as session:
            documents = list(
                session.scalars(
                    select(DocumentRecord).where(
                        DocumentRecord.workspace_id == workspace_id
                    )
                )
            )
            chunks = list(
                session.scalars(
                    select(ChunkRecord).where(
                        ChunkRecord.file_entry_id.in_(
                            [document.file_entry_id for document in documents]
                        )
                    )
                )
            )
            document_formats = {document.source_format for document in documents}
            document_texts = [document.normalized_text for document in documents]
            chunk_texts = [chunk.text for chunk in chunks]
            response = ReadOnlyAgentRunExecutor(lambda: model_client).run(
                session,
                workspace_id=workspace_id,
                request_text="请基于四种格式的知识证据回答。",
            )

        assert document_formats == {
            source_format for _, source_format in PARSE_CASES
        }
        assert len(documents) == len(SUPPORTED_FORMATS)
        assert len(chunks) == len(SUPPORTED_FORMATS)
        assert all(KNOWLEDGE_ANCHOR in text for text in document_texts)
        assert all(KNOWLEDGE_ANCHOR in text for text in chunk_texts)
        assert KNOWLEDGE_ANCHOR in model_client.calls[1].messages[-1].content
        assert response.status == "completed"
        assert response.final_answer == "完整链路确认了四种格式的知识证据。"
        assert {
            source.relative_path for source in response.sources
        } == {f"knowledge_sample{suffix}" for suffix in SUPPORTED_FORMATS}
    finally:
        engine.dispose()
