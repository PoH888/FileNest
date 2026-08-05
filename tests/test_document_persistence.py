from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.document_chunker import chunk_document
from backend.app.document_contracts import Document
from backend.app.models import (
    ChunkRecord,
    ChunkEmbeddingRecord,
    DocumentRecord,
    FileEntry,
    Workspace,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_CONFIG_PATH = PROJECT_ROOT / "backend" / "alembic.ini"
DOCUMENT_ID = UUID("df9d9c2b-4f4f-4a2a-8756-f1b19cb3d7d2")
SECOND_DOCUMENT_ID = UUID("ba1e1bc6-4f84-4a6f-9c86-5b6ef6ba18c0")
SOURCE_UPDATED_AT = datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc)


def _upgrade_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database_path = tmp_path / "document-persistence.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("FILENEST_DATABASE_URL", database_url)

    alembic_config = Config(str(ALEMBIC_CONFIG_PATH))
    command.upgrade(alembic_config, "head")
    return create_engine(database_url)


def test_document_and_chunks_persist_traceability_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _upgrade_database(tmp_path, monkeypatch)
    try:
        with Session(engine) as session:
            workspace = Workspace(
                name="文档持久化测试",
                root_path=str(tmp_path / "workspace"),
            )
            session.add(workspace)
            session.flush()

            file_entry = FileEntry(
                workspace_id=workspace.id,
                relative_path="notes/project.md",
                name="project.md",
                extension=".md",
                size_bytes=12,
                mtime_ns=1_800_000_000_000_000_000,
            )
            session.add(file_entry)
            session.flush()
            file_entry_id = file_entry.id

            document = Document(
                document_id=DOCUMENT_ID,
                workspace_id=workspace.id,
                file_entry_id=file_entry_id,
                source_relative_path=file_entry.relative_path,
                source_format="markdown",
                normalized_text="one\ntwo\nthree",
                source_version="c" * 64,
                source_updated_at=SOURCE_UPDATED_AT,
            )
            chunks = chunk_document(document, max_chars=8)
            session.add(DocumentRecord.from_contract(document))
            session.add_all(
                ChunkRecord.from_contract(chunk) for chunk in chunks
            )
            session.commit()

        with Session(engine) as session:
            restored_document = session.get(
                DocumentRecord,
                str(DOCUMENT_ID),
            )
            restored_chunks = list(
                session.scalars(
                    select(ChunkRecord)
                    .where(ChunkRecord.document_id == str(DOCUMENT_ID))
                    .order_by(ChunkRecord.chunk_index)
                )
            )

            assert restored_document is not None
            assert restored_document.file_entry_id == file_entry_id
            assert restored_document.source_relative_path == (
                "notes/project.md"
            )
            assert restored_document.source_version == "c" * 64
            assert restored_document.source_updated_at == (
                SOURCE_UPDATED_AT.replace(tzinfo=None)
            )
            assert restored_document.normalized_text == (
                "one\ntwo\nthree"
            )
            assert [chunk.text for chunk in restored_chunks] == [
                "one\ntwo\n",
                "three",
            ]
            assert [
                (chunk.start_offset, chunk.end_offset)
                for chunk in restored_chunks
            ] == [(0, 8), (8, 13)]
            assert all(
                chunk.document_id == restored_document.document_id
                and chunk.file_entry_id == file_entry_id
                and chunk.source_relative_path == "notes/project.md"
                for chunk in restored_chunks
            )
    finally:
        engine.dispose()


def test_chunk_embedding_persists_vector_and_rejects_duplicate_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _upgrade_database(tmp_path, monkeypatch)
    try:
        with Session(engine) as session:
            workspace = Workspace(
                name="片段向量持久化测试",
                root_path=str(tmp_path / "workspace"),
            )
            session.add(workspace)
            session.flush()

            file_entry = FileEntry(
                workspace_id=workspace.id,
                relative_path="notes/project.md",
                name="project.md",
                extension=".md",
                size_bytes=12,
                mtime_ns=1_800_000_000_000_000_000,
            )
            session.add(file_entry)
            session.flush()

            document = Document(
                document_id=DOCUMENT_ID,
                workspace_id=workspace.id,
                file_entry_id=file_entry.id,
                source_relative_path=file_entry.relative_path,
                source_format="markdown",
                normalized_text="one\ntwo\nthree",
                source_version="e" * 64,
                source_updated_at=SOURCE_UPDATED_AT,
            )
            chunk = chunk_document(document, max_chars=8)[0]
            session.add(DocumentRecord.from_contract(document))
            session.add(ChunkRecord.from_contract(chunk))
            session.flush()

            embedding = ChunkEmbeddingRecord.from_vector(
                chunk_id=str(chunk.chunk_id),
                embedding_model="fake-v1",
                vector=(0.25, 0.75),
            )
            session.add(embedding)
            session.commit()
            embedding_id = embedding.id

        with Session(engine) as session:
            restored = session.get(ChunkEmbeddingRecord, embedding_id)

            assert restored is not None
            assert restored.chunk_id == str(chunk.chunk_id)
            assert restored.embedding_model == "fake-v1"
            assert restored.dimension == 2
            assert restored.vector_json == "[0.25,0.75]"
            assert restored.vector == (0.25, 0.75)

            session.add(
                ChunkEmbeddingRecord.from_vector(
                    chunk_id=str(chunk.chunk_id),
                    embedding_model="fake-v1",
                    vector=(0.25, 0.75),
                )
            )
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()
    finally:
        engine.dispose()


def test_same_file_version_cannot_be_imported_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _upgrade_database(tmp_path, monkeypatch)
    try:
        with Session(engine) as session:
            workspace = Workspace(
                name="重复导入测试",
                root_path=str(tmp_path / "workspace"),
            )
            session.add(workspace)
            session.flush()

            file_entry = FileEntry(
                workspace_id=workspace.id,
                relative_path="notes/project.md",
                name="project.md",
                extension=".md",
                size_bytes=12,
                mtime_ns=1_800_000_000_000_000_000,
            )
            session.add(file_entry)
            session.flush()
            file_entry_id = file_entry.id

            document = Document(
                document_id=DOCUMENT_ID,
                workspace_id=workspace.id,
                file_entry_id=file_entry_id,
                source_relative_path=file_entry.relative_path,
                source_format="markdown",
                normalized_text="one\ntwo",
                source_version="d" * 64,
                source_updated_at=SOURCE_UPDATED_AT,
            )
            duplicate = document.model_copy(
                update={"document_id": SECOND_DOCUMENT_ID}
            )
            session.add(DocumentRecord.from_contract(document))
            session.commit()

            session.add(DocumentRecord.from_contract(duplicate))
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()

            saved_documents = list(
                session.scalars(
                    select(DocumentRecord).where(
                        DocumentRecord.file_entry_id == file_entry_id
                    )
                )
            )
            assert [saved.document_id for saved in saved_documents] == [
                str(DOCUMENT_ID)
            ]
    finally:
        engine.dispose()


def test_document_chunks_reject_invalid_persisted_position(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _upgrade_database(tmp_path, monkeypatch)
    try:
        with Session(engine) as session:
            session.add(
                ChunkRecord(
                    chunk_id="1a6e2a5b-b413-4d77-ae76-5ac24d46c36e",
                    document_id=str(DOCUMENT_ID),
                    file_entry_id=7,
                    source_relative_path="notes/project.md",
                    chunk_index=0,
                    text="invalid",
                    start_offset=4,
                    end_offset=4,
                    start_line=1,
                    end_line=1,
                )
            )

            with pytest.raises(IntegrityError):
                session.commit()
    finally:
        engine.dispose()
