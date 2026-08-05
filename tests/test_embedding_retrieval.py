from collections.abc import Sequence
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.document_chunker import chunk_document
from backend.app.document_contracts import Document
from backend.app.embedding_client import EmbeddingBatch
from backend.app.embedding_retrieval import (
    VectorDimensionMismatchError,
    search_hybrid_chunks,
    search_vector_chunks,
)
from backend.app.fake_embedding_client import FakeEmbeddingClient
from backend.app.models import (
    ChunkEmbeddingRecord,
    ChunkRecord,
    DocumentRecord,
    FileEntry,
    Workspace,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_CONFIG_PATH = PROJECT_ROOT / "backend" / "alembic.ini"


def _upgrade_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database_path = tmp_path / "embedding-retrieval.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("FILENEST_DATABASE_URL", database_url)

    alembic_config = Config(str(ALEMBIC_CONFIG_PATH))
    command.upgrade(alembic_config, "head")
    return create_engine(database_url)


def _seed_index(
    session: Session,
    tmp_path: Path,
    entries: list[tuple[str, str, tuple[float, ...]]],
) -> int:
    workspace = Workspace(
        name="向量检索测试工作区",
        root_path=str(tmp_path / "workspace"),
    )
    session.add(workspace)
    session.flush()

    for index, (relative_path, text, vector) in enumerate(entries):
        file_entry = FileEntry(
            workspace_id=workspace.id,
            relative_path=relative_path,
            name=Path(relative_path).name,
            extension=Path(relative_path).suffix,
            size_bytes=len(text),
            mtime_ns=1_800_000_000_000_000_000 + index,
        )
        session.add(file_entry)
        session.flush()

        document = Document(
            document_id=uuid5(NAMESPACE_URL, f"embedding-test:{index}"),
            workspace_id=workspace.id,
            file_entry_id=file_entry.id,
            source_relative_path=relative_path,
            source_format="markdown",
            normalized_text=text,
            source_version=f"{index + 1:064x}",
            source_updated_at="2026-08-31T08:00:00+00:00",
        )
        chunks = chunk_document(document)
        assert len(chunks) == 1
        chunk = chunks[0]
        session.add(DocumentRecord.from_contract(document))
        session.add(ChunkRecord.from_contract(chunk))
        session.add(
            ChunkEmbeddingRecord.from_vector(
                chunk_id=str(chunk.chunk_id),
                embedding_model="fake-v1",
                vector=vector,
            )
        )

    session.commit()
    return workspace.id


def test_vector_search_returns_semantic_match_with_traceable_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _upgrade_database(tmp_path, monkeypatch)
    query = "文件整理前是否必须征得用户同意？"
    approval_text = "先生成预览，再等待用户审批；批准后才允许移动文件。"
    legacy_text = "旧计划只记录文件扫描。"
    client = FakeEmbeddingClient(
        {
            query: (1.0, 0.0),
            approval_text: (1.0, 0.0),
            legacy_text: (0.0, 1.0),
        }
    )

    try:
        with Session(engine) as session:
            workspace_id = _seed_index(
                session,
                tmp_path,
                [
                    ("guides/approval.md", approval_text, (1.0, 0.0)),
                    ("archive/legacy.md", legacy_text, (0.0, 1.0)),
                ],
            )
            results = search_vector_chunks(
                session,
                workspace_id=workspace_id,
                query=query,
                embedding_client=client,
                embedding_model="fake-v1",
                top_k=1,
            )

        assert len(results) == 1
        assert results[0].source_relative_path == "guides/approval.md"
        assert results[0].text == approval_text
        assert results[0].score == pytest.approx(1.0)
        assert results[0].vector_score == pytest.approx(1.0)
        assert results[0].keyword_score == 0.0
        assert results[0].start_offset == 0
        assert results[0].end_offset == len(approval_text)
        assert (results[0].start_line, results[0].end_line) == (1, 1)
    finally:
        engine.dispose()


def test_hybrid_search_can_use_keyword_signal_in_ranking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _upgrade_database(tmp_path, monkeypatch)
    query = "文件"
    semantic_text = "审批流程：等待用户审批后才允许移动。"
    keyword_text = "文件文件文件：旧记录。"
    client = FakeEmbeddingClient(
        {
            query: (1.0, 0.0),
            semantic_text: (1.0, 0.0),
            keyword_text: (0.8, 0.6),
        }
    )

    try:
        with Session(engine) as session:
            workspace_id = _seed_index(
                session,
                tmp_path,
                [
                    ("guides/approval.md", semantic_text, (1.0, 0.0)),
                    ("notes/keyword.md", keyword_text, (0.8, 0.6)),
                ],
            )
            vector_results = search_vector_chunks(
                session,
                workspace_id=workspace_id,
                query=query,
                embedding_client=client,
                embedding_model="fake-v1",
                top_k=2,
            )
            hybrid_results = search_hybrid_chunks(
                session,
                workspace_id=workspace_id,
                query=query,
                embedding_client=client,
                embedding_model="fake-v1",
                top_k=2,
            )

        assert vector_results[0].source_relative_path == "guides/approval.md"
        assert hybrid_results[0].source_relative_path == "notes/keyword.md"
        assert hybrid_results[0].keyword_score == pytest.approx(1.0)
        assert hybrid_results[1].keyword_score == pytest.approx(0.0)
    finally:
        engine.dispose()


def test_vector_search_rejects_query_dimension_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _upgrade_database(tmp_path, monkeypatch)
    text = "审批流程需要用户确认。"

    class MismatchedQueryClient:
        def embed(self, *, texts: Sequence[str]) -> EmbeddingBatch:
            assert texts == ("维度错误查询",)
            return ((1.0, 0.0, 0.0),)

    client = MismatchedQueryClient()

    try:
        with Session(engine) as session:
            workspace_id = _seed_index(
                session,
                tmp_path,
                [("guides/approval.md", text, (1.0, 0.0))],
            )

            with pytest.raises(
                VectorDimensionMismatchError,
                match="维度不一致",
            ):
                search_vector_chunks(
                    session,
                    workspace_id=workspace_id,
                    query="维度错误查询",
                    embedding_client=client,
                    embedding_model="fake-v1",
                )
    finally:
        engine.dispose()
