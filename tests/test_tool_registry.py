from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.document_chunker import chunk_document
from backend.app.document_contracts import Document, DocumentPage
from backend.app.models import ChunkRecord, DocumentRecord, FileEntry, Workspace
from backend.app.read_tools import (
    build_find_similar_folders_tool,
    build_list_directory_tool,
)
from backend.app.tool_contracts import Tool, ToolResult
from backend.app.tool_registry import ToolRegistry, build_read_tool_registry


class EmptyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'tool-registry.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _seed_knowledge_index(session: Session) -> int:
    workspace = Workspace(
        name="知识搜索工作区",
        root_path="D:/Private/KnowledgeSearch",
    )
    other_workspace = Workspace(
        name="其他知识工作区",
        root_path="D:/Private/OtherKnowledgeSearch",
    )
    session.add_all([workspace, other_workspace])
    session.flush()

    entries = [
        (
            workspace,
            "guides/approval.md",
            "第一行：审批流程\n第二行：批准后才能移动文件。",
        ),
        (
            other_workspace,
            "private/other.md",
            "其他工作区：审批信息不应泄露。",
        ),
    ]
    for index, (owner, relative_path, text) in enumerate(entries, start=1):
        file_entry = FileEntry(
            workspace_id=owner.id,
            relative_path=relative_path,
            name=Path(relative_path).name,
            extension=Path(relative_path).suffix,
            size_bytes=len(text.encode("utf-8")),
            mtime_ns=1_800_000_000_000_000_000 + index,
        )
        session.add(file_entry)
        session.flush()

        document = Document(
            document_id=uuid4(),
            workspace_id=owner.id,
            file_entry_id=file_entry.id,
            source_relative_path=relative_path,
            source_format="markdown",
            normalized_text=text,
            source_version=f"{index:064x}",
            source_updated_at="2026-09-01T00:00:00+00:00",
        )
        session.add(DocumentRecord.from_contract(document))
        session.add_all(
            ChunkRecord.from_contract(chunk)
            for chunk in chunk_document(document)
        )

    session.commit()
    return workspace.id


def _fake_tool(name: str, handler_called: list[bool] | None = None) -> Tool:
    def handle(_: BaseModel) -> ToolResult:
        if handler_called is not None:
            handler_called.append(True)
        return ToolResult.success({"tool": name})

    return Tool(
        name=name,
        description=f"测试工具 {name}",
        arguments_model=EmptyArguments,
        handler=handle,
    )


def test_read_tool_registry_contains_only_formal_read_tools() -> None:
    with Session() as session:
        registry = build_read_tool_registry(session)

    assert registry.names == (
        "list_workspaces",
        "list_directory",
        "find_similar_folders",
        "search_files",
        "get_file_metadata",
        "knowledge_search",
    )


def test_registry_definitions_expose_schema_but_not_handlers() -> None:
    with Session() as session:
        registry = build_read_tool_registry(session)

    definitions = [definition.model_dump() for definition in registry.definitions()]

    assert [definition["name"] for definition in definitions] == [
        "list_workspaces",
        "list_directory",
        "find_similar_folders",
        "search_files",
        "get_file_metadata",
        "knowledge_search",
    ]
    assert all(
        definition["parameters"]["additionalProperties"] is False
        for definition in definitions
    )
    assert all("handler" not in definition for definition in definitions)


def test_registry_dispatches_registered_list_workspaces_tool(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        session.add(
            Workspace(
                name="注册表工作区",
                root_path="D:/Private/RegistryWorkspace",
            )
        )
        session.commit()
        registry = build_read_tool_registry(session)

        result = registry.invoke("list_workspaces", {})

    assert result.ok is True
    assert result.data == {
        "items": [{"id": 1, "name": "注册表工作区"}],
        "count": 1,
    }
    assert "root_path" not in result.model_dump_json()


def test_list_directory_tool_returns_safe_metadata_and_ignore_reasons(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "directory-tool-workspace"
    workspace_root.mkdir()
    (workspace_root / "visible.txt").write_text("visible", encoding="utf-8")
    (workspace_root / "nested").mkdir()
    (workspace_root / ".env").write_text("secret", encoding="utf-8")
    (workspace_root / "denied.txt").write_text("private", encoding="utf-8")

    with Session(engine) as session:
        workspace = Workspace(
            name="目录工具工作区",
            root_path=str(workspace_root),
        )
        session.add(workspace)
        session.commit()
        result = build_list_directory_tool(
            session,
            user_denylist=(workspace_root / "denied.txt",),
        ).invoke({"workspace_id": workspace.id})

    assert result.ok is True
    assert isinstance(result.data, dict)
    items = result.data["items"]
    by_path = {item["relative_path"]: item for item in items}
    assert by_path["visible.txt"]["entry_type"] == "file"
    assert by_path["visible.txt"]["size_bytes"] == len("visible")
    assert by_path["nested"]["entry_type"] == "directory"
    assert by_path[".env"] == {
        "relative_path": ".env",
        "entry_type": "ignored",
        "size_bytes": None,
        "modified_at": None,
        "ignored_reason": "sensitive_path",
    }
    assert by_path["denied.txt"]["ignored_reason"] == "path_denylisted"
    assert "secret" not in result.model_dump_json()
    assert str(workspace_root) not in result.model_dump_json()


def test_list_directory_tool_supports_bounded_cursor_and_stable_errors(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "directory-tool-paging-workspace"
    workspace_root.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        (workspace_root / name).write_text(name, encoding="utf-8")

    with Session(engine) as session:
        workspace = Workspace(
            name="目录分页工作区",
            root_path=str(workspace_root),
        )
        session.add(workspace)
        session.commit()
        tool = build_list_directory_tool(session)
        first = tool.invoke(
            {"workspace_id": workspace.id, "page_size": 2}
        )
        second = tool.invoke(
            {
                "workspace_id": workspace.id,
                "page_size": 2,
                "cursor": first.data["next_cursor"],
            }
        )
        missing = tool.invoke(
            {
                "workspace_id": workspace.id,
                "relative_directory": "missing",
            }
        )
        invalid = tool.invoke(
            {
                "workspace_id": workspace.id,
                "relative_directory": "../",
            }
        )

    assert first.ok is True and second.ok is True
    assert [item["relative_path"] for item in first.data["items"]] == [
        "a.txt",
        "b.txt",
    ]
    assert [item["relative_path"] for item in second.data["items"]] == [
        "c.txt",
    ]
    assert first.data["has_more"] is True
    assert second.data["has_more"] is False
    assert missing.error is not None
    assert missing.error.code == "directory_not_found"
    assert invalid.error is not None
    assert invalid.error.code == "invalid_arguments"


def test_find_similar_folders_returns_explainable_index_candidates(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "similar-folders-workspace"
    for relative_directory in (
        "inbox",
        "reports/quarterly",
        "archive",
    ):
        (workspace_root / relative_directory).mkdir(parents=True)
    source_path = workspace_root / "inbox" / "quarterly-report.txt"
    source_path.write_text("keep unchanged", encoding="utf-8")
    (workspace_root / "reports" / "quarterly" / "summary.txt").write_text(
        "candidate",
        encoding="utf-8",
    )
    (workspace_root / "archive" / "old.pdf").write_text(
        "other candidate",
        encoding="utf-8",
    )

    with Session(engine) as session:
        workspace = Workspace(
            name="相似目录工作区",
            root_path=str(workspace_root),
        )
        session.add(workspace)
        session.flush()
        source_metadata = source_path.stat()
        candidate_metadata = (
            workspace_root / "reports" / "quarterly" / "summary.txt"
        ).stat()
        archive_metadata = (workspace_root / "archive" / "old.pdf").stat()
        source_entry = FileEntry(
            workspace_id=workspace.id,
            relative_path="inbox/quarterly-report.txt",
            name="quarterly-report.txt",
            extension=".txt",
            size_bytes=source_metadata.st_size,
            mtime_ns=source_metadata.st_mtime_ns,
        )
        session.add_all(
            [
                source_entry,
                FileEntry(
                    workspace_id=workspace.id,
                    relative_path="reports/quarterly/summary.txt",
                    name="summary.txt",
                    extension=".txt",
                    size_bytes=candidate_metadata.st_size,
                    mtime_ns=candidate_metadata.st_mtime_ns,
                ),
                FileEntry(
                    workspace_id=workspace.id,
                    relative_path="archive/old.pdf",
                    name="old.pdf",
                    extension=".pdf",
                    size_bytes=archive_metadata.st_size,
                    mtime_ns=archive_metadata.st_mtime_ns,
                ),
            ]
        )
        session.commit()
        result = build_find_similar_folders_tool(session).invoke(
            {
                "workspace_id": workspace.id,
                "source_file_id": source_entry.id,
            }
        )

    assert result.ok is True
    assert isinstance(result.data, dict)
    assert result.data["empty_reason"] is None
    candidate = result.data["items"][0]
    assert candidate["relative_directory"] == "reports/quarterly"
    assert candidate["score"] == 65
    assert candidate["reasons"] == [
        "extension_match",
        "directory_name_token_overlap",
    ]
    assert candidate["file_ids"] == [2]
    assert source_path.read_text(encoding="utf-8") == "keep unchanged"
    assert str(workspace_root) not in result.model_dump_json()


def test_find_similar_folders_reports_insufficient_samples_and_scope_errors(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "similar-folders-empty-workspace"
    workspace_root.mkdir()
    source_path = workspace_root / "only.txt"
    source_path.write_text("only", encoding="utf-8")

    with Session(engine) as session:
        workspace = Workspace(
            name="无候选目录工作区",
            root_path=str(workspace_root),
        )
        session.add(workspace)
        session.flush()
        metadata = source_path.stat()
        source_entry = FileEntry(
            workspace_id=workspace.id,
            relative_path="only.txt",
            name="only.txt",
            extension=".txt",
            size_bytes=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
        )
        session.add(source_entry)
        session.commit()
        tool = build_find_similar_folders_tool(session)
        empty = tool.invoke(
            {
                "workspace_id": workspace.id,
                "source_file_id": source_entry.id,
            }
        )
        missing_workspace = tool.invoke(
            {"workspace_id": 999, "source_file_id": source_entry.id}
        )
        invalid = tool.invoke(
            {
                "workspace_id": workspace.id,
                "source_file_id": source_entry.id,
                "unexpected": True,
            }
        )

    assert empty.ok is True
    assert empty.data == {
        "workspace_id": workspace.id,
        "source_file_id": source_entry.id,
        "items": [],
        "empty_reason": "insufficient_directory_samples",
    }
    assert missing_workspace.error is not None
    assert missing_workspace.error.code == "workspace_not_found"
    assert invalid.error is not None
    assert invalid.error.code == "invalid_arguments"


def test_registry_dispatches_traceable_knowledge_search(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        workspace_id = _seed_knowledge_index(session)
        registry = build_read_tool_registry(session)

        result = registry.invoke(
            "knowledge_search",
            {
                "workspace_id": workspace_id,
                "query": "审批",
            },
        )

    assert result.ok is True
    assert isinstance(result.data, dict)
    assert result.data["total"] == 1
    assert result.data["has_more"] is False
    item = result.data["items"][0]
    assert item["source_relative_path"] == "guides/approval.md"
    assert item["text"] == "第一行：审批流程\n第二行：批准后才能移动文件。"
    assert item["start_offset"] == 0
    assert item["end_offset"] == len(item["text"])
    assert (item["start_line"], item["end_line"]) == (1, 2)
    assert item["score"] == 1
    assert "private/other.md" not in result.model_dump_json()
    assert "D:/Private" not in result.model_dump_json()


def test_registry_returns_pdf_page_provenance_for_knowledge_search(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        workspace = Workspace(
            name="PDF 知识搜索工作区",
            root_path="D:/Private/PdfKnowledgeSearch",
        )
        session.add(workspace)
        session.flush()

        relative_path = "reports/approval.pdf"
        text = "第一页审批说明\n\n第二页审批说明"
        first_page_end = len("第一页审批说明")
        file_entry = FileEntry(
            workspace_id=workspace.id,
            relative_path=relative_path,
            name=Path(relative_path).name,
            extension=".pdf",
            size_bytes=len(text.encode("utf-8")),
            mtime_ns=1_800_000_000_000_000_000,
        )
        session.add(file_entry)
        session.flush()

        document = Document(
            document_id=uuid4(),
            workspace_id=workspace.id,
            file_entry_id=file_entry.id,
            source_relative_path=relative_path,
            source_format="pdf",
            normalized_text=text,
            pages=(
                DocumentPage(
                    page_number=1,
                    start_offset=0,
                    end_offset=first_page_end,
                ),
                DocumentPage(
                    page_number=2,
                    start_offset=first_page_end + 2,
                    end_offset=len(text),
                ),
            ),
            source_version="a" * 64,
            source_updated_at="2026-09-01T00:00:00+00:00",
        )
        session.add(DocumentRecord.from_contract(document))
        session.add_all(
            ChunkRecord.from_contract(chunk)
            for chunk in chunk_document(document)
        )
        session.commit()

        result = build_read_tool_registry(session).invoke(
            "knowledge_search",
            {"workspace_id": workspace.id, "query": "审批"},
        )

    assert result.ok is True
    assert result.data is not None
    assert result.data["items"][0]["source_relative_path"] == relative_path
    assert (result.data["items"][0]["page_start"], result.data["items"][0]["page_end"]) == (
        1,
        2,
    )


def test_registry_rejects_invalid_knowledge_search_arguments() -> None:
    with Session() as session:
        registry = build_read_tool_registry(session)

        result = registry.invoke(
            "knowledge_search",
            {
                "workspace_id": 1,
                "query": "   ",
                "unexpected": True,
            },
        )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "invalid_arguments"


def test_registry_validates_registered_call_without_running_handler() -> None:
    handler_calls: list[bool] = []
    registry = ToolRegistry([_fake_tool("safe_tool", handler_calls)])

    result = registry.validate("safe_tool", {})

    assert result == ToolResult.success()
    assert handler_calls == []


def test_registry_rejects_unregistered_write_tool_during_validation() -> None:
    handler_calls: list[bool] = []
    registry = ToolRegistry([_fake_tool("safe_tool", handler_calls)])

    result = registry.validate("delete_file", {"path": "private.txt"})

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "unknown_tool"
    assert handler_calls == []


def test_registry_rejects_invalid_arguments_without_running_handler() -> None:
    handler_calls: list[bool] = []
    registry = ToolRegistry([_fake_tool("safe_tool", handler_calls)])

    result = registry.validate("safe_tool", {"unexpected": True})

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "invalid_arguments"
    assert handler_calls == []


@pytest.mark.parametrize("unknown_name", ["delete_all_files", "", 123, None])
def test_registry_rejects_unknown_tool_without_calling_handler(
    unknown_name: object,
) -> None:
    handler_calls: list[bool] = []
    registry = ToolRegistry([_fake_tool("safe_tool", handler_calls)])

    result = registry.invoke(unknown_name, {})

    assert result.model_dump() == {
        "ok": False,
        "data": None,
        "error": {
            "code": "unknown_tool",
            "message": "请求的工具未注册",
            "details": {},
        },
    }
    assert handler_calls == []


def test_registry_rejects_duplicate_tool_names() -> None:
    with pytest.raises(ValueError, match="duplicate tool name: same_tool"):
        ToolRegistry([_fake_tool("same_tool"), _fake_tool("same_tool")])


def test_registry_keeps_registered_tool_argument_validation() -> None:
    registry = ToolRegistry([_fake_tool("safe_tool")])

    result = registry.invoke("safe_tool", {"unknown": True})

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "invalid_arguments"
