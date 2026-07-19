from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from backend.app import read_tools
from backend.app.database import Base
from backend.app.models import FileEntry, Workspace
from backend.app.read_tools import build_search_files_tool
from backend.app.services import FileSearchResult


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'search-files-tool.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _seed_search_index(session: Session) -> int:
    workspace = Workspace(
        name="搜索工具工作区",
        root_path="D:/Private/SearchTool",
    )
    other_workspace = Workspace(
        name="其他工作区",
        root_path="D:/Private/OtherWorkspace",
    )
    session.add_all([workspace, other_workspace])
    session.flush()

    session.add_all(
        [
            FileEntry(
                workspace_id=workspace.id,
                relative_path=f"reports/report-{index:02d}.txt",
                name=f"report-{index:02d}.txt",
                extension=".txt",
                size_bytes=index,
                mtime_ns=1_700_000_000_000_000_000 + index,
            )
            for index in range(25)
        ]
    )
    session.add(
        FileEntry(
            workspace_id=other_workspace.id,
            relative_path="reports/report-must-not-leak.txt",
            name="report-must-not-leak.txt",
            extension=".txt",
            size_bytes=1,
            mtime_ns=1_700_000_000_000_000_000,
        )
    )
    session.commit()
    return workspace.id


def test_search_files_tool_caps_results_and_supports_paging(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        workspace_id = _seed_search_index(session)
        tool = build_search_files_tool(session)

        first_page = tool.invoke(
            {"workspace_id": workspace_id, "keyword": " report "}
        )
        second_page = tool.invoke(
            {
                "workspace_id": workspace_id,
                "keyword": "report",
                "page": 2,
            }
        )

    assert first_page.ok is True
    assert isinstance(first_page.data, dict)
    assert first_page.data["total"] == 25
    assert first_page.data["limit"] == 20
    assert first_page.data["has_more"] is True
    assert len(first_page.data["items"]) == 20

    assert second_page.ok is True
    assert isinstance(second_page.data, dict)
    assert second_page.data["total"] == 25
    assert second_page.data["has_more"] is False
    assert len(second_page.data["items"]) == 5
    assert "report-must-not-leak" not in second_page.model_dump_json()


def test_search_files_tool_returns_only_safe_index_metadata(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        workspace_id = _seed_search_index(session)
        tool = build_search_files_tool(session)

        result = tool.invoke(
            {
                "workspace_id": workspace_id,
                "keyword": "report-00",
                "extension": " TXT ",
            }
        )

    assert result.ok is True
    assert isinstance(result.data, dict)
    assert result.data["items"] == [
        {
            "file_id": 1,
            "relative_path": "reports/report-00.txt",
            "name": "report-00.txt",
            "extension": ".txt",
            "size_bytes": 0,
            "modified_at": "2023-11-14T22:13:20Z",
        }
    ]
    assert "root_path" not in result.model_dump_json()
    assert "D:/Private" not in result.model_dump_json()


@pytest.mark.parametrize(
    "arguments",
    [
        {"workspace_id": 0, "keyword": "report"},
        {"workspace_id": 1, "keyword": "   "},
        {"workspace_id": 1, "keyword": "report", "extension": " "},
        {"workspace_id": 1, "keyword": "report", "page": 0},
        {"workspace_id": 1, "keyword": "report", "limit": 0},
        {"workspace_id": 1, "keyword": "report", "limit": 21},
        {"workspace_id": 1, "keyword": "report", "unknown": True},
    ],
)
def test_search_files_tool_rejects_invalid_arguments_before_query(
    arguments: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_called = False

    def fake_search_files(*args: object, **kwargs: object) -> FileSearchResult:
        del args, kwargs
        nonlocal service_called
        service_called = True
        raise AssertionError("invalid arguments reached the service")

    monkeypatch.setattr(read_tools, "search_files_service", fake_search_files)

    with Session() as session:
        tool = build_search_files_tool(session)
        result = tool.invoke(arguments)

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "invalid_arguments"
    assert service_called is False


def test_search_files_tool_returns_structured_missing_workspace_error(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        tool = build_search_files_tool(session)

        result = tool.invoke({"workspace_id": 999, "keyword": "report"})

    assert result.model_dump() == {
        "ok": False,
        "data": None,
        "error": {
            "code": "workspace_not_found",
            "message": "工作区不存在",
            "details": {"workspace_id": 999},
        },
    }
