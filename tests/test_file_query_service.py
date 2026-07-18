from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import FileEntry, Workspace
from backend.app.services import (
    FileEntryNotFoundError,
    WorkspaceNotFoundError,
    get_file_detail,
    search_files,
)


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'file-search.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _epoch_ns(year: int, month: int, day: int) -> int:
    value = datetime(year, month, day, tzinfo=timezone.utc)
    return int(value.timestamp()) * 1_000_000_000


def _seed_file_entries(session: Session) -> int:
    workspace = Workspace(
        name="文件搜索工作区",
        root_path="D:/Test/FileSearch",
    )
    other_workspace = Workspace(
        name="其他搜索工作区",
        root_path="D:/Test/OtherFileSearch",
    )
    session.add_all([workspace, other_workspace])
    session.flush()

    session.add_all(
        [
            FileEntry(
                workspace_id=workspace.id,
                relative_path="Reports/a-old.pdf",
                name="a-old.pdf",
                extension=".pdf",
                size_bytes=10,
                mtime_ns=_epoch_ns(2026, 8, 1),
            ),
            FileEntry(
                workspace_id=workspace.id,
                relative_path="Reports/b-middle.pdf",
                name="b-middle.pdf",
                extension=".pdf",
                size_bytes=20,
                mtime_ns=_epoch_ns(2026, 8, 15),
            ),
            FileEntry(
                workspace_id=workspace.id,
                relative_path="Reports/c-new.pdf",
                name="c-new.pdf",
                extension=".pdf",
                size_bytes=30,
                mtime_ns=_epoch_ns(2026, 8, 20),
            ),
            FileEntry(
                workspace_id=workspace.id,
                relative_path="notes/Budget_%_Plan.txt",
                name="Budget_%_Plan.txt",
                extension=".txt",
                size_bytes=40,
                mtime_ns=_epoch_ns(2026, 8, 15),
            ),
            FileEntry(
                workspace_id=other_workspace.id,
                relative_path="Reports/must-not-leak.pdf",
                name="must-not-leak.pdf",
                extension=".pdf",
                size_bytes=1,
                mtime_ns=_epoch_ns(2026, 8, 20),
            ),
        ]
    )
    session.commit()

    return workspace.id


def _seed_file_detail(session: Session) -> tuple[int, int, int]:
    first_workspace = Workspace(
        name="详情工作区",
        root_path="D:/Test/FileDetail",
    )
    second_workspace = Workspace(
        name="其他详情工作区",
        root_path="D:/Test/OtherFileDetail",
    )
    session.add_all([first_workspace, second_workspace])
    session.flush()

    file_entry = FileEntry(
        workspace_id=first_workspace.id,
        relative_path="documents/detail.pdf",
        name="detail.pdf",
        extension=".pdf",
        size_bytes=512,
        mtime_ns=_epoch_ns(2026, 8, 30),
    )
    session.add(file_entry)
    session.commit()

    return first_workspace.id, second_workspace.id, file_entry.id


def test_search_files_matches_name_or_path_without_case_sensitivity(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        workspace_id = _seed_file_entries(session)

        result = search_files(
            session,
            workspace_id,
            keyword="reports",
        )

        assert result.total == 3
        assert [entry.relative_path for entry in result.items] == [
            "Reports/a-old.pdf",
            "Reports/b-middle.pdf",
            "Reports/c-new.pdf",
        ]


@pytest.mark.parametrize("keyword", ["%", "_"])
def test_search_files_treats_sql_wildcards_as_plain_text(
    engine: Engine,
    keyword: str,
) -> None:
    with Session(engine) as session:
        workspace_id = _seed_file_entries(session)

        result = search_files(
            session,
            workspace_id,
            keyword=keyword,
        )

        assert result.total == 1
        assert [entry.name for entry in result.items] == ["Budget_%_Plan.txt"]


def test_search_files_filters_before_counting_sorting_and_pagination(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        workspace_id = _seed_file_entries(session)
        query_options = {
            "extension": "PDF",
            "modified_from": datetime(2026, 8, 15, tzinfo=timezone.utc),
            "modified_to": datetime(2026, 8, 20, tzinfo=timezone.utc),
            "sort_by": "modified_at",
            "sort_order": "desc",
            "page_size": 1,
        }

        first_page = search_files(
            session,
            workspace_id,
            page=1,
            **query_options,
        )
        second_page = search_files(
            session,
            workspace_id,
            page=2,
            **query_options,
        )

        assert first_page.total == 2
        assert first_page.page == 1
        assert [entry.name for entry in first_page.items] == ["c-new.pdf"]
        assert second_page.total == 2
        assert second_page.page == 2
        assert [entry.name for entry in second_page.items] == ["b-middle.pdf"]


def test_search_files_rejects_missing_workspace(engine: Engine) -> None:
    with Session(engine) as session:
        with pytest.raises(WorkspaceNotFoundError):
            search_files(session, 999, keyword="report")


def test_get_file_detail_returns_index_from_requested_workspace(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        workspace_id, _, file_id = _seed_file_detail(session)

        file_entry = get_file_detail(session, workspace_id, file_id)

        assert file_entry.relative_path == "documents/detail.pdf"
        assert file_entry.workspace_id == workspace_id


def test_get_file_detail_rejects_missing_file(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        workspace_id, _, _ = _seed_file_detail(session)

        with pytest.raises(FileEntryNotFoundError):
            get_file_detail(session, workspace_id, 999)


def test_get_file_detail_does_not_cross_workspace_boundary(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        _, other_workspace_id, file_id = _seed_file_detail(session)

        with pytest.raises(FileEntryNotFoundError):
            get_file_detail(session, other_workspace_id, file_id)


def test_get_file_detail_rejects_missing_workspace(engine: Engine) -> None:
    with Session(engine) as session:
        with pytest.raises(WorkspaceNotFoundError):
            get_file_detail(session, 999, 1)
