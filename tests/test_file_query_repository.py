from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import FileEntry, Workspace
from backend.app.repositories import count_file_entries, find_file_entries


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'file-query.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _seed_file_entries(session: Session) -> tuple[int, int]:
    first_workspace = Workspace(
        name="查询工作区",
        root_path="D:/Test/FileQuery",
    )
    second_workspace = Workspace(
        name="其他工作区",
        root_path="D:/Test/OtherFileQuery",
    )
    session.add_all([first_workspace, second_workspace])
    session.flush()

    session.add_all(
        [
            FileEntry(
                workspace_id=first_workspace.id,
                relative_path="a-first.txt",
                name="same.txt",
                extension=".txt",
                size_bytes=20,
                mtime_ns=300,
            ),
            FileEntry(
                workspace_id=first_workspace.id,
                relative_path="b-second.txt",
                name="same.txt",
                extension=".txt",
                size_bytes=10,
                mtime_ns=200,
            ),
            FileEntry(
                workspace_id=first_workspace.id,
                relative_path="c-third.pdf",
                name="other.pdf",
                extension=".pdf",
                size_bytes=10,
                mtime_ns=100,
            ),
            FileEntry(
                workspace_id=second_workspace.id,
                relative_path="000-must-not-leak.txt",
                name="leak.txt",
                extension=".txt",
                size_bytes=1,
                mtime_ns=1,
            ),
        ]
    )
    session.commit()

    return first_workspace.id, second_workspace.id


@pytest.mark.parametrize(
    ("sort_by", "sort_order", "expected_paths"),
    [
        (
            "name",
            "asc",
            ["c-third.pdf", "a-first.txt", "b-second.txt"],
        ),
        (
            "size_bytes",
            "asc",
            ["b-second.txt", "c-third.pdf", "a-first.txt"],
        ),
        (
            "mtime_ns",
            "desc",
            ["a-first.txt", "b-second.txt", "c-third.pdf"],
        ),
        (
            "relative_path",
            "desc",
            ["c-third.pdf", "b-second.txt", "a-first.txt"],
        ),
    ],
)
def test_repository_applies_stable_file_sorting(
    engine: Engine,
    sort_by: str,
    sort_order: str,
    expected_paths: list[str],
) -> None:
    with Session(engine) as session:
        workspace_id, _ = _seed_file_entries(session)

        entries = find_file_entries(
            session,
            workspace_id,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        assert [entry.relative_path for entry in entries] == expected_paths


def test_repository_paginates_only_requested_workspace(engine: Engine) -> None:
    with Session(engine) as session:
        workspace_id, other_workspace_id = _seed_file_entries(session)

        entries = find_file_entries(
            session,
            workspace_id,
            offset=1,
            limit=1,
        )

        assert [entry.relative_path for entry in entries] == ["b-second.txt"]
        assert count_file_entries(session, workspace_id) == 3
        assert count_file_entries(session, other_workspace_id) == 1


@pytest.mark.parametrize(
    "query_options",
    [
        {"sort_by": "unknown"},
        {"sort_order": "newest"},
        {"offset": -1},
        {"limit": 0},
    ],
)
def test_repository_rejects_invalid_query_options(
    engine: Engine,
    query_options: dict[str, object],
) -> None:
    with Session(engine) as session:
        workspace_id, _ = _seed_file_entries(session)

        with pytest.raises(ValueError):
            find_file_entries(session, workspace_id, **query_options)
