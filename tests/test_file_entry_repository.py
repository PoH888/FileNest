from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import FileEntry, Workspace
from backend.app.repositories import (
    add_file_entry,
    delete_file_entry,
    find_file_entries,
    get_file_entry_by_path,
)


def _file_entry(
    workspace_id: int,
    relative_path: str,
    size_bytes: int = 10,
) -> FileEntry:
    path = Path(relative_path)
    return FileEntry(
        workspace_id=workspace_id,
        relative_path=relative_path,
        name=path.name,
        extension=path.suffix,
        size_bytes=size_bytes,
        mtime_ns=100,
    )


def test_repository_reads_only_requested_workspace(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'repository.db').as_posix()}")
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            first_workspace = Workspace(
                name="第一个工作区",
                root_path="D:/Test/RepositoryFirst",
            )
            second_workspace = Workspace(
                name="第二个工作区",
                root_path="D:/Test/RepositorySecond",
            )
            session.add_all([first_workspace, second_workspace])
            session.flush()

            add_file_entry(
                session,
                _file_entry(first_workspace.id, "z-last.txt"),
            )
            add_file_entry(
                session,
                _file_entry(first_workspace.id, "a-first.txt"),
            )
            add_file_entry(
                session,
                _file_entry(second_workspace.id, "other.txt"),
            )
            session.commit()

            entries = find_file_entries(session, first_workspace.id)

            assert [entry.relative_path for entry in entries] == [
                "a-first.txt",
                "z-last.txt",
            ]
            assert get_file_entry_by_path(
                session,
                first_workspace.id,
                "a-first.txt",
            ) is not None
            assert get_file_entry_by_path(
                session,
                first_workspace.id,
                "other.txt",
            ) is None
    finally:
        engine.dispose()


def test_repository_leaves_transactions_to_caller(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'transactions.db').as_posix()}")
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            workspace = Workspace(
                name="事务测试工作区",
                root_path="D:/Test/RepositoryTransactions",
            )
            session.add(workspace)
            session.commit()

            rolled_back_entry = _file_entry(workspace.id, "rolled-back.txt")
            add_file_entry(session, rolled_back_entry)
            assert rolled_back_entry in session.new

            session.rollback()

            assert get_file_entry_by_path(
                session,
                workspace.id,
                "rolled-back.txt",
            ) is None

            persisted_entry = _file_entry(workspace.id, "persisted.txt")
            add_file_entry(session, persisted_entry)
            session.commit()

            delete_file_entry(session, persisted_entry)
            assert persisted_entry in session.deleted

            session.rollback()

            restored_entry = get_file_entry_by_path(
                session,
                workspace.id,
                "persisted.txt",
            )
            assert restored_entry is not None

            delete_file_entry(session, restored_entry)
            session.commit()

            assert get_file_entry_by_path(
                session,
                workspace.id,
                "persisted.txt",
            ) is None
    finally:
        engine.dispose()
