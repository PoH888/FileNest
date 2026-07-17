from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import FileEntry, Workspace


def test_file_entry_persists_file_metadata(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'file-entry.db').as_posix()}")
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            workspace = Workspace(
                name="索引测试",
                root_path="D:/Test/Index",
            )
            session.add(workspace)
            session.flush()

            file_entry = FileEntry(
                workspace_id=workspace.id,
                relative_path="documents/report.txt",
                name="report.txt",
                extension=".txt",
                size_bytes=1024,
                mtime_ns=1_800_000_000_000_000_000,
            )
            session.add(file_entry)
            session.commit()

            saved_entry = session.get(FileEntry, file_entry.id)

            assert saved_entry is not None
            assert saved_entry.workspace_id == workspace.id
            assert saved_entry.relative_path == "documents/report.txt"
            assert saved_entry.name == "report.txt"
            assert saved_entry.extension == ".txt"
            assert saved_entry.size_bytes == 1024
            assert saved_entry.mtime_ns == 1_800_000_000_000_000_000
    finally:
        engine.dispose()


def test_relative_path_is_unique_only_within_its_workspace(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'unique-path.db').as_posix()}")
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            first_workspace = Workspace(
                name="第一个工作区",
                root_path="D:/Test/First",
            )
            second_workspace = Workspace(
                name="第二个工作区",
                root_path="D:/Test/Second",
            )
            session.add_all([first_workspace, second_workspace])
            session.flush()

            session.add_all(
                [
                    FileEntry(
                        workspace_id=first_workspace.id,
                        relative_path="shared/name.txt",
                        name="name.txt",
                        extension=".txt",
                        size_bytes=10,
                        mtime_ns=100,
                    ),
                    FileEntry(
                        workspace_id=second_workspace.id,
                        relative_path="shared/name.txt",
                        name="name.txt",
                        extension=".txt",
                        size_bytes=20,
                        mtime_ns=200,
                    ),
                ]
            )
            session.commit()

            session.add(
                FileEntry(
                    workspace_id=first_workspace.id,
                    relative_path="shared/name.txt",
                    name="name.txt",
                    extension=".txt",
                    size_bytes=30,
                    mtime_ns=300,
                )
            )

            with pytest.raises(IntegrityError):
                session.commit()

            session.rollback()
    finally:
        engine.dispose()
