from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import Workspace
from backend.app.repositories import find_file_entries
from backend.app.services import (
    DuplicateScannedPathError,
    FileIndexSyncResult,
    WorkspaceNotFoundError,
    sync_file_index,
)
from backend.app.workspace_scanner import ScannedFile


def _scanned_file(
    relative_path: str,
    size_bytes: int = 10,
    mtime_ns: int = 100,
) -> ScannedFile:
    path = Path(relative_path)
    return ScannedFile(
        relative_path=relative_path,
        name=path.name,
        extension=path.suffix.casefold(),
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
    )


def test_sync_file_index_is_idempotent_and_tracks_changes(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'sync.db').as_posix()}")
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            first_workspace = Workspace(
                name="同步工作区",
                root_path="D:/Test/SyncFirst",
            )
            second_workspace = Workspace(
                name="隔离工作区",
                root_path="D:/Test/SyncSecond",
            )
            session.add_all([first_workspace, second_workspace])
            session.commit()

            initial_scan = [
                _scanned_file("a.txt"),
                _scanned_file("folder/b.md", size_bytes=20, mtime_ns=200),
            ]
            first_result = sync_file_index(
                session,
                first_workspace.id,
                initial_scan,
            )

            assert first_result == FileIndexSyncResult(
                created=2,
                updated=0,
                deleted=0,
                unchanged=0,
            )

            first_entries = find_file_entries(session, first_workspace.id)
            original_a_id = next(
                entry.id for entry in first_entries if entry.relative_path == "a.txt"
            )

            repeated_result = sync_file_index(
                session,
                first_workspace.id,
                initial_scan,
            )

            assert repeated_result == FileIndexSyncResult(
                created=0,
                updated=0,
                deleted=0,
                unchanged=2,
            )

            sync_file_index(
                session,
                second_workspace.id,
                [_scanned_file("other.txt")],
            )
            changed_result = sync_file_index(
                session,
                first_workspace.id,
                [
                    _scanned_file("a.txt", size_bytes=99, mtime_ns=999),
                    _scanned_file("new.pdf", size_bytes=30, mtime_ns=300),
                ],
            )

            assert changed_result == FileIndexSyncResult(
                created=1,
                updated=1,
                deleted=1,
                unchanged=0,
            )

            changed_entries = find_file_entries(session, first_workspace.id)
            assert [entry.relative_path for entry in changed_entries] == [
                "a.txt",
                "new.pdf",
            ]

            changed_a = next(
                entry for entry in changed_entries if entry.relative_path == "a.txt"
            )
            assert changed_a.id == original_a_id
            assert changed_a.size_bytes == 99
            assert changed_a.mtime_ns == 999
            assert [
                entry.relative_path
                for entry in find_file_entries(session, second_workspace.id)
            ] == ["other.txt"]
    finally:
        engine.dispose()


def test_sync_rejects_duplicate_paths_without_changing_index(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'duplicate.db').as_posix()}")
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            workspace = Workspace(
                name="重复扫描测试",
                root_path="D:/Test/SyncDuplicate",
            )
            session.add(workspace)
            session.commit()

            sync_file_index(
                session,
                workspace.id,
                [_scanned_file("stable.txt")],
            )

            with pytest.raises(DuplicateScannedPathError):
                sync_file_index(
                    session,
                    workspace.id,
                    [
                        _scanned_file("duplicate.txt", size_bytes=10),
                        _scanned_file("duplicate.txt", size_bytes=20),
                    ],
                )

            assert [
                entry.relative_path
                for entry in find_file_entries(session, workspace.id)
            ] == ["stable.txt"]
    finally:
        engine.dispose()


def test_sync_rejects_missing_workspace(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'missing.db').as_posix()}")
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            with pytest.raises(WorkspaceNotFoundError):
                sync_file_index(
                    session,
                    999,
                    [_scanned_file("orphan.txt")],
                )

            assert find_file_entries(session, 999) == []
    finally:
        engine.dispose()


def test_sync_rolls_back_when_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'rollback.db').as_posix()}")
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            workspace = Workspace(
                name="回滚测试",
                root_path="D:/Test/SyncRollback",
            )
            session.add(workspace)
            session.commit()
            workspace_id = workspace.id

            def fail_commit() -> None:
                raise RuntimeError("模拟提交失败")

            monkeypatch.setattr(session, "commit", fail_commit)

            with pytest.raises(RuntimeError, match="模拟提交失败"):
                sync_file_index(
                    session,
                    workspace_id,
                    [_scanned_file("not-persisted.txt")],
                )

            assert find_file_entries(session, workspace_id) == []
    finally:
        engine.dispose()
