from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import FileEntry, Workspace
from backend.app.operation_preview import OperationPreviewRequest
from backend.app.path_policy import PathPolicyError
from backend.app.services import (
    FileEntryNotFoundError,
    OperationPreviewPathUnavailableError,
    WorkspaceNotFoundError,
    generate_operation_preview,
)


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'operation-preview.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _seed_preview_workspace(
    session: Session,
    workspace_root: Path,
) -> tuple[int, int, int]:
    source_path = workspace_root / "inbox" / "Python_Project_Code.zip"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"preview source")
    (workspace_root / "programming" / "Python").mkdir(parents=True)
    (workspace_root / "programming" / "Java").mkdir(parents=True)

    workspace = Workspace(
        name="整理预览工作区",
        root_path=str(workspace_root),
    )
    other_workspace = Workspace(
        name="其他工作区",
        root_path=str(workspace_root.parent / "other-workspace"),
    )
    session.add_all([workspace, other_workspace])
    session.flush()

    source_stat = source_path.stat()
    source_entry = FileEntry(
        workspace_id=workspace.id,
        relative_path="inbox/Python_Project_Code.zip",
        name="Python_Project_Code.zip",
        extension=".zip",
        size_bytes=source_stat.st_size,
        mtime_ns=source_stat.st_mtime_ns,
    )
    other_entry = FileEntry(
        workspace_id=other_workspace.id,
        relative_path="private.txt",
        name="private.txt",
        extension=".txt",
        size_bytes=1,
        mtime_ns=1,
    )
    session.add_all([source_entry, other_entry])
    session.commit()

    return workspace.id, source_entry.id, other_entry.id


def test_generate_operation_preview_returns_read_only_ranked_candidates(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    with Session(engine) as session:
        workspace_id, source_file_id, _ = _seed_preview_workspace(
            session,
            workspace_root,
        )
        request = OperationPreviewRequest(
            workspace_id=workspace_id,
            source_file_ids=[source_file_id],
            target_directories=["programming/Python", "programming/Java"],
        )

        response = generate_operation_preview(session, request)

        assert response.workspace_id == workspace_id
        assert response.read_only is True
        assert response.items[0].source_relative_path == (
            "inbox/Python_Project_Code.zip"
        )
        assert response.items[0].candidates[0].relative_directory == (
            "programming/Python"
        )
        assert not session.new
        assert not session.dirty
        assert not session.deleted


def test_generate_operation_preview_rejects_missing_workspace(
    engine: Engine,
) -> None:
    request = OperationPreviewRequest(
        workspace_id=404,
        source_file_ids=[1],
        target_directories=["documents"],
    )

    with Session(engine) as session, pytest.raises(WorkspaceNotFoundError):
        generate_operation_preview(session, request)


def test_generate_operation_preview_hides_cross_workspace_file(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    with Session(engine) as session:
        workspace_id, _, other_file_id = _seed_preview_workspace(
            session,
            workspace_root,
        )
        request = OperationPreviewRequest(
            workspace_id=workspace_id,
            source_file_ids=[other_file_id],
            target_directories=["programming/Python"],
        )

        with pytest.raises(FileEntryNotFoundError):
            generate_operation_preview(session, request)


@pytest.mark.parametrize("missing_kind", ["source", "target"])
def test_generate_operation_preview_rejects_missing_disk_paths(
    engine: Engine,
    tmp_path: Path,
    missing_kind: str,
) -> None:
    workspace_root = tmp_path / "workspace"
    with Session(engine) as session:
        workspace_id, source_file_id, _ = _seed_preview_workspace(
            session,
            workspace_root,
        )
        if missing_kind == "source":
            (workspace_root / "inbox" / "Python_Project_Code.zip").unlink()
            target_directories = ["programming/Python"]
        else:
            target_directories = ["programming/Missing"]

        request = OperationPreviewRequest(
            workspace_id=workspace_id,
            source_file_ids=[source_file_id],
            target_directories=target_directories,
        )

        with pytest.raises(OperationPreviewPathUnavailableError):
            generate_operation_preview(session, request)


def test_generate_operation_preview_preserves_path_policy_errors(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    with Session(engine) as session:
        workspace_id, source_file_id, _ = _seed_preview_workspace(
            session,
            workspace_root,
        )
        request = OperationPreviewRequest(
            workspace_id=workspace_id,
            source_file_ids=[source_file_id],
            target_directories=[".git"],
        )

        with pytest.raises(PathPolicyError) as error:
            generate_operation_preview(session, request)

    assert error.value.code.value == "sensitive_path"
