from hashlib import sha256
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import FileEntry, Workspace
from backend.app.operation_preview import OperationPreviewRequest
from backend.app.services import generate_operation_preview


DiskEntry = tuple[str, int, int, int, int, str | None]


def _snapshot_workspace(workspace_root: Path) -> dict[str, DiskEntry]:
    """记录足以暴露创建、删除、移动、改写和元数据变化的磁盘状态。"""

    paths = [
        workspace_root,
        *sorted(workspace_root.rglob("*"), key=lambda path: path.as_posix()),
    ]
    snapshot: dict[str, DiskEntry] = {}
    for path in paths:
        relative_path = "."
        if path != workspace_root:
            relative_path = path.relative_to(workspace_root).as_posix()
        if path.is_file():
            kind = "file"
            content_digest = sha256(path.read_bytes()).hexdigest()
        elif path.is_dir():
            kind = "directory"
            content_digest = None
        else:
            kind = "other"
            content_digest = None

        metadata = path.stat()
        snapshot[relative_path] = (
            kind,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
            content_digest,
        )
    return snapshot


def test_operation_preview_leaves_complete_workspace_snapshot_unchanged(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    source_paths = [
        workspace_root / "inbox" / "Python_Project_Code.zip",
        workspace_root / "inbox" / "Tokyo_Travel_Plan.xlsx",
    ]
    source_paths[0].parent.mkdir(parents=True)
    source_paths[0].write_bytes(b"python project")
    source_paths[1].write_bytes(b"tokyo travel")
    (workspace_root / "programming" / "Python").mkdir(parents=True)
    (workspace_root / "travel" / "Tokyo Trip").mkdir(parents=True)
    unrelated_path = workspace_root / "unrelated" / "keep.txt"
    unrelated_path.parent.mkdir()
    unrelated_path.write_text("must stay unchanged", encoding="utf-8")

    engine = create_engine(
        f"sqlite:///{(tmp_path / 'operation-preview-readonly.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            workspace = Workspace(
                name="磁盘不变预览工作区",
                root_path=str(workspace_root),
            )
            session.add(workspace)
            session.flush()

            entries: list[FileEntry] = []
            for source_path in source_paths:
                metadata = source_path.stat()
                entry = FileEntry(
                    workspace_id=workspace.id,
                    relative_path=source_path.relative_to(
                        workspace_root
                    ).as_posix(),
                    name=source_path.name,
                    extension=source_path.suffix.casefold(),
                    size_bytes=metadata.st_size,
                    mtime_ns=metadata.st_mtime_ns,
                )
                entries.append(entry)
                session.add(entry)
            session.commit()

            request = OperationPreviewRequest(
                workspace_id=workspace.id,
                source_file_ids=[entry.id for entry in entries],
                target_directories=[
                    "programming/Python",
                    "travel/Tokyo Trip",
                ],
            )
            before = _snapshot_workspace(workspace_root)

            response = generate_operation_preview(session, request)

            after = _snapshot_workspace(workspace_root)

        assert len(response.items) == 2
        assert response.read_only is True
        assert after == before
    finally:
        engine.dispose()
