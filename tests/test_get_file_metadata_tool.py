from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from backend.app import read_tools
from backend.app.database import Base
from backend.app.models import FileEntry, Workspace
from backend.app.read_tools import build_get_file_metadata_tool


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'file-metadata-tool.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _add_workspace(session: Session, root: Path, name: str) -> Workspace:
    root.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(name=name, root_path=str(root))
    session.add(workspace)
    session.flush()
    return workspace


def _add_file_entry(
    session: Session,
    workspace_id: int,
    relative_path: str,
) -> FileEntry:
    file_entry = FileEntry(
        workspace_id=workspace_id,
        relative_path=relative_path,
        name=Path(relative_path).name,
        extension=Path(relative_path).suffix,
        size_bytes=2048,
        mtime_ns=1_700_000_000_000_000_000,
    )
    session.add(file_entry)
    session.commit()
    return file_entry


def test_get_file_metadata_tool_returns_authorized_index_metadata(
    engine: Engine,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    with Session(engine) as session:
        workspace = _add_workspace(session, workspace_root, "元数据工作区")
        file_entry = _add_file_entry(
            session,
            workspace.id,
            "documents/report.txt",
        )
        tool = build_get_file_metadata_tool(session)

        result = tool.invoke(
            {"workspace_id": workspace.id, "file_id": file_entry.id}
        )

    assert result.model_dump() == {
        "ok": True,
        "data": {
            "file_id": file_entry.id,
            "relative_path": "documents/report.txt",
            "name": "report.txt",
            "extension": ".txt",
            "size_bytes": 2048,
            "modified_at": "2023-11-14T22:13:20Z",
            "workspace_id": workspace.id,
        },
        "error": None,
    }
    assert str(workspace_root) not in result.model_dump_json()
    assert "root_path" not in result.model_dump_json()


def test_get_file_metadata_tool_rejects_file_from_another_workspace(
    engine: Engine,
    tmp_path: Path,
) -> None:
    with Session(engine) as session:
        first = _add_workspace(session, tmp_path / "first", "第一个工作区")
        second = _add_workspace(session, tmp_path / "second", "第二个工作区")
        file_entry = _add_file_entry(session, first.id, "report.txt")
        tool = build_get_file_metadata_tool(session)

        result = tool.invoke(
            {"workspace_id": second.id, "file_id": file_entry.id}
        )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "file_not_found"


@pytest.mark.parametrize(
    ("relative_path", "expected_code"),
    [
        ("../outside.txt", "path_outside_workspace"),
        (".env", "sensitive_path"),
    ],
)
def test_get_file_metadata_tool_rejects_unauthorized_index_path(
    engine: Engine,
    tmp_path: Path,
    relative_path: str,
    expected_code: str,
) -> None:
    with Session(engine) as session:
        workspace = _add_workspace(
            session,
            tmp_path / expected_code,
            expected_code,
        )
        file_entry = _add_file_entry(
            session,
            workspace.id,
            relative_path,
        )
        tool = build_get_file_metadata_tool(session)

        result = tool.invoke(
            {"workspace_id": workspace.id, "file_id": file_entry.id}
        )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == expected_code
    assert relative_path not in result.model_dump_json()


def test_get_file_metadata_tool_returns_structured_missing_errors(
    engine: Engine,
    tmp_path: Path,
) -> None:
    with Session(engine) as session:
        tool = build_get_file_metadata_tool(session)
        missing_workspace = tool.invoke({"workspace_id": 999, "file_id": 1})

        workspace = _add_workspace(session, tmp_path / "empty", "空工作区")
        session.commit()
        missing_file = tool.invoke(
            {"workspace_id": workspace.id, "file_id": 999}
        )

    assert missing_workspace.error is not None
    assert missing_workspace.error.code == "workspace_not_found"
    assert missing_file.error is not None
    assert missing_file.error.code == "file_not_found"


@pytest.mark.parametrize(
    "arguments",
    [
        {"workspace_id": 0, "file_id": 1},
        {"workspace_id": 1, "file_id": 0},
        {"workspace_id": 1, "file_id": 1, "unknown": True},
    ],
)
def test_get_file_metadata_tool_rejects_invalid_arguments_before_service(
    arguments: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_called = False

    def fake_get_file_metadata(*args: object, **kwargs: object) -> FileEntry:
        del args, kwargs
        nonlocal service_called
        service_called = True
        raise AssertionError("invalid arguments reached the service")

    monkeypatch.setattr(
        read_tools,
        "get_file_metadata_service",
        fake_get_file_metadata,
    )

    with Session() as session:
        tool = build_get_file_metadata_tool(session)
        result = tool.invoke(arguments)

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "invalid_arguments"
    assert service_called is False
