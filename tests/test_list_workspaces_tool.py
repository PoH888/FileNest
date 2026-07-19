from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from backend.app import read_tools
from backend.app.database import Base
from backend.app.models import Workspace
from backend.app.read_tools import build_list_workspaces_tool


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'list-workspaces-tool.db').as_posix()}"
    )
    Base.metadata.create_all(bind=test_engine)

    yield test_engine

    test_engine.dispose()


def _seed_workspaces(session: Session) -> None:
    session.add_all(
        [
            Workspace(name="文档", root_path="D:/Private/Documents"),
            Workspace(name="照片", root_path="D:/Private/Photos"),
        ]
    )
    session.commit()


def test_list_workspaces_tool_lists_safe_workspace_identifiers(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        _seed_workspaces(session)
        tool = build_list_workspaces_tool(session)

        result = tool.invoke({})

    assert result.model_dump() == {
        "ok": True,
        "data": {
            "items": [
                {"id": 1, "name": "文档"},
                {"id": 2, "name": "照片"},
            ],
            "count": 2,
        },
        "error": None,
    }
    assert "root_path" not in result.model_dump_json()
    assert "D:/Private" not in result.model_dump_json()


def test_list_workspaces_tool_normalizes_and_filters_name(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        _seed_workspaces(session)
        tool = build_list_workspaces_tool(session)

        result = tool.invoke({"name": "  照片  "})

    assert result.ok is True
    assert result.data == {
        "items": [{"id": 2, "name": "照片"}],
        "count": 1,
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {"name": "   "},
        {"name": 123},
        {"unknown": True},
    ],
)
def test_list_workspaces_tool_rejects_invalid_arguments_before_query(
    arguments: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_called = False

    def fake_list_workspaces(
        session: Session,
        name: str | None = None,
    ) -> list[Workspace]:
        del session, name
        nonlocal service_called
        service_called = True
        return []

    monkeypatch.setattr(
        read_tools,
        "list_workspaces_service",
        fake_list_workspaces,
    )

    with Session() as session:
        tool = build_list_workspaces_tool(session)
        result = tool.invoke(arguments)

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "invalid_arguments"
    assert service_called is False


def test_list_workspaces_tool_exposes_strict_input_schema() -> None:
    with Session() as session:
        tool = build_list_workspaces_tool(session)

    assert tool.name == "list_workspaces"
    assert tool.input_schema()["additionalProperties"] is False
