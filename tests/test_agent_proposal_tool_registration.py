from pathlib import Path

from sqlalchemy.orm import Session

from backend.app.agent_evaluation import evaluate_forbidden_tools
from backend.app.agent_api import _WorkspaceScopedToolRegistry
from backend.app.workflow_graph import build_workflow_graph


def _registry(tmp_path: Path) -> _WorkspaceScopedToolRegistry:
    return _WorkspaceScopedToolRegistry(
        Session(),
        7,
        build_workflow_graph(),
        tmp_path / "quarantine",
    )


def test_workspace_agent_registry_exposes_read_search_rag_and_proposal_tools(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    assert registry.names == (
        "search_files",
        "get_file_metadata",
        "knowledge_search",
        "propose_move",
        "propose_rename",
        "propose_quarantine",
    )
    assert [definition.name for definition in registry.definitions()] == list(
        registry.names
    )


def test_workspace_agent_registry_rejects_direct_approval_execution_and_undo(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    for blocked_tool in ("approve", "execute", "undo"):
        result = registry.invoke(
            blocked_tool,
            {"workspace_id": 7},
        )

        assert result.ok is False
        assert result.error is not None
        assert result.error.code == "unknown_tool"


def test_workspace_agent_registry_passes_forbidden_tool_evaluation(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    result = evaluate_forbidden_tools(registry.names)

    assert result.passed is True
    assert result.forbidden_tool_names == ()
    assert result.unapproved_tool_names == ()
