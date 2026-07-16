from pathlib import Path

import pytest

from backend.app.filesystem_adapter import FileSystemAdapter
from backend.app.path_policy import PathPolicyError, PathPolicyErrorCode


def test_read_text_reads_file_inside_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    document = workspace_root / "documents" / "report.txt"
    document.parent.mkdir(parents=True)
    document.write_text("FileNest 报告", encoding="utf-8")
    adapter = FileSystemAdapter(workspace_root)

    content = adapter.read_text(Path("documents") / "report.txt")

    assert content == "FileNest 报告"


def test_list_directory_returns_only_authorized_names(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "b-report.txt").write_text("B", encoding="utf-8")
    (workspace_root / "A-report.txt").write_text("A", encoding="utf-8")
    (workspace_root / ".env").write_text("SECRET=value", encoding="utf-8")
    (workspace_root / ".git").mkdir()
    adapter = FileSystemAdapter(workspace_root)

    visible_names = adapter.list_directory()

    assert visible_names == ["A-report.txt", "b-report.txt"]


def test_read_text_rejects_path_outside_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    outside_file = tmp_path / "outside.txt"
    workspace_root.mkdir()
    outside_file.write_text("outside", encoding="utf-8")
    adapter = FileSystemAdapter(workspace_root)

    with pytest.raises(PathPolicyError) as captured_error:
        adapter.read_text(Path("..") / "outside.txt")

    assert captured_error.value.code is PathPolicyErrorCode.PATH_OUTSIDE_WORKSPACE


def test_read_text_rejects_sensitive_file(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / ".env").write_text("SECRET=value", encoding="utf-8")
    adapter = FileSystemAdapter(workspace_root)

    with pytest.raises(PathPolicyError) as captured_error:
        adapter.read_text(Path(".env"))

    assert captured_error.value.code is PathPolicyErrorCode.SENSITIVE_PATH


def test_list_directory_rejects_path_outside_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    adapter = FileSystemAdapter(workspace_root)

    with pytest.raises(PathPolicyError) as captured_error:
        adapter.list_directory(Path(".."))

    assert captured_error.value.code is PathPolicyErrorCode.PATH_OUTSIDE_WORKSPACE
