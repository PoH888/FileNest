from pathlib import Path

import pytest

from backend.app.filesystem_adapter import FileSystemAdapter
from backend.app.path_policy import (
    PathPolicyError,
    PathPolicyErrorCode,
    SensitivePathWriteAuditRecord,
)


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


def test_authorized_write_path_records_sensitive_rejection(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    records: list[SensitivePathWriteAuditRecord] = []
    adapter = FileSystemAdapter(
        workspace_root,
        sensitive_path_audit_writer=records.append,
    )

    allowed_path = adapter.authorized_write_path(Path("documents/report.txt"))

    assert allowed_path == workspace_root / "documents/report.txt"
    assert records == []

    with pytest.raises(PathPolicyError) as captured_error:
        adapter.authorized_write_path(Path(".env"))

    assert captured_error.value.code is PathPolicyErrorCode.SENSITIVE_PATH
    assert len(records) == 1
    assert records[0].requested_path == Path(".env")
    assert records[0].operation == "write"
    assert records[0].outcome == "denied"
    assert records[0].reason == "sensitive_path"


def test_read_text_rejects_user_denylisted_path(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    denied_file = workspace_root / "private" / "secret.txt"
    denied_file.parent.mkdir(parents=True)
    denied_file.write_text("secret", encoding="utf-8")
    adapter = FileSystemAdapter(
        workspace_root,
        user_denylist=(Path("private"),),
    )

    with pytest.raises(PathPolicyError) as captured_error:
        adapter.read_text(Path("private") / "secret.txt")

    assert captured_error.value.code is PathPolicyErrorCode.PATH_DENYLISTED


def test_list_directory_rejects_path_outside_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    adapter = FileSystemAdapter(workspace_root)

    with pytest.raises(PathPolicyError) as captured_error:
        adapter.list_directory(Path(".."))

    assert captured_error.value.code is PathPolicyErrorCode.PATH_OUTSIDE_WORKSPACE
