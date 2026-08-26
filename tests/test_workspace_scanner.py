from pathlib import Path

from backend.app.path_policy import GlobalIgnorePolicy
from backend.app.workspace_scanner import IgnoredEntry, scan_workspace_files


def test_scan_workspace_files_returns_safe_relative_metadata(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    root_file = workspace_root / "Report.TXT"
    root_file.write_text("root", encoding="utf-8")

    included_directory = workspace_root / "Documents"
    included_directory.mkdir()
    nested_file = included_directory / "note.md"
    nested_file.write_text("nested", encoding="utf-8")

    deep_directory = included_directory / "Deep"
    deep_directory.mkdir()
    (deep_directory / "too-deep.txt").write_text("deep", encoding="utf-8")

    ignored_directory = workspace_root / "Ignored"
    ignored_directory.mkdir()
    (ignored_directory / "ignored.txt").write_text("ignored", encoding="utf-8")

    (workspace_root / ".env").write_text("SECRET=value", encoding="utf-8")
    git_directory = workspace_root / ".git"
    git_directory.mkdir()
    (git_directory / "config").write_text("secret", encoding="utf-8")

    scanned_files = scan_workspace_files(
        workspace_root,
        max_depth=1,
        ignore_patterns=["Ignored"],
    )

    assert [file.relative_path for file in scanned_files] == [
        "Documents/note.md",
        "Report.TXT",
    ]

    files_by_path = {file.relative_path: file for file in scanned_files}
    root_result = files_by_path["Report.TXT"]
    nested_result = files_by_path["Documents/note.md"]

    assert root_result.name == "Report.TXT"
    assert root_result.extension == ".txt"
    assert root_result.size_bytes == root_file.stat().st_size
    assert root_result.mtime_ns == root_file.stat().st_mtime_ns
    assert nested_result.size_bytes == nested_file.stat().st_size


def test_scan_workspace_files_returns_empty_for_missing_workspace(
    tmp_path: Path,
) -> None:
    scanned_files = scan_workspace_files(
        tmp_path / "missing",
        ignore_patterns=[],
    )

    assert scanned_files == []


def test_scan_workspace_files_applies_global_ignore_policy(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "report.txt").write_text("report", encoding="utf-8")
    (workspace_root / "draft.secret").write_text("secret", encoding="utf-8")
    ignored_directory = workspace_root / "Generated"
    ignored_directory.mkdir()
    (ignored_directory / "build.txt").write_text("build", encoding="utf-8")

    scanned_files = scan_workspace_files(
        workspace_root,
        ignore_patterns=[],
        global_ignore_policy=GlobalIgnorePolicy(
            patterns={"Generated", "*.secret"},
        ),
    )

    assert [file.relative_path for file in scanned_files] == ["report.txt"]


def test_scan_workspace_files_records_ignored_reason(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "report.txt").write_text("report", encoding="utf-8")
    (workspace_root / ".env").write_text("SECRET=value", encoding="utf-8")
    (workspace_root / "draft.tmp").write_text("draft", encoding="utf-8")
    ignored_entries: list[IgnoredEntry] = []

    scanned_files = scan_workspace_files(
        workspace_root,
        ignore_patterns=[],
        global_ignore_policy=GlobalIgnorePolicy(patterns={"*.tmp"}),
        ignored_entries=ignored_entries,
    )

    assert [file.relative_path for file in scanned_files] == ["report.txt"]
    assert {
        entry.relative_path: entry.ignored_reason
        for entry in ignored_entries
    } == {
        ".env": "sensitive_path",
        "draft.tmp": "global_ignore",
    }
