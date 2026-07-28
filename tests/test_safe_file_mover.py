from pathlib import Path
from unittest.mock import Mock

import pytest

import backend.app.safe_file_mover as safe_file_mover_module
from backend.app.filesystem_adapter import FileSystemAdapter
from backend.app.path_policy import PathPolicyError, PathPolicyErrorCode
from backend.app.safe_file_mover import (
    SafeFileMoveError,
    SafeFileMoveErrorCode,
    SafeFileMover,
)
from core.file_mover import move_file as real_v1_move_file


def _workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace_root = tmp_path / "safe-move-workspace"
    source_path = workspace_root / "inbox" / "report.pdf"
    target_directory = workspace_root / "documents"
    source_path.parent.mkdir(parents=True)
    target_directory.mkdir(parents=True)
    source_path.write_bytes(b"safe move content")
    return workspace_root, source_path, target_directory


def test_safe_file_mover_authorizes_paths_and_uses_non_overwrite_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root, source_path, target_directory = _workspace(tmp_path)
    wrapped_v1_mover = Mock(wraps=real_v1_move_file)
    monkeypatch.setattr(
        safe_file_mover_module,
        "v1_move_file",
        wrapped_v1_mover,
    )

    mover = SafeFileMover(FileSystemAdapter(workspace_root))
    result = mover.move(
        Path("inbox/report.pdf"),
        Path("documents/report.pdf"),
    )

    expected_target = (target_directory / source_path.name).resolve()
    assert result == expected_target
    assert result.read_bytes() == b"safe move content"
    assert not source_path.exists()
    wrapped_v1_mover.assert_called_once_with(
        source_path.resolve(),
        target_directory.resolve(),
        collision_strategy="skip",
        target_name="report.pdf",
    )


@pytest.mark.parametrize(
    ("source_path", "target_path", "expected_code"),
    [
        (
            Path("../outside.txt"),
            Path("documents/report.pdf"),
            PathPolicyErrorCode.PATH_OUTSIDE_WORKSPACE,
        ),
        (
            Path("inbox/report.pdf"),
            Path("../outside/report.pdf"),
            PathPolicyErrorCode.PATH_OUTSIDE_WORKSPACE,
        ),
        (
            Path(".ssh/id.txt"),
            Path("documents/id.txt"),
            PathPolicyErrorCode.SENSITIVE_PATH,
        ),
        (
            Path("inbox/report.pdf"),
            Path(".git/report.pdf"),
            PathPolicyErrorCode.SENSITIVE_PATH,
        ),
    ],
)
def test_rejected_path_never_reaches_v1_mover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_path: Path,
    target_path: Path,
    expected_code: PathPolicyErrorCode,
) -> None:
    workspace_root, _, _ = _workspace(tmp_path)
    (workspace_root / ".ssh").mkdir()
    (workspace_root / ".ssh" / "id.txt").write_text(
        "sensitive",
        encoding="utf-8",
    )
    v1_mover = Mock()
    monkeypatch.setattr(
        safe_file_mover_module,
        "v1_move_file",
        v1_mover,
    )

    mover = SafeFileMover(FileSystemAdapter(workspace_root))
    with pytest.raises(PathPolicyError) as error:
        mover.move(source_path, target_path)

    assert error.value.code is expected_code
    v1_mover.assert_not_called()


def test_missing_target_directory_is_not_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root, source_path, _ = _workspace(tmp_path)
    missing_directory = workspace_root / "missing"
    v1_mover = Mock()
    monkeypatch.setattr(
        safe_file_mover_module,
        "v1_move_file",
        v1_mover,
    )

    mover = SafeFileMover(FileSystemAdapter(workspace_root))
    with pytest.raises(SafeFileMoveError) as error:
        mover.move(
            Path("inbox/report.pdf"),
            Path("missing/report.pdf"),
        )

    assert (
        error.value.code
        is SafeFileMoveErrorCode.TARGET_DIRECTORY_UNAVAILABLE
    )
    assert source_path.exists()
    assert not missing_directory.exists()
    v1_mover.assert_not_called()


def test_existing_target_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root, source_path, target_directory = _workspace(tmp_path)
    existing_target = target_directory / source_path.name
    existing_target.write_bytes(b"existing content")
    v1_mover = Mock()
    monkeypatch.setattr(
        safe_file_mover_module,
        "v1_move_file",
        v1_mover,
    )

    mover = SafeFileMover(FileSystemAdapter(workspace_root))
    with pytest.raises(SafeFileMoveError) as error:
        mover.move(
            Path("inbox/report.pdf"),
            Path("documents/report.pdf"),
        )

    assert error.value.code is SafeFileMoveErrorCode.TARGET_CONFLICT
    assert source_path.read_bytes() == b"safe move content"
    assert existing_target.read_bytes() == b"existing content"
    v1_mover.assert_not_called()


def test_v1_failure_becomes_stable_safe_move_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root, source_path, _ = _workspace(tmp_path)
    v1_mover = Mock(return_value=None)
    monkeypatch.setattr(
        safe_file_mover_module,
        "v1_move_file",
        v1_mover,
    )

    mover = SafeFileMover(FileSystemAdapter(workspace_root))
    with pytest.raises(SafeFileMoveError) as error:
        mover.move(
            Path("inbox/report.pdf"),
            Path("documents/report.pdf"),
        )

    assert error.value.code is SafeFileMoveErrorCode.MOVE_FAILED
    assert source_path.read_bytes() == b"safe move content"


def test_safe_file_mover_moves_and_renames_to_exact_target(
    tmp_path: Path,
) -> None:
    workspace_root, source_path, _ = _workspace(tmp_path)
    archive_directory = workspace_root / "archive"
    archive_directory.mkdir()

    mover = SafeFileMover(FileSystemAdapter(workspace_root))
    result = mover.move(
        Path("inbox/report.pdf"),
        Path("archive/approved-report.pdf"),
    )

    assert result == (archive_directory / "approved-report.pdf").resolve()
    assert result.read_bytes() == b"safe move content"
    assert not source_path.exists()


def test_safe_file_mover_renames_inside_same_directory(tmp_path: Path) -> None:
    workspace_root, source_path, _ = _workspace(tmp_path)

    mover = SafeFileMover(FileSystemAdapter(workspace_root))
    result = mover.move(
        Path("inbox/report.pdf"),
        Path("inbox/final-report.pdf"),
    )

    assert result == (source_path.parent / "final-report.pdf").resolve()
    assert result.read_bytes() == b"safe move content"
    assert not source_path.exists()


def test_conflict_created_during_move_is_reported_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root, source_path, target_directory = _workspace(tmp_path)
    target_path = target_directory / "report.pdf"

    def create_competing_target(*args: object, **kwargs: object) -> None:
        target_path.write_bytes(b"competing content")
        return None

    monkeypatch.setattr(
        safe_file_mover_module,
        "v1_move_file",
        create_competing_target,
    )

    mover = SafeFileMover(FileSystemAdapter(workspace_root))
    with pytest.raises(SafeFileMoveError) as error:
        mover.move(
            Path("inbox/report.pdf"),
            Path("documents/report.pdf"),
        )

    assert error.value.code is SafeFileMoveErrorCode.TARGET_CONFLICT
    assert source_path.read_bytes() == b"safe move content"
    assert target_path.read_bytes() == b"competing content"


def test_v1_target_name_is_optional_and_rejects_path_components(
    tmp_path: Path,
) -> None:
    legacy_source = tmp_path / "legacy.txt"
    legacy_target_directory = tmp_path / "legacy-target"
    legacy_source.write_text("legacy", encoding="utf-8")

    legacy_result = real_v1_move_file(
        legacy_source,
        legacy_target_directory,
    )

    assert legacy_result == (legacy_target_directory / "legacy.txt").resolve()

    unsafe_source = tmp_path / "unsafe.txt"
    unsafe_target_directory = tmp_path / "unsafe-target"
    unsafe_source.write_text("unsafe", encoding="utf-8")
    rejected_result = real_v1_move_file(
        unsafe_source,
        unsafe_target_directory,
        target_name="../escape.txt",
    )

    assert rejected_result is None
    assert unsafe_source.read_text(encoding="utf-8") == "unsafe"
    assert not unsafe_target_directory.exists()
    assert not (tmp_path / "escape.txt").exists()
