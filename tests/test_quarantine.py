from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

import pytest

import backend.app.quarantine as quarantine_module
from backend.app.filesystem_adapter import FileSystemAdapter
from backend.app.path_policy import PathPolicyError, PathPolicyErrorCode
from backend.app.quarantine import (
    QuarantineError,
    QuarantineErrorCode,
    QuarantineManager,
)
from backend.app.v2_file_mover import move_file as real_v2_move_file


PLAN_ID = UUID("2d053752-d3c4-45cb-b696-bd043e78ed92")


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace_root = tmp_path / "workspace"
    source_path = workspace_root / "inbox" / "report.pdf"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"content to quarantine")
    return workspace_root, source_path


def _manager(
    workspace_root: Path,
    quarantine_root: Path,
) -> QuarantineManager:
    return QuarantineManager(
        FileSystemAdapter(workspace_root),
        FileSystemAdapter(quarantine_root),
    )


def _quarantine(
    manager: QuarantineManager,
    source_path: Path = Path("inbox/report.pdf"),
):
    return manager.quarantine(
        source_path,
        workspace_id=3,
        plan_id=PLAN_ID,
        source_file_id=7,
    )


def test_quarantine_moves_file_to_deterministic_isolated_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root, source_path = _workspace(tmp_path)
    quarantine_root = tmp_path / "application-quarantine"
    wrapped_v2_mover = Mock(wraps=real_v2_move_file)
    monkeypatch.setattr(
        quarantine_module,
        "v2_move_file",
        wrapped_v2_mover,
    )

    result = _quarantine(_manager(workspace_root, quarantine_root))

    expected_target = (
        quarantine_root
        / "workspace-3"
        / str(PLAN_ID)
        / "7"
        / "report.pdf"
    ).resolve()
    assert result.original_path == source_path.resolve()
    assert result.quarantine_path == expected_target
    assert expected_target.read_bytes() == b"content to quarantine"
    assert not source_path.exists()
    wrapped_v2_mover.assert_called_once_with(
        source_path.resolve(),
        expected_target,
    )


@pytest.mark.parametrize("quarantine_root_kind", ["same", "inside", "parent"])
def test_quarantine_rejects_overlapping_roots(
    tmp_path: Path,
    quarantine_root_kind: str,
) -> None:
    workspace_root, _ = _workspace(tmp_path)
    if quarantine_root_kind == "same":
        quarantine_root = workspace_root
    elif quarantine_root_kind == "inside":
        quarantine_root = workspace_root / "quarantine"
    else:
        quarantine_root = tmp_path

    with pytest.raises(QuarantineError) as error:
        _manager(workspace_root, quarantine_root)

    assert error.value.code is QuarantineErrorCode.INVALID_ROOTS


@pytest.mark.parametrize(
    ("source_path", "expected_code"),
    [
        (
            Path("../outside.txt"),
            PathPolicyErrorCode.PATH_OUTSIDE_WORKSPACE,
        ),
        (Path(".ssh/id.txt"), PathPolicyErrorCode.SENSITIVE_PATH),
    ],
)
def test_rejected_source_never_creates_quarantine_or_calls_v2_mover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_path: Path,
    expected_code: PathPolicyErrorCode,
) -> None:
    workspace_root, _ = _workspace(tmp_path)
    (workspace_root / ".ssh").mkdir()
    (workspace_root / ".ssh" / "id.txt").write_text(
        "sensitive",
        encoding="utf-8",
    )
    quarantine_root = tmp_path / "application-quarantine"
    v2_mover = Mock()
    monkeypatch.setattr(quarantine_module, "v2_move_file", v2_mover)

    with pytest.raises(PathPolicyError) as error:
        _quarantine(
            _manager(workspace_root, quarantine_root),
            source_path,
        )

    assert error.value.code is expected_code
    assert not quarantine_root.exists()
    v2_mover.assert_not_called()


def test_missing_source_does_not_create_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root, _ = _workspace(tmp_path)
    quarantine_root = tmp_path / "application-quarantine"
    v2_mover = Mock()
    monkeypatch.setattr(quarantine_module, "v2_move_file", v2_mover)

    with pytest.raises(QuarantineError) as error:
        _quarantine(
            _manager(workspace_root, quarantine_root),
            Path("inbox/missing.pdf"),
        )

    assert error.value.code is QuarantineErrorCode.SOURCE_UNAVAILABLE
    assert not quarantine_root.exists()
    v2_mover.assert_not_called()


def test_existing_quarantine_target_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root, source_path = _workspace(tmp_path)
    quarantine_root = tmp_path / "application-quarantine"
    existing_target = (
        quarantine_root
        / "workspace-3"
        / str(PLAN_ID)
        / "7"
        / "report.pdf"
    )
    existing_target.parent.mkdir(parents=True)
    existing_target.write_bytes(b"existing quarantined content")
    v2_mover = Mock()
    monkeypatch.setattr(quarantine_module, "v2_move_file", v2_mover)

    with pytest.raises(QuarantineError) as error:
        _quarantine(_manager(workspace_root, quarantine_root))

    assert error.value.code is QuarantineErrorCode.TARGET_CONFLICT
    assert source_path.read_bytes() == b"content to quarantine"
    assert existing_target.read_bytes() == b"existing quarantined content"
    v2_mover.assert_not_called()


def test_conflict_created_during_quarantine_is_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root, source_path = _workspace(tmp_path)
    quarantine_root = tmp_path / "application-quarantine"
    expected_target = (
        quarantine_root
        / "workspace-3"
        / str(PLAN_ID)
        / "7"
        / "report.pdf"
    )

    def create_competing_target(*args: object, **kwargs: object) -> None:
        expected_target.write_bytes(b"competing content")
        return None

    monkeypatch.setattr(
        quarantine_module,
        "v2_move_file",
        create_competing_target,
    )

    with pytest.raises(QuarantineError) as error:
        _quarantine(_manager(workspace_root, quarantine_root))

    assert error.value.code is QuarantineErrorCode.TARGET_CONFLICT
    assert source_path.read_bytes() == b"content to quarantine"
    assert expected_target.read_bytes() == b"competing content"


def test_v2_failure_becomes_stable_quarantine_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root, source_path = _workspace(tmp_path)
    quarantine_root = tmp_path / "application-quarantine"
    monkeypatch.setattr(
        quarantine_module,
        "v2_move_file",
        Mock(return_value=None),
    )

    with pytest.raises(QuarantineError) as error:
        _quarantine(_manager(workspace_root, quarantine_root))

    assert error.value.code is QuarantineErrorCode.MOVE_FAILED
    assert source_path.read_bytes() == b"content to quarantine"
