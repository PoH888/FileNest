from pathlib import Path

import pytest

from backend.app.scale_test_workspaces import (
    SCALE_PROFILES,
    ScaleWorkspaceError,
    generate_scale_workspace,
)


def _relative_files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )


def test_scale_profiles_and_small_workspace_are_reproducible(
    tmp_path: Path,
) -> None:
    assert tuple(SCALE_PROFILES) == ("small", "medium", "large")
    assert [profile.file_count for profile in SCALE_PROFILES.values()] == [
        100,
        1_000,
        10_000,
    ]

    first_root = tmp_path / "small-first"
    second_root = tmp_path / "small-second"
    first_manifest = generate_scale_workspace(first_root, "small")
    second_manifest = generate_scale_workspace(second_root, "small")

    assert first_manifest == second_manifest
    assert first_manifest.file_count == 100
    assert first_manifest.document_file_count == 80
    assert first_manifest.non_document_file_count == 20
    assert first_manifest.leaf_directory_count == 10
    assert len(_relative_files(first_root)) == 100
    assert _relative_files(first_root) == _relative_files(second_root)
    assert (
        first_root / "area-000" / "bucket-0000" / "item-000000.md"
    ).read_text(encoding="utf-8").startswith(
        "FileNest scale fixture\nscale=small seed=3501 file=000000\n"
    )


def test_generator_rejects_existing_directory_without_overwriting(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "existing"
    output_root.mkdir()
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    with pytest.raises(ScaleWorkspaceError, match="must not already exist"):
        generate_scale_workspace(output_root, "small")

    assert sentinel.read_text(encoding="utf-8") == "keep me"
    assert _relative_files(output_root) == ["sentinel.txt"]
