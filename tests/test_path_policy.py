import os
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from backend.app.path_policy import (
    AuthorizedPath,
    PathPolicyError,
    PathPolicyErrorCode,
    PathPolicyRequest,
    authorize_path,
)


def test_path_policy_request_keeps_authorization_boundary() -> None:
    workspace_root = Path("D:/Workspaces/Project")
    requested_path = Path("docs/report.txt")

    request = PathPolicyRequest(
        workspace_root=workspace_root,
        requested_path=requested_path,
    )

    assert request.workspace_root == workspace_root
    assert request.requested_path == requested_path


def test_authorized_path_keeps_policy_output() -> None:
    workspace_root = Path("D:/Workspaces/Project")
    safe_path = workspace_root / "docs" / "report.txt"

    result = AuthorizedPath(
        workspace_root=workspace_root,
        path=safe_path,
    )

    assert result.workspace_root == workspace_root
    assert result.path == safe_path


def test_path_policy_contracts_are_immutable() -> None:
    request = PathPolicyRequest(
        workspace_root=Path("D:/Workspaces/Project"),
        requested_path=Path("docs/report.txt"),
    )

    with pytest.raises(FrozenInstanceError):
        request.requested_path = Path("other.txt")


@pytest.mark.parametrize(
    ("error_code", "expected_value"),
    [
        (PathPolicyErrorCode.INVALID_PATH, "invalid_path"),
        (
            PathPolicyErrorCode.PATH_OUTSIDE_WORKSPACE,
            "path_outside_workspace",
        ),
        (
            PathPolicyErrorCode.PATH_LINK_OUTSIDE_WORKSPACE,
            "path_link_outside_workspace",
        ),
        (PathPolicyErrorCode.SENSITIVE_PATH, "sensitive_path"),
    ],
)
def test_path_policy_error_codes_are_stable(
    error_code: PathPolicyErrorCode,
    expected_value: str,
) -> None:
    assert error_code.value == expected_value


def test_path_policy_error_provides_structured_detail() -> None:
    error = PathPolicyError(
        PathPolicyErrorCode.PATH_OUTSIDE_WORKSPACE,
        "请求路径超出授权工作区。",
    )

    assert str(error) == "请求路径超出授权工作区。"
    assert error.as_detail() == {
        "code": "path_outside_workspace",
        "message": "请求路径超出授权工作区。",
    }


def test_authorize_path_normalizes_relative_path_inside_workspace(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    request = PathPolicyRequest(
        workspace_root=workspace_root,
        requested_path=Path("docs") / "." / "drafts" / ".." / "report.txt",
    )

    result = authorize_path(request)

    assert result == AuthorizedPath(
        workspace_root=workspace_root,
        path=workspace_root / "docs" / "report.txt",
    )


def test_authorize_path_accepts_absolute_path_inside_workspace(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    requested_path = workspace_root / "docs" / "report.txt"

    result = authorize_path(
        PathPolicyRequest(
            workspace_root=workspace_root,
            requested_path=requested_path,
        )
    )

    assert result.path == requested_path


@pytest.mark.parametrize(
    "requested_path",
    [
        Path("..") / "outside.txt",
        Path("..") / "workspace-backup" / "outside.txt",
    ],
)
def test_authorize_path_rejects_relative_path_outside_workspace(
    tmp_path: Path,
    requested_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"

    with pytest.raises(PathPolicyError) as captured_error:
        authorize_path(
            PathPolicyRequest(
                workspace_root=workspace_root,
                requested_path=requested_path,
            )
        )

    assert captured_error.value.as_detail() == {
        "code": "path_outside_workspace",
        "message": "请求路径超出授权工作区。",
    }


def test_authorize_path_rejects_absolute_path_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    requested_path = tmp_path / "other" / "outside.txt"

    with pytest.raises(PathPolicyError) as captured_error:
        authorize_path(
            PathPolicyRequest(
                workspace_root=workspace_root,
                requested_path=requested_path,
            )
        )

    assert captured_error.value.code is PathPolicyErrorCode.PATH_OUTSIDE_WORKSPACE


def test_authorize_path_rejects_symlink_to_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    outside_root = tmp_path / "outside"
    workspace_root.mkdir()
    outside_root.mkdir()

    linked_directory = workspace_root / "linked-directory"
    try:
        linked_directory.symlink_to(outside_root, target_is_directory=True)
    except OSError as error:
        if error.winerror == 1314:
            pytest.skip("当前 Windows 环境未授权普通进程创建 symlink")
        raise

    with pytest.raises(PathPolicyError) as captured_error:
        authorize_path(
            PathPolicyRequest(
                workspace_root=workspace_root,
                requested_path=linked_directory / "outside.txt",
            )
        )

    assert captured_error.value.as_detail() == {
        "code": "path_link_outside_workspace",
        "message": "请求路径通过链接指向授权工作区之外。",
    }


@pytest.mark.skipif(os.name != "nt", reason="junction 是 Windows 专属能力")
def test_authorize_path_rejects_junction_to_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    outside_root = tmp_path / "outside"
    workspace_root.mkdir()
    outside_root.mkdir()

    junction = workspace_root / "linked-junction"
    subprocess.run(
        [
            os.environ.get("COMSPEC", "cmd.exe"),
            "/d",
            "/c",
            "mklink",
            "/J",
            str(junction),
            str(outside_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert os.path.isjunction(junction)

    with pytest.raises(PathPolicyError) as captured_error:
        authorize_path(
            PathPolicyRequest(
                workspace_root=workspace_root,
                requested_path=junction / "outside.txt",
            )
        )

    assert (
        captured_error.value.code
        is PathPolicyErrorCode.PATH_LINK_OUTSIDE_WORKSPACE
    )


@pytest.mark.parametrize(
    "sensitive_directory",
    [
        ".git",
        ".SSH",
        ".aws",
        ".gnupg",
        "$RECYCLE.BIN",
        "System Volume Information",
    ],
)
def test_authorize_path_rejects_sensitive_directory(
    tmp_path: Path,
    sensitive_directory: str,
) -> None:
    workspace_root = tmp_path / "workspace"

    with pytest.raises(PathPolicyError) as captured_error:
        authorize_path(
            PathPolicyRequest(
                workspace_root=workspace_root,
                requested_path=Path("documents")
                / sensitive_directory
                / "protected.txt",
            )
        )

    assert captured_error.value.as_detail() == {
        "code": "sensitive_path",
        "message": "请求路径属于受保护的敏感文件或目录。",
    }


@pytest.mark.parametrize(
    "sensitive_file",
    [
        ".env",
        ".env.production",
        "ID_RSA",
        "id_ed25519",
        "private.key",
        "certificate.PEM",
        "certificate.p12",
        "certificate.pfx",
    ],
)
def test_authorize_path_rejects_sensitive_file(
    tmp_path: Path,
    sensitive_file: str,
) -> None:
    workspace_root = tmp_path / "workspace"

    with pytest.raises(PathPolicyError) as captured_error:
        authorize_path(
            PathPolicyRequest(
                workspace_root=workspace_root,
                requested_path=Path("documents") / sensitive_file,
            )
        )

    assert captured_error.value.code is PathPolicyErrorCode.SENSITIVE_PATH


@pytest.mark.parametrize(
    "safe_path",
    [
        Path("documents") / "my.git.notes",
        Path("documents") / "environment.txt",
        Path("documents") / "public-key.txt",
    ],
)
def test_authorize_path_allows_non_sensitive_similar_names(
    tmp_path: Path,
    safe_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"

    result = authorize_path(
        PathPolicyRequest(
            workspace_root=workspace_root,
            requested_path=safe_path,
        )
    )

    assert result.path == workspace_root / safe_path
