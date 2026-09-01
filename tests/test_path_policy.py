import os
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from backend.app.filesystem_adapter import FileSystemAdapter
from backend.app.path_policy import (
    AuthorizedPath,
    SensitivePathRules,
    PathPolicyError,
    PathPolicyErrorCode,
    PathPolicyRequest,
    WorkspacePolicy,
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
        (PathPolicyErrorCode.PATH_DENYLISTED, "path_denylisted"),
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
        ".Azure",
        ".Docker",
        ".Kube",
        ".Terraform.d",
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
        ".dockerconfigjson",
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "auth.json",
        "credentials",
        "credentials.json",
        "credentials.tfrc.json",
        "credentials.xml",
        "ID_RSA",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_ecdsa_sk",
        "id_ed25519_sk",
        "id_xmss",
        "private.key",
        "secret.json",
        "secrets.json",
        "service-account.json",
        "kubeconfig",
        "certificate.PEM",
        "certificate.p12",
        "certificate.pfx",
        "truststore.jks",
        "truststore.kdb",
        "truststore.kdbx",
        "truststore.keystore",
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


def test_authorize_path_applies_user_denylist_to_a_subtree(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    denylist = (Path("documents") / "private",)

    allowed = authorize_path(
        PathPolicyRequest(
            workspace_root=workspace_root,
            requested_path=Path("documents") / "public" / "report.txt",
            user_denylist=denylist,
        )
    )

    assert allowed.path == workspace_root / "documents" / "public" / "report.txt"

    with pytest.raises(PathPolicyError) as captured_error:
        authorize_path(
            PathPolicyRequest(
                workspace_root=workspace_root,
                requested_path=Path("documents") / "private" / "secret.txt",
                user_denylist=denylist,
            )
        )

    assert captured_error.value.code is PathPolicyErrorCode.PATH_DENYLISTED


def test_authorize_path_uses_configured_sensitive_path_rules(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    rules = SensitivePathRules(
        directory_names={"private-data"},
        file_names={"service-token"},
        file_name_prefixes={"secret-"},
        file_suffixes={".vault"},
    )

    result = authorize_path(
        PathPolicyRequest(
            workspace_root=workspace_root,
            requested_path=Path("documents") / "report.txt",
            sensitive_path_rules=rules,
        )
    )

    assert result.path == workspace_root / "documents" / "report.txt"

    with pytest.raises(PathPolicyError) as captured_error:
        authorize_path(
            PathPolicyRequest(
                workspace_root=workspace_root,
                requested_path=Path("documents") / "private-data" / "report.txt",
                sensitive_path_rules=rules,
            )
        )

    assert captured_error.value.code is PathPolicyErrorCode.SENSITIVE_PATH


def test_sensitive_path_rules_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="file_names"):
        SensitivePathRules(file_names="service-token")  # type: ignore[arg-type]


def test_workspace_policy_defaults_are_compatible_and_immutable() -> None:
    policy = WorkspacePolicy()

    assert policy.policy_revision == 0
    assert policy.read_enabled is True
    assert policy.proposal_enabled is True
    assert policy.safe_execution_enabled is True
    assert policy.user_denylist == ()
    assert policy.ignore_patterns == ()

    with pytest.raises(FrozenInstanceError):
        policy.read_enabled = False


def test_workspace_policy_only_adds_relative_denylist_and_ignore_rules(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    policy = WorkspacePolicy(
        policy_revision=3,
        user_denylist=("private/data",),
        ignore_patterns=("*.tmp", "cache/**"),
    )
    adapter = FileSystemAdapter(
        workspace_root,
        workspace_policy=policy,
    )

    assert policy.denylisted_paths == (Path("private/data"),)
    assert policy.ignore_policy.matches(Path("draft.tmp"))
    assert policy.ignore_policy.matches(Path("cache/item.txt"))
    assert adapter.workspace_policy == policy
    with pytest.raises(PathPolicyError) as denylisted:
        adapter.authorized_path(Path("private/data/secret.txt"))
    assert denylisted.value.code is PathPolicyErrorCode.PATH_DENYLISTED
    with pytest.raises(PathPolicyError) as sensitive:
        adapter.authorized_path(Path(".env"))
    assert sensitive.value.code is PathPolicyErrorCode.SENSITIVE_PATH


@pytest.mark.parametrize(
    "kwargs",
    [
        {"user_denylist": (".",)},
        {"user_denylist": ("../private",)},
        {"user_denylist": ("C:/private",)},
        {"user_denylist": ("private\\data",)},
        {"user_denylist": ("private", "PRIVATE")},
        {"ignore_patterns": ("",)},
        {"ignore_patterns": ("../*.tmp",)},
        {"ignore_patterns": ("cache//*.tmp",)},
        {"ignore_patterns": ("*.tmp", "*.TMP")},
    ],
)
def test_workspace_policy_rejects_unstable_or_duplicate_rules(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        WorkspacePolicy(**kwargs)  # type: ignore[arg-type]
