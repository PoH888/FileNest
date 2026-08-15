"""路径安全策略的输入、成功输出与结构化错误契约。"""

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypedDict


_SENSITIVE_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".ssh",
        ".aws",
        ".gnupg",
        "$recycle.bin",
        "system volume information",
    }
)
_SENSITIVE_FILE_NAMES = frozenset(
    {
        ".env",
        "id_rsa",
        "id_ed25519",
    }
)
_SENSITIVE_FILE_SUFFIXES = frozenset(
    {
        ".key",
        ".pem",
        ".p12",
        ".pfx",
    }
)


class PathPolicyErrorCode(StrEnum):
    """Path Policy 对外稳定的错误码。"""

    INVALID_PATH = "invalid_path"
    WORKSPACE_ROOT_NOT_FOUND = "workspace_root_not_found"
    WORKSPACE_ROOT_NOT_DIRECTORY = "workspace_root_not_directory"
    PATH_OUTSIDE_WORKSPACE = "path_outside_workspace"
    PATH_LINK_OUTSIDE_WORKSPACE = "path_link_outside_workspace"
    SENSITIVE_PATH = "sensitive_path"


class PathPolicyErrorDetail(TypedDict):
    """可直接交给 API 错误响应的结构。"""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class PathPolicyRequest:
    """一次路径授权判断所需的最小输入。"""

    workspace_root: Path
    requested_path: Path


@dataclass(frozen=True, slots=True)
class AuthorizedPath:
    """Path Policy 判断通过后交给文件系统层的路径。"""

    workspace_root: Path
    path: Path


class PathPolicyError(Exception):
    """路径未通过安全策略时产生的结构化业务错误。"""

    def __init__(
        self,
        code: PathPolicyErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_detail(self) -> PathPolicyErrorDetail:
        """生成与现有 FastAPI 错误响应一致的 detail。"""

        return {
            "code": self.code.value,
            "message": self.message,
        }


def normalize_workspace_root(root_path: str | Path) -> Path:
    """将工作区根路径统一为绝对、解析后的路径。"""

    try:
        path = Path(root_path)
    except (TypeError, ValueError) as error:
        raise PathPolicyError(
            PathPolicyErrorCode.INVALID_PATH,
            "路径格式无效。",
        ) from error

    return _resolve_path(_normalize_path(path))


def validate_workspace_root(root_path: str | Path) -> Path:
    """验证工作区根路径安全可用，并返回规范化路径。"""

    normalized_root = normalize_workspace_root(root_path)

    if _is_sensitive_path(normalized_root):
        raise PathPolicyError(
            PathPolicyErrorCode.SENSITIVE_PATH,
            "请求路径属于受保护的敏感文件或目录。",
        )

    try:
        if not normalized_root.exists():
            raise PathPolicyError(
                PathPolicyErrorCode.WORKSPACE_ROOT_NOT_FOUND,
                "工作区根目录不存在。",
            )
        if not normalized_root.is_dir():
            raise PathPolicyError(
                PathPolicyErrorCode.WORKSPACE_ROOT_NOT_DIRECTORY,
                "工作区根路径不是目录。",
            )
    except OSError as error:
        raise PathPolicyError(
            PathPolicyErrorCode.INVALID_PATH,
            "路径格式无效。",
        ) from error

    return normalized_root


def authorize_path(request: PathPolicyRequest) -> AuthorizedPath:
    """规范化请求路径，并确保结果仍位于授权工作区内。"""

    workspace_root = normalize_workspace_root(request.workspace_root)
    requested_path = request.requested_path

    if not requested_path.is_absolute():
        requested_path = workspace_root / requested_path

    normalized_path = _normalize_path(requested_path)

    if not _is_within(normalized_path, workspace_root):
        raise PathPolicyError(
            PathPolicyErrorCode.PATH_OUTSIDE_WORKSPACE,
            "请求路径超出授权工作区。",
        )

    resolved_path = _resolve_path(normalized_path)

    if not _is_within(resolved_path, workspace_root):
        raise PathPolicyError(
            PathPolicyErrorCode.PATH_LINK_OUTSIDE_WORKSPACE,
            "请求路径通过链接指向授权工作区之外。",
        )

    if _is_sensitive_path(resolved_path):
        raise PathPolicyError(
            PathPolicyErrorCode.SENSITIVE_PATH,
            "请求路径属于受保护的敏感文件或目录。",
        )

    return AuthorizedPath(
        workspace_root=workspace_root,
        path=resolved_path,
    )


def _normalize_path(path: Path) -> Path:
    """只做词法规范化，不跟随文件系统中的链接。"""

    try:
        return Path(os.path.abspath(path))
    except (OSError, TypeError, ValueError) as error:
        raise PathPolicyError(
            PathPolicyErrorCode.INVALID_PATH,
            "路径格式无效。",
        ) from error


def _resolve_path(path: Path) -> Path:
    """解析已存在的 symlink/junction，暴露其真实目标。"""

    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise PathPolicyError(
            PathPolicyErrorCode.INVALID_PATH,
            "路径格式无效。",
        ) from error


def _is_within(path: Path, workspace_root: Path) -> bool:
    """按完整路径层级判断归属，避免相似字符串前缀绕过。"""

    try:
        path.relative_to(workspace_root)
    except ValueError:
        return False

    return True


def _is_sensitive_path(path: Path) -> bool:
    """对完整路径段和最终文件名应用统一的敏感项规则。"""

    normalized_parts = (part.casefold() for part in path.parts)
    if any(part in _SENSITIVE_DIRECTORY_NAMES for part in normalized_parts):
        return True

    file_name = path.name.casefold()
    return (
        file_name in _SENSITIVE_FILE_NAMES
        or file_name.startswith(".env.")
        or path.suffix.casefold() in _SENSITIVE_FILE_SUFFIXES
    )
