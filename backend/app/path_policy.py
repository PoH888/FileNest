"""路径安全策略的输入、成功输出与结构化错误契约。"""

import fnmatch
import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TypedDict


_SENSITIVE_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".ssh",
        ".aws",
        ".gnupg",
        ".azure",
        ".docker",
        ".kube",
        ".terraform.d",
        "$recycle.bin",
        "system volume information",
    }
)
_SENSITIVE_FILE_NAMES = frozenset(
    {
        ".env",
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
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_ecdsa_sk",
        "id_ed25519_sk",
        "id_xmss",
        "secret.json",
        "secrets.json",
        "service-account.json",
        "kubeconfig",
    }
)
_SENSITIVE_FILE_SUFFIXES = frozenset(
    {
        ".key",
        ".pem",
        ".p12",
        ".pfx",
        ".jks",
        ".kdb",
        ".kdbx",
        ".keystore",
    }
)


def _normalize_rule_values(
    values: Iterable[str],
    rule_name: str,
) -> frozenset[str]:
    """规范化配置项，避免大小写差异绕过敏感规则。"""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{rule_name} 必须是字符串集合。")

    try:
        normalized_values = tuple(values)
    except TypeError as error:
        raise ValueError(f"{rule_name} 必须是字符串集合。") from error

    if any(not isinstance(value, str) or not value for value in normalized_values):
        raise ValueError(f"{rule_name} 必须只包含非空字符串。")

    return frozenset(value.casefold() for value in normalized_values)


@dataclass(frozen=True, slots=True)
class SensitivePathRules:
    """可注入的敏感路径规则集合。"""

    directory_names: frozenset[str] = frozenset()
    file_names: frozenset[str] = frozenset()
    file_name_prefixes: frozenset[str] = frozenset()
    file_suffixes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "directory_names",
            _normalize_rule_values(self.directory_names, "directory_names"),
        )
        object.__setattr__(
            self,
            "file_names",
            _normalize_rule_values(self.file_names, "file_names"),
        )
        object.__setattr__(
            self,
            "file_name_prefixes",
            _normalize_rule_values(
                self.file_name_prefixes,
                "file_name_prefixes",
            ),
        )
        object.__setattr__(
            self,
            "file_suffixes",
            _normalize_rule_values(self.file_suffixes, "file_suffixes"),
        )


DEFAULT_SENSITIVE_PATH_RULES = SensitivePathRules(
    directory_names=_SENSITIVE_DIRECTORY_NAMES,
    file_names=_SENSITIVE_FILE_NAMES,
    file_name_prefixes=frozenset({".env."}),
    file_suffixes=_SENSITIVE_FILE_SUFFIXES,
)


@dataclass(frozen=True, slots=True)
class GlobalIgnorePolicy:
    """对工作区扫描结果统一生效的忽略模式。"""

    patterns: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "patterns",
            _normalize_rule_values(self.patterns, "patterns"),
        )

    def matches(self, path: Path) -> bool:
        """按文件名或工作区相对路径匹配忽略模式。"""

        normalized_path = path.as_posix().lstrip("./").casefold()
        file_name = path.name.casefold()
        return any(
            fnmatch.fnmatchcase(file_name, pattern)
            or fnmatch.fnmatchcase(normalized_path, pattern)
            for pattern in self.patterns
        )


DEFAULT_GLOBAL_IGNORE_POLICY = GlobalIgnorePolicy()


def _normalize_workspace_relative_values(
    values: Iterable[str],
    value_name: str,
    *,
    allow_glob: bool,
) -> tuple[str, ...]:
    """固定 Workspace Policy 的相对路径/模式表示，供后续持久化复用。"""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{value_name} 必须是字符串集合。")
    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise ValueError(f"{value_name} 必须是字符串集合。") from error
    if len(raw_values) > 200:
        raise ValueError(f"{value_name} 不能超过 200 项。")

    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or "\\" in value
            or len(value) > 500
        ):
            raise ValueError(f"{value_name} 必须是规范化 workspace 相对路径。")
        if not allow_glob and any(marker in value for marker in "*?["):
            raise ValueError(f"{value_name} 不能包含通配符。")

        path = PurePosixPath(value)
        if (
            value == "."
            or path.is_absolute()
            or PureWindowsPath(value).drive
            or any(part in {"", ".", ".."} for part in path.parts)
            or str(path) != value
        ):
            raise ValueError(f"{value_name} 必须是规范化 workspace 相对路径。")

        folded = value.casefold()
        if folded in seen:
            raise ValueError(f"{value_name} 不能包含重复项。")
        seen.add(folded)
        normalized.append(value)

    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class WorkspacePolicy:
    """只能收窄全局安全边界的 Workspace 级策略契约。"""

    policy_revision: int = 0
    # 默认值保持既有 Workspace 的读、提案和执行行为；新增策略只能显式收窄。
    read_enabled: bool = True
    proposal_enabled: bool = True
    safe_execution_enabled: bool = True
    user_denylist: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.policy_revision, int)
            or isinstance(self.policy_revision, bool)
            or self.policy_revision < 0
        ):
            raise ValueError("policy_revision must be a non-negative integer")
        for field_name in (
            "read_enabled",
            "proposal_enabled",
            "safe_execution_enabled",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")

        normalized_denylist = _normalize_workspace_relative_values(
            self.user_denylist,
            "user_denylist",
            allow_glob=False,
        )
        normalized_patterns = _normalize_workspace_relative_values(
            self.ignore_patterns,
            "ignore_patterns",
            allow_glob=True,
        )
        object.__setattr__(self, "user_denylist", normalized_denylist)
        object.__setattr__(self, "ignore_patterns", normalized_patterns)

    @property
    def denylisted_paths(self) -> tuple[Path, ...]:
        """把已验证的 workspace 相对 denylist 转换为 Adapter 输入。"""

        return tuple(Path(value) for value in self.user_denylist)

    @property
    def ignore_policy(self) -> GlobalIgnorePolicy:
        """返回叠加在全局忽略策略之上的 workspace 忽略策略。"""

        return GlobalIgnorePolicy(frozenset(self.ignore_patterns))

    def ignores(self, relative_path: Path) -> bool:
        """判断相对路径或其父目录是否命中 Workspace 忽略规则。"""

        path = PurePosixPath(relative_path.as_posix())
        parts = tuple(part for part in path.parts if part not in {"", "."})
        if not parts:
            return False

        return any(
            self.ignore_policy.matches(Path(PurePosixPath(*parts[:index])))
            for index in range(1, len(parts) + 1)
        )


DEFAULT_WORKSPACE_POLICY = WorkspacePolicy()


class WorkspacePolicyPersistenceError(ValueError):
    """持久化策略缺失或无法通过严格契约解析。"""


def serialize_workspace_policy_rules(
    policy: WorkspacePolicy,
) -> tuple[str, str]:
    """以稳定 JSON 序列化已校验的 denylist 和 ignore patterns。"""

    return (
        json.dumps(
            list(policy.user_denylist),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        json.dumps(
            list(policy.ignore_patterns),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def workspace_policy_rule_summary(
    policy: WorkspacePolicy,
) -> dict[str, list[str]]:
    """生成只包含规则增删所需字段的审计摘要。"""

    return {
        "user_denylist": list(policy.user_denylist),
        "ignore_patterns": list(policy.ignore_patterns),
    }


def parse_workspace_policy(
    *,
    policy_revision: int,
    read_enabled: bool,
    proposal_enabled: bool,
    safe_execution_enabled: bool,
    user_denylist_json: str,
    ignore_patterns_json: str,
) -> WorkspacePolicy:
    """严格解析数据库策略；任何损坏都不能回退到全局默认策略。"""

    try:
        user_denylist = _parse_policy_rule_list(
            user_denylist_json,
            "user_denylist_json",
        )
        ignore_patterns = _parse_policy_rule_list(
            ignore_patterns_json,
            "ignore_patterns_json",
        )
        return WorkspacePolicy(
            policy_revision=policy_revision,
            read_enabled=read_enabled,
            proposal_enabled=proposal_enabled,
            safe_execution_enabled=safe_execution_enabled,
            user_denylist=user_denylist,
            ignore_patterns=ignore_patterns,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, WorkspacePolicyPersistenceError):
            raise
        raise WorkspacePolicyPersistenceError(
            "持久化 Workspace Policy 无法通过严格校验"
        ) from error


def _parse_policy_rule_list(raw_json: str, field_name: str) -> tuple[str, ...]:
    """只接受 JSON 字符串数组，不把损坏值静默转成空策略。"""

    try:
        parsed = json.loads(raw_json)
    except (TypeError, json.JSONDecodeError) as error:
        raise WorkspacePolicyPersistenceError(
            f"{field_name} 不是有效 JSON"
        ) from error
    if not isinstance(parsed, list) or any(
        not isinstance(value, str) for value in parsed
    ):
        raise WorkspacePolicyPersistenceError(
            f"{field_name} 必须是字符串数组"
        )
    return tuple(parsed)


class PathPolicyErrorCode(StrEnum):
    """Path Policy 对外稳定的错误码。"""

    INVALID_PATH = "invalid_path"
    WORKSPACE_ROOT_NOT_FOUND = "workspace_root_not_found"
    WORKSPACE_ROOT_NOT_DIRECTORY = "workspace_root_not_directory"
    PATH_OUTSIDE_WORKSPACE = "path_outside_workspace"
    PATH_LINK_OUTSIDE_WORKSPACE = "path_link_outside_workspace"
    SENSITIVE_PATH = "sensitive_path"
    PATH_DENYLISTED = "path_denylisted"
    WORKSPACE_IGNORED = "workspace_ignore"


@dataclass(frozen=True, slots=True)
class SensitivePathWriteAuditRecord:
    """一次被拒绝的敏感路径写访问审计记录。"""

    workspace_root: Path
    requested_path: Path
    operation: str = "write"
    outcome: str = "denied"
    reason: str = PathPolicyErrorCode.SENSITIVE_PATH.value
    recorded_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


def write_sensitive_path_audit(
    record: SensitivePathWriteAuditRecord,
) -> None:
    """通过现有日志管道写入结构化敏感路径审计记录。"""

    logging.getLogger("FileNest.security_audit").warning(
        "敏感路径写访问被拒绝",
        extra={
            "audit_event": "sensitive_path_write",
            "audit_workspace_root": str(record.workspace_root),
            "audit_requested_path": str(record.requested_path),
            "audit_operation": record.operation,
            "audit_outcome": record.outcome,
            "audit_reason": record.reason,
            "audit_recorded_at": record.recorded_at.isoformat(),
        },
    )


class PathPolicyErrorDetail(TypedDict):
    """可直接交给 API 错误响应的结构。"""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class PathPolicyRequest:
    """一次路径授权判断所需的最小输入。"""

    workspace_root: Path
    requested_path: Path
    sensitive_path_rules: SensitivePathRules = DEFAULT_SENSITIVE_PATH_RULES
    user_denylist: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        """将用户提供的 denylist 固定为不可变路径集合。"""

        if isinstance(self.user_denylist, (str, bytes)):
            raise ValueError("user_denylist 必须是路径集合。")

        try:
            normalized_denylist = tuple(
                Path(denylisted_path)
                for denylisted_path in self.user_denylist
            )
        except (TypeError, ValueError) as error:
            raise ValueError("user_denylist 必须是路径集合。") from error

        object.__setattr__(self, "user_denylist", normalized_denylist)


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


def validate_workspace_root(
    root_path: str | Path,
    *,
    sensitive_path_rules: SensitivePathRules = DEFAULT_SENSITIVE_PATH_RULES,
) -> Path:
    """验证工作区根路径安全可用，并返回规范化路径。"""

    normalized_root = normalize_workspace_root(root_path)

    if _is_sensitive_path(normalized_root, sensitive_path_rules):
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

    if _is_sensitive_path(resolved_path, request.sensitive_path_rules):
        raise PathPolicyError(
            PathPolicyErrorCode.SENSITIVE_PATH,
            "请求路径属于受保护的敏感文件或目录。",
        )

    if _is_user_denylisted(
        normalized_path,
        resolved_path,
        workspace_root,
        request.user_denylist,
    ):
        raise PathPolicyError(
            PathPolicyErrorCode.PATH_DENYLISTED,
            "请求路径命中用户拒绝列表。",
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


def _is_user_denylisted(
    normalized_path: Path,
    resolved_path: Path,
    workspace_root: Path,
    user_denylist: tuple[Path, ...],
) -> bool:
    """检查词法路径和解析路径是否命中用户拒绝项及其子树。"""

    for denylisted_path in user_denylist:
        candidate = denylisted_path
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
        normalized_denylisted_path = _normalize_path(candidate)

        if _is_within(normalized_path, normalized_denylisted_path):
            return True
        if _is_within(resolved_path, normalized_denylisted_path):
            return True

    return False


def _is_sensitive_path(path: Path, rules: SensitivePathRules) -> bool:
    """对完整路径段和最终文件名应用统一的敏感项规则。"""

    normalized_parts = (part.casefold() for part in path.parts)
    if any(part in rules.directory_names for part in normalized_parts):
        return True

    file_name = path.name.casefold()
    return (
        file_name in rules.file_names
        or any(
            file_name.startswith(prefix)
            for prefix in rules.file_name_prefixes
        )
        or path.suffix.casefold() in rules.file_suffixes
    )
