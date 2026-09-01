"""将 V1 目录扫描结果包装成可持久化的 Workspace 文件清单。"""

import logging
from dataclasses import dataclass
from pathlib import Path

from core.folder_scanner import scan_directory

from .filesystem_adapter import FileSystemAdapter
from .path_policy import (
    DEFAULT_GLOBAL_IGNORE_POLICY,
    GlobalIgnorePolicy,
    PathPolicyError,
    WorkspacePolicy,
)


@dataclass(frozen=True, slots=True)
class ScannedFile:
    """一次安全扫描观察到的文件元数据。"""

    relative_path: str
    name: str
    extension: str
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class IgnoredEntry:
    """扫描期间被排除的工作区相对路径及稳定原因。"""

    relative_path: str
    ignored_reason: str


def scan_workspace_files(
    workspace_root: Path,
    max_depth: int = 5,
    ignore_patterns: list[str] | None = None,
    *,
    global_ignore_policy: GlobalIgnorePolicy = DEFAULT_GLOBAL_IGNORE_POLICY,
    ignored_entries: list[IgnoredEntry] | None = None,
    workspace_policy: WorkspacePolicy = WorkspacePolicy(),
) -> list[ScannedFile]:
    """通过 FileSystem Adapter 扫描工作区中的普通文件。"""

    adapter = FileSystemAdapter(
        workspace_root,
        workspace_policy=workspace_policy,
    )
    logger = logging.getLogger("FileNest")

    try:
        if not adapter.is_directory(Path(".")):
            return []
        authorized_root = adapter.workspace_root
    except (OSError, PathPolicyError) as error:
        logger.warning("无法扫描工作区 %s: %s", workspace_root, error)
        return []

    recorded_ignored: set[tuple[str, str]] = set()

    def record_policy_ignored(
        ignored_path: Path,
        error: PathPolicyError,
    ) -> None:
        _append_ignored_entry(
            ignored_entries,
            recorded_ignored,
            authorized_root,
            ignored_path,
            error.code.value,
        )

    effective_ignore_patterns = set(global_ignore_policy.patterns)
    effective_ignore_patterns.update(ignore_patterns or ())
    effective_ignore_patterns.update(workspace_policy.ignore_patterns)
    effective_ignore_policy = GlobalIgnorePolicy(
        frozenset(effective_ignore_patterns)
    )

    folders = scan_directory(
        authorized_root,
        max_depth=max_depth,
        ignore_patterns=sorted(effective_ignore_patterns),
    )
    relative_directories = {Path(".")}

    for folder in folders:
        try:
            relative_directories.add(folder.relative_to(authorized_root))
        except ValueError:
            logger.warning("扫描器返回工作区之外的目录，已跳过: %s", folder)

    scanned_files: list[ScannedFile] = []

    for directory in sorted(
        relative_directories,
        key=lambda path: path.as_posix().casefold(),
    ):
        if effective_ignore_policy.matches(directory):
            _append_ignored_entry(
                ignored_entries,
                recorded_ignored,
                authorized_root,
                directory,
                _ignore_reason(
                    directory,
                    workspace_policy=workspace_policy,
                ),
            )
            continue

        try:
            child_names = adapter.list_directory(
                directory,
                on_ignored=record_policy_ignored,
            )
        except (OSError, PathPolicyError) as error:
            _append_ignored_entry(
                ignored_entries,
                recorded_ignored,
                authorized_root,
                directory,
                _exclusion_reason(error, "directory_access_error"),
            )
            logger.warning("无法读取工作区目录 %s: %s", directory, error)
            continue

        for name in child_names:
            relative_path = directory / name
            if effective_ignore_policy.matches(relative_path):
                _append_ignored_entry(
                    ignored_entries,
                    recorded_ignored,
                    authorized_root,
                    relative_path,
                    _ignore_reason(
                        relative_path,
                        workspace_policy=workspace_policy,
                    ),
                )
                continue

            try:
                metadata = adapter.get_file_metadata(relative_path)
            except (OSError, PathPolicyError) as error:
                _append_ignored_entry(
                    ignored_entries,
                    recorded_ignored,
                    authorized_root,
                    relative_path,
                    _exclusion_reason(error, "file_metadata_error"),
                )
                logger.warning("无法读取文件元数据 %s: %s", relative_path, error)
                continue

            if metadata is None:
                continue

            scanned_files.append(
                ScannedFile(
                    relative_path=relative_path.as_posix(),
                    name=name,
                    extension=relative_path.suffix.casefold(),
                    size_bytes=metadata.size_bytes,
                    mtime_ns=metadata.mtime_ns,
                )
            )

    return sorted(
        scanned_files,
        key=lambda file: file.relative_path.casefold(),
    )


def _append_ignored_entry(
    ignored_entries: list[IgnoredEntry] | None,
    recorded_ignored: set[tuple[str, str]],
    workspace_root: Path,
    ignored_path: Path,
    ignored_reason: str,
) -> None:
    """以工作区相对路径记录一次排除，避免同一路径重复记录。"""

    if ignored_entries is None:
        return

    relative_path = ignored_path
    if ignored_path.is_absolute():
        try:
            relative_path = ignored_path.relative_to(workspace_root)
        except ValueError:
            return

    relative_path_text = relative_path.as_posix()
    key = (relative_path_text, ignored_reason)
    if key in recorded_ignored:
        return

    recorded_ignored.add(key)
    ignored_entries.append(
        IgnoredEntry(
            relative_path=relative_path_text,
            ignored_reason=ignored_reason,
        )
    )


def _exclusion_reason(error: OSError | PathPolicyError, fallback: str) -> str:
    """将扫描异常映射为稳定的排除原因。"""

    if isinstance(error, PathPolicyError):
        return error.code.value
    return fallback


def _ignore_reason(
    relative_path: Path,
    *,
    workspace_policy: WorkspacePolicy,
) -> str:
    """区分 Workspace Policy 排除与全局/调用方排除原因。"""

    if workspace_policy.ignores(relative_path):
        return "workspace_ignore"
    return "global_ignore"
