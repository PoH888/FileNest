"""将 V1 目录扫描结果包装成可持久化的 Workspace 文件清单。"""

import logging
from dataclasses import dataclass
from pathlib import Path

from core.folder_scanner import scan_directory

from .filesystem_adapter import FileSystemAdapter
from .path_policy import PathPolicyError


@dataclass(frozen=True, slots=True)
class ScannedFile:
    """一次安全扫描观察到的文件元数据。"""

    relative_path: str
    name: str
    extension: str
    size_bytes: int
    mtime_ns: int


def scan_workspace_files(
    workspace_root: Path,
    max_depth: int = 5,
    ignore_patterns: list[str] | None = None,
) -> list[ScannedFile]:
    """通过 FileSystem Adapter 扫描工作区中的普通文件。"""

    adapter = FileSystemAdapter(workspace_root)
    logger = logging.getLogger("FileNest")

    try:
        if not adapter.is_directory(Path(".")):
            return []
        authorized_root = adapter.workspace_root
    except (OSError, PathPolicyError) as error:
        logger.warning("无法扫描工作区 %s: %s", workspace_root, error)
        return []

    folders = scan_directory(
        authorized_root,
        max_depth=max_depth,
        ignore_patterns=ignore_patterns,
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
        try:
            child_names = adapter.list_directory(directory)
        except (OSError, PathPolicyError) as error:
            logger.warning("无法读取工作区目录 %s: %s", directory, error)
            continue

        for name in child_names:
            relative_path = directory / name

            try:
                metadata = adapter.get_file_metadata(relative_path)
            except (OSError, PathPolicyError) as error:
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
