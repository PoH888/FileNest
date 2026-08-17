"""把文件移入独立隔离区，代替不可恢复的直接删除。"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from core.file_mover import move_file as v1_move_file

from .filesystem_adapter import FileSystemAdapter


class QuarantineErrorCode(StrEnum):
    """隔离失败时供程序稳定判断的错误码。"""

    INVALID_ROOTS = "quarantine_roots_overlap"
    INVALID_IDENTIFIER = "quarantine_identifier_invalid"
    SOURCE_UNAVAILABLE = "quarantine_source_unavailable"
    TARGET_CONFLICT = "quarantine_target_conflict"
    DIRECTORY_UNAVAILABLE = "quarantine_directory_unavailable"
    MOVE_FAILED = "quarantine_move_failed"
    RESULT_MISMATCH = "quarantine_result_mismatch"


class QuarantineError(RuntimeError):
    """文件无法在不覆盖数据的前提下进入隔离区。"""

    def __init__(
        self,
        code: QuarantineErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class QuarantinedFile:
    """一次隔离完成后的原路径与新路径。"""

    original_path: Path
    quarantine_path: Path


def build_quarantine_relative_path(
    *,
    workspace_id: int,
    plan_id: UUID,
    source_file_id: int,
    file_name: str,
) -> Path:
    """生成隔离区内稳定且不依赖工作区绝对路径的目标位置。"""

    if workspace_id < 1 or source_file_id < 1 or plan_id.int == 0:
        raise QuarantineError(
            QuarantineErrorCode.INVALID_IDENTIFIER,
            "隔离路径所需的业务标识无效",
        )
    if not file_name or Path(file_name).name != file_name:
        raise QuarantineError(
            QuarantineErrorCode.INVALID_IDENTIFIER,
            "隔离路径所需的文件名无效",
        )

    return (
        Path(f"workspace-{workspace_id}")
        / str(plan_id)
        / str(source_file_id)
        / file_name
    )


class QuarantineManager:
    """在授权工作区与独立应用隔离区之间移动文件。"""

    def __init__(
        self,
        workspace_adapter: FileSystemAdapter,
        quarantine_adapter: FileSystemAdapter,
    ) -> None:
        workspace_root = workspace_adapter.workspace_root
        quarantine_root = quarantine_adapter.workspace_root
        if (
            workspace_root == quarantine_root
            or workspace_root.is_relative_to(quarantine_root)
            or quarantine_root.is_relative_to(workspace_root)
        ):
            raise QuarantineError(
                QuarantineErrorCode.INVALID_ROOTS,
                "工作区与隔离区必须是互不包含的独立目录",
            )

        self._workspace_adapter = workspace_adapter
        self._quarantine_adapter = quarantine_adapter

    def quarantine(
        self,
        source_path: Path,
        *,
        workspace_id: int,
        plan_id: UUID,
        source_file_id: int,
    ) -> QuarantinedFile:
        """把一个普通文件移动到由计划和文件标识确定的隔离路径。"""

        if workspace_id < 1 or source_file_id < 1 or plan_id.int == 0:
            raise QuarantineError(
                QuarantineErrorCode.INVALID_IDENTIFIER,
                "隔离路径所需的业务标识无效",
            )

        authorized_source = self._workspace_adapter.authorized_path(
            source_path
        )
        try:
            source_metadata = self._workspace_adapter.get_file_metadata(
                authorized_source
            )
        except OSError as error:
            raise QuarantineError(
                QuarantineErrorCode.SOURCE_UNAVAILABLE,
                "待隔离的源文件当前不可用",
            ) from error
        if source_metadata is None:
            raise QuarantineError(
                QuarantineErrorCode.SOURCE_UNAVAILABLE,
                "待隔离的源路径不是普通文件",
            )

        quarantine_relative_path = build_quarantine_relative_path(
            workspace_id=workspace_id,
            plan_id=plan_id,
            source_file_id=source_file_id,
            file_name=authorized_source.name,
        )
        expected_target = self._quarantine_adapter.authorized_path(
            quarantine_relative_path
        )
        if self._quarantine_adapter.path_exists(expected_target):
            raise QuarantineError(
                QuarantineErrorCode.TARGET_CONFLICT,
                "隔离目标已经存在，禁止覆盖",
            )

        target_directory = expected_target.parent
        try:
            target_directory.mkdir(parents=True, exist_ok=True)
            target_directory = self._quarantine_adapter.authorized_path(
                quarantine_relative_path.parent
            )
            expected_target = self._quarantine_adapter.authorized_path(
                quarantine_relative_path
            )
        except OSError as error:
            raise QuarantineError(
                QuarantineErrorCode.DIRECTORY_UNAVAILABLE,
                "无法创建或使用隔离目录",
            ) from error

        if (
            not self._quarantine_adapter.is_directory(target_directory)
            or self._quarantine_adapter.path_exists(expected_target)
        ):
            raise QuarantineError(
                QuarantineErrorCode.TARGET_CONFLICT,
                "隔离目标在移动前被占用",
            )

        moved_path = v1_move_file(
            authorized_source,
            target_directory,
            collision_strategy="skip",
            target_name=expected_target.name,
        )
        if moved_path is None:
            if self._quarantine_adapter.path_exists(expected_target):
                raise QuarantineError(
                    QuarantineErrorCode.TARGET_CONFLICT,
                    "移动期间隔离目标被占用",
                )
            raise QuarantineError(
                QuarantineErrorCode.MOVE_FAILED,
                "V1 文件移动失败",
            )

        authorized_result = self._quarantine_adapter.authorized_path(
            moved_path
        )
        try:
            result_metadata = self._quarantine_adapter.get_file_metadata(
                authorized_result
            )
        except OSError as error:
            raise QuarantineError(
                QuarantineErrorCode.RESULT_MISMATCH,
                "隔离后的结果文件不可用",
            ) from error
        if authorized_result != expected_target or result_metadata is None:
            raise QuarantineError(
                QuarantineErrorCode.RESULT_MISMATCH,
                "隔离操作返回了非预期结果",
            )

        return QuarantinedFile(
            original_path=authorized_source,
            quarantine_path=authorized_result,
        )
