"""受 PathPolicy 约束的 V2 文件移动执行器。"""

from enum import StrEnum
from pathlib import Path

from .filesystem_adapter import FileSystemAdapter
from .v2_file_mover import move_file as v2_move_file


class SafeFileMoveErrorCode(StrEnum):
    """安全移动失败时供程序稳定判断的错误码。"""

    SOURCE_UNAVAILABLE = "safe_move_source_unavailable"
    TARGET_DIRECTORY_UNAVAILABLE = "safe_move_target_directory_unavailable"
    TARGET_CONFLICT = "safe_move_target_conflict"
    MOVE_FAILED = "safe_move_failed"
    RESULT_MISMATCH = "safe_move_result_mismatch"


class SafeFileMoveError(RuntimeError):
    """V2 移动原语无法满足安全移动约束。"""

    def __init__(
        self,
        code: SafeFileMoveErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


class SafeFileMover:
    """只在一个已授权工作区内调用 V2 专用移动原语。"""

    def __init__(self, adapter: FileSystemAdapter) -> None:
        self._adapter = adapter

    def move(
        self,
        source_path: Path,
        target_path: Path,
    ) -> Path:
        """把普通文件移动到确定目标路径，并禁止覆盖或自动改名。"""

        authorized_source = self._adapter.authorized_write_path(source_path)
        try:
            source_metadata = self._adapter.get_file_metadata(
                authorized_source
            )
        except OSError as error:
            raise SafeFileMoveError(
                SafeFileMoveErrorCode.SOURCE_UNAVAILABLE,
                "待移动的源文件当前不可用",
            ) from error
        if source_metadata is None:
            raise SafeFileMoveError(
                SafeFileMoveErrorCode.SOURCE_UNAVAILABLE,
                "待移动的源路径不是普通文件",
            )

        expected_target = self._adapter.authorized_write_path(target_path)
        authorized_target_directory = expected_target.parent
        if not self._adapter.is_directory(authorized_target_directory):
            raise SafeFileMoveError(
                SafeFileMoveErrorCode.TARGET_DIRECTORY_UNAVAILABLE,
                "目标目录不存在或不是目录",
            )

        if self._adapter.path_exists(expected_target):
            raise SafeFileMoveError(
                SafeFileMoveErrorCode.TARGET_CONFLICT,
                "目标路径已经被占用",
            )

        moved_path = v2_move_file(
            authorized_source,
            expected_target,
        )
        if moved_path is None:
            if self._adapter.path_exists(expected_target):
                raise SafeFileMoveError(
                    SafeFileMoveErrorCode.TARGET_CONFLICT,
                    "移动期间目标路径被其他文件占用",
                )
            raise SafeFileMoveError(
                SafeFileMoveErrorCode.MOVE_FAILED,
                "V2 文件移动失败",
            )

        authorized_result = self._adapter.authorized_path(moved_path)
        try:
            result_metadata = self._adapter.get_file_metadata(
                authorized_result
            )
        except OSError as error:
            raise SafeFileMoveError(
                SafeFileMoveErrorCode.RESULT_MISMATCH,
                "V2 移动器返回的结果文件不可用",
            ) from error
        if authorized_result != expected_target or result_metadata is None:
            raise SafeFileMoveError(
                SafeFileMoveErrorCode.RESULT_MISMATCH,
                "V2 移动器返回了非预期结果",
            )

        return authorized_result
