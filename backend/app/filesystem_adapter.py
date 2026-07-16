"""所有正式文件读取都必须经过的安全文件系统边界。"""

from pathlib import Path

from .path_policy import (
    AuthorizedPath,
    PathPolicyError,
    PathPolicyRequest,
    authorize_path,
)


class FileSystemAdapter:
    """只允许在一个授权工作区内读取文件和列出目录。"""

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root

    @property
    def workspace_root(self) -> Path:
        """返回经过 Path Policy 解析的工作区根路径。"""

        return self._authorize(Path(".")).workspace_root

    def authorized_path(self, requested_path: Path) -> Path:
        """返回经过完整策略校验的规范路径。"""

        return self._authorize(requested_path).path

    def is_directory(self, requested_path: Path) -> bool:
        """授权路径后判断它是否为目录。"""

        return self._authorize(requested_path).path.is_dir()

    def read_text(
        self,
        requested_path: Path,
        encoding: str = "utf-8",
    ) -> str:
        """授权路径后读取文本文件。"""

        authorized_path = self._authorize(requested_path)
        return authorized_path.path.read_text(encoding=encoding)

    def list_directory(self, requested_path: Path = Path(".")) -> list[str]:
        """列出目录中经过 Path Policy 授权的直接子项名称。"""

        authorized_directory = self._authorize(requested_path)
        visible_names: list[str] = []

        for child_path in authorized_directory.path.iterdir():
            try:
                authorize_path(
                    PathPolicyRequest(
                        workspace_root=authorized_directory.workspace_root,
                        requested_path=child_path,
                    )
                )
            except PathPolicyError:
                continue

            visible_names.append(child_path.name)

        return sorted(visible_names, key=str.casefold)

    def _authorize(self, requested_path: Path) -> AuthorizedPath:
        """确保 Adapter 的每个文件系统操作使用同一策略入口。"""

        return authorize_path(
            PathPolicyRequest(
                workspace_root=self._workspace_root,
                requested_path=requested_path,
            )
        )
