"""
folder_scanner.py 独立测试

从源文件 ``if __name__ == "__main__":`` 块迁移而来。
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.utils import init_logging  # noqa: E402
from core.folder_scanner import scan_directory  # noqa: E402


def _makedirs(root: Path, rel: str) -> Path:
    p = root / rel
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_folder_scanner() -> None:
    """folder_scanner 所有自测。"""
    init_logging()

    tmp_root = Path(tempfile.mkdtemp())

    _makedirs(tmp_root, "Music/Rock")
    _makedirs(tmp_root, "Music/Jazz")
    _makedirs(tmp_root, "Music/Classical")
    _makedirs(tmp_root, "Documents/Work/Reports")
    _makedirs(tmp_root, "Documents/Personal")
    _makedirs(tmp_root, ".hidden_dir/sub")
    _makedirs(tmp_root, ".git/objects")
    _makedirs(tmp_root, "backup.tmp/cache")

    try:
        # ---- 测试 1: 默认扫描 (depth=5, 无忽略) ----
        folders = scan_directory(tmp_root, max_depth=5, ignore_patterns=[])
        names = {p.name for p in folders}
        assert "Music" in names, "应包含 Music"
        assert "Documents" in names, "应包含 Documents"
        assert "Work" in names, "应包含 Work"
        assert "Reports" in names, "应包含 Reports"
        assert "Personal" in names, "应包含 Personal"
        assert ".hidden_dir" not in names, "应跳过隐藏目录"
        assert ".git" not in names, "应跳过 .git"
        assert "backup.tmp" in names, "无忽略模式时应包含 backup.tmp"

        # ---- 测试 2: max_depth 限制 ----
        folders_d1 = scan_directory(tmp_root, max_depth=1, ignore_patterns=[])
        names_d1 = {p.name for p in folders_d1}
        assert "Music" in names_d1
        assert "Documents" in names_d1
        assert "Work" not in names_d1, "depth=1 不应扫描到 Work"
        assert "Reports" not in names_d1, "depth=1 不应扫描到 Reports"

        # ---- 测试 3: ignore_patterns 过滤 ----
        ignore = [".tmp", "__pycache__"]
        folders_ig = scan_directory(tmp_root, max_depth=5, ignore_patterns=ignore)
        names_ig = {p.name for p in folders_ig}
        assert "backup.tmp" not in names_ig, "应忽略 .tmp 目录"
        assert "Music" in names_ig, "正常目录应保留"

        # ---- 测试 4: 空目录 / 不存在路径 ----
        empty_result = scan_directory(tmp_root / "NotExists")
        assert empty_result == [], "不存在路径应返回空列表"
    finally:
        shutil.rmtree(tmp_root)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
