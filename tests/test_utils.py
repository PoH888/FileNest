"""
utils.py 独立测试

从源文件 ``if __name__ == "__main__":`` 块迁移而来。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.utils import init_logging, normalize_path, create_tray_icon, destroy_tray_icon  # noqa: E402


def test_init_logging() -> None:
    """测试日志初始化。"""
    logger, log_file = init_logging()
    logger.info("utils.py 自测：日志初始化成功。")
    # log_file 可能是 str 或 Path，取决于实现
    assert log_file is not None


def test_normalize_path() -> None:
    """测试路径规范化。"""
    p = normalize_path(".\\some\\path")
    assert isinstance(p, Path)


def test_tray_icon() -> None:
    """测试托盘图标导入和创建/销毁。"""
    icon = create_tray_icon(None)
    if icon is None:
        # 系统托盘依赖未安装 → 跳过
        return
    destroy_tray_icon()
    # 不报错即通过


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
