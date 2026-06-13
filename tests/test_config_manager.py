"""
config_manager.py 独立测试

从源文件 ``if __name__ == "__main__":`` 块迁移而来。
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.utils import init_logging  # noqa: E402
from core import config_manager  # noqa: E402
from core.config_manager import DEFAULT_CONFIG, load_config, save_config  # noqa: E402


def test_config_manager(tmp_path_factory) -> None:
    """config_manager 所有自测（共享临时目录）。"""
    init_logging()

    tmp_dir = Path(tempfile.mkdtemp())

    def _test_config_path() -> Path:
        return tmp_dir / "settings.json"

    # 替换模块级函数引用
    orig = config_manager._config_path
    config_manager._config_path = _test_config_path  # type: ignore[assignment]

    # 也修改 sys.modules 中的引用（因为 save_config/load_config 通过 `from .utils import` 使用了模块层的 _config_path）
    mod = sys.modules["core.config_manager"]
    mod._config_path = _test_config_path  # type: ignore[assignment]

    try:
        # ---- 测试 1：加载默认配置 ----
        cfg = load_config()
        assert cfg == DEFAULT_CONFIG, "默认配置不匹配"

        # ---- 测试 2：保存并重新加载 ----
        cfg["root_directories"] = [r"D:\Test"]
        ok = save_config(cfg)
        assert ok, "保存配置失败"
        reloaded = load_config()
        assert reloaded["root_directories"] == [r"D:\Test"], "重新加载后配置不一致"

        # ---- 测试 3：损坏文件容错 ----
        path = _test_config_path()
        with open(str(path), "w", encoding="utf-8") as f:
            f.write("{invalid json!!!}")
        corrupted = load_config()
        assert corrupted == DEFAULT_CONFIG, "损坏后应回退默认配置"
        bak = path.with_suffix(".json.bak")
        assert bak.exists(), "损坏文件应被备份为 .bak"

        # ---- 测试 4：原子写入完整性 ----
        cfg["auto_threshold"] = 90
        save_config(cfg)
        tmp = path.with_suffix(".json.tmp")
        assert not tmp.exists(), "原子写入后临时文件应被清理"
    finally:
        shutil.rmtree(tmp_dir)
        config_manager._config_path = orig  # type: ignore[assignment]
        mod._config_path = orig  # type: ignore[assignment]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
