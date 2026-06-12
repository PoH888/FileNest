"""
FileNest — 配置管理模块

负责 JSON 配置文件的加载、保存、容错恢复以及默认配置的生成。
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict

from utils import init_logging

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    # ---- 分类目录 ----
    "root_directories": [],
    "max_depth": 5,
    # ---- 监控 ----
    "monitor_directory": "",
    "monitor_enabled": False,
    # ---- 匹配策略 ----
    "auto_threshold": 85,
    "mid_range_auto": False,
    "mid_range_min": 60,
    "mid_range_max": 85,
    "parent_weight_enabled": True,
    # ---- 忽略名单 ----
    "ignore_patterns": "",
    # ---- 外观与行为 ----
    "open_folder_after_move": False,
    # ---- 语言 ----
    "language": "zh",
}
"""默认配置字典，首次运行或配置文件损坏时使用。"""

# ---------------------------------------------------------------------------
# 配置路径
# ---------------------------------------------------------------------------

def _get_logger() -> logging.Logger:
    """获取 FileNest 全局日志记录器（延迟初始化）。"""
    logger, _ = init_logging()
    return logger


def _config_dir() -> Path:
    """返回配置文件所在目录（可执行文件同级）。"""
    try:
        base = Path(sys.executable).parent
    except Exception:
        base = Path.cwd()
    return base


def _config_path() -> Path:
    """返回 settings.json 的完整路径。"""
    return _config_dir() / "settings.json"


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    """加载 JSON 配置文件。

    读取程序目录下的 settings.json；若文件不存在则返回默认配置并写盘；
    若 JSON 格式损坏则备份为 ``settings.json.bak``，回退默认配置。

    Returns:
        配置字典。
    """
    logger = _get_logger()
    path = _config_path()

    if not path.exists():
        logger.info("配置文件不存在，使用默认配置。")
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)

    try:
        with open(str(path), "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
    except json.JSONDecodeError:
        logger.warning("配置文件损坏，正在备份并恢复默认配置。")
        _backup_corrupted(path, logger)
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    except (OSError, PermissionError) as e:
        logger.error("读取配置文件失败: %s", e)
        return dict(DEFAULT_CONFIG)

    # 合并缺失的默认字段（配置可能来自旧版本）
    merged = dict(DEFAULT_CONFIG)
    merged.update(data)
    return merged


def _backup_corrupted(path: Path, logger: logging.Logger) -> None:
    """将损坏的配置文件重命名为 ``.bak``。

    Args:
        path: 配置文件路径。
        logger: 日志记录器。
    """
    bak_path = path.with_suffix(".json.bak")
    try:
        os.replace(str(path), str(bak_path))
        logger.info("已备份损坏配置文件至: %s", bak_path)
    except OSError as e:
        logger.error("备份损坏配置文件失败: %s", e)


# ---------------------------------------------------------------------------
# 保存（原子写入）
# ---------------------------------------------------------------------------

def save_config(config: Dict[str, Any]) -> bool:
    """原子写入配置文件。

    将配置先写入 ``settings.json.tmp``，再通过 ``os.replace``
    重命名为正式文件，防止写入中断导致文件损坏。

    Args:
        config: 要保存的配置字典。

    Returns:
        写入成功返回 ``True``，否则返回 ``False``。
    """
    logger = _get_logger()
    path = _config_path()
    tmp_path = path.with_suffix(".json.tmp")

    try:
        content = json.dumps(config, ensure_ascii=False, indent=2)
        with open(str(tmp_path), "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(str(tmp_path), str(path))
        logger.info("配置已保存至: %s", path)
        return True
    except (OSError, PermissionError) as e:
        logger.error("保存配置文件失败: %s", e)
        # 清理残留临时文件
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


# ---------------------------------------------------------------------------
# 独立验证
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    import tempfile

    logger, _ = init_logging()
    logger.info("config_manager.py 自测开始。")

    # ---- 临时重定向配置路径到临时目录 ----
    _orig_config_path = _config_path
    _tmp_dir_obj = Path(tempfile.mkdtemp())

    def _test_config_path() -> Path:
        return _tmp_dir_obj / "settings.json"

    # 替换模块级函数引用供本次自测使用
    import config_manager  # noqa: F811
    config_manager._config_path = _test_config_path  # type: ignore[assignment]

    # 重新绑定当前模块中的函数（使 load_config/save_config 也走测试路径）
    # 直接修改全局模块的 _config_path 引用
    import sys
    this_module = sys.modules[__name__]
    this_module._config_path = _test_config_path  # type: ignore[assignment]

    # ---- 测试 1：加载默认配置 ----
    cfg = load_config()
    assert cfg == DEFAULT_CONFIG, "默认配置不匹配"
    print("[OK] 测试 1: 加载/生成默认配置 -- 通过")

    # ---- 测试 2：保存并重新加载 ----
    cfg["root_directories"] = [r"D:\Test"]
    ok = save_config(cfg)
    assert ok, "保存配置失败"
    reloaded = load_config()
    assert reloaded["root_directories"] == [r"D:\Test"], "重新加载后配置不一致"
    print("[OK] 测试 2: 保存并重新加载配置 -- 通过")

    # ---- 测试 3：损坏文件容错 ----
    path = _test_config_path()
    with open(str(path), "w", encoding="utf-8") as f:
        f.write("{invalid json!!!}")
    corrupted = load_config()
    assert corrupted == DEFAULT_CONFIG, "损坏后应回退默认配置"
    bak = path.with_suffix(".json.bak")
    assert bak.exists(), "损坏文件应被备份为 .bak"
    print("[OK] 测试 3: 损坏 JSON 容错备份 -- 通过")

    # ---- 测试 4：原子写入完整性 ----
    cfg["auto_threshold"] = 90
    save_config(cfg)
    tmp = path.with_suffix(".json.tmp")
    assert not tmp.exists(), "原子写入后临时文件应被清理"
    print("[OK] 测试 4: 原子写入(临时文件清理) -- 通过")

    # 清理临时目录
    shutil.rmtree(_tmp_dir_obj)
    # 恢复原始函数
    this_module._config_path = _orig_config_path

    print("\nconfig_manager.py 全部自测通过.")
