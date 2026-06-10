"""
FileNest — 工具函数模块

提供全项目共享的基础工具函数：日志初始化、路径规范化、系统托盘管理。
本模块不依赖 FileNest 其他内部模块。
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Optional, Tuple

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

APP_NAME: str = "FileNest"
"""应用程序名称"""

LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"
"""日志格式：时间 - 级别 - 消息"""

LOG_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB
"""单个日志文件最大字节数"""

LOG_BACKUP_COUNT: int = 3
"""日志轮转保留的备份文件数量"""


# ---------------------------------------------------------------------------
# 日志初始化
# ---------------------------------------------------------------------------

def init_logging(log_path: Optional[str] = None) -> Tuple[logging.Logger, str]:
    """初始化并返回全局日志记录器。

    使用 RotatingFileHandler 限制磁盘占用；若无法写入指定路径则回退到
    应用程序同级目录下的 logs/ 子目录。

    Args:
        log_path: 日志文件完整路径。为 None 时自动选择默认位置。

    Returns:
        (logger, 实际使用的日志文件路径) 的二元组。
    """
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if logger.handlers:
        return logger, logger.handlers[0].baseFilename  # type: ignore[return-value]

    # 确定日志文件路径
    if log_path:
        log_file = Path(log_path)
    else:
        log_file = _default_log_path()

    # 确保父目录存在
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        # 无法创建目录时回退
        log_file = _fallback_log_path()
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # 文件 handler —— 轮转
    file_handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    # 控制台 handler —— 便于调试
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

    return logger, str(log_file)


def _default_log_path() -> Path:
    """返回默认的日志文件路径（可执行文件同级的 logs/ 目录）。"""
    try:
        base = Path(sys.executable).parent
    except Exception:
        base = Path.cwd()
    return base / "logs" / f"{APP_NAME}.log"


def _fallback_log_path() -> Path:
    """当默认路径不可写时回退到当前工作目录。"""
    return Path.cwd() / "logs" / f"{APP_NAME}.log"


# ---------------------------------------------------------------------------
# 路径规范化
# ---------------------------------------------------------------------------

def normalize_path(path_str: str) -> Path:
    """规范化路径字符串，返回绝对路径。

    - 统一使用 pathlib.Path
    - 处理 Windows 反斜杠/正斜杠混用
    - 解析相对路径为绝对路径

    Args:
        path_str: 原始路径字符串。

    Returns:
        规范化后的绝对路径 Path 对象。
    """
    return Path(path_str).resolve()


# ---------------------------------------------------------------------------
# 系统托盘管理
# ---------------------------------------------------------------------------

_tray_icon_instance: Optional[object] = None
"""保存当前活跃的托盘图标对象，供 destroy_tray_icon 引用。"""


def create_tray_icon(
    root: object,
    exit_callback: Optional[Callable[[], None]] = None,
) -> Optional[object]:
    """创建系统托盘图标，主窗口关闭时最小化到托盘而非退出。

    使用 pystray 实现；若 pystray（或依赖的 Pillow）不可用，则静默跳过。

    Args:
        root: tkinter.Tk 主窗口实例，用于实现显示/隐藏。
        exit_callback: 退出菜单项的回调，为 None 时调用 sys.exit。

    Returns:
        托盘图标对象（可用于后续 destroy），导入失败时返回 None。
    """
    # pylint: disable=import-outside-toplevel
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError:
        logging.getLogger(APP_NAME).warning(
            "pystray 或 Pillow 不可用，系统托盘功能已跳过。"
        )
        return None

    def _on_show(icon: object, item: object) -> None:
        """显示主窗口。"""
        try:
            root.after(0, _show_window)  # type: ignore[union-attr]
        except Exception:
            pass

    def _show_window() -> None:
        """线程安全地显示 tkinter 窗口。"""
        try:
            root.deiconify()  # type: ignore[union-attr]
            root.lift()  # type: ignore[union-attr]
        except Exception:
            pass

    def _on_exit(icon: object, item: object) -> None:
        """退出程序。"""
        icon.stop()  # type: ignore[union-attr]
        if exit_callback:
            exit_callback()
        else:
            sys.exit(0)

    # 生成一个 16×16 的简易图标（纯色圆点）
    image = _create_default_icon(16, 16)

    menu = (
        pystray.MenuItem("显示主窗口", _on_show, default=True),
        pystray.MenuItem("退出", _on_exit),
    )

    icon = pystray.Icon(APP_NAME, image, APP_NAME, menu)
    global _tray_icon_instance  # noqa: PLW0603
    _tray_icon_instance = icon

    # pystray 需要独立线程运行
    import threading

    t = threading.Thread(target=icon.run, daemon=True)
    t.start()
    return icon


def _create_default_icon(width: int, height: int) -> object:
    """绘制一个纯色简易图标（PIL Image），避免依赖外部图标文件。"""
    # pylint: disable=import-outside-toplevel
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse([2, 2, width - 2, height - 2], fill=(70, 130, 180, 255))
    return image


def destroy_tray_icon() -> None:
    """停止并清理系统托盘图标。"""
    global _tray_icon_instance  # noqa: PLW0603
    if _tray_icon_instance is not None:
        try:
            _tray_icon_instance.stop()  # type: ignore[union-attr]
        except Exception:
            pass
        _tray_icon_instance = None


# ---------------------------------------------------------------------------
# 独立验证
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 简单自测：初始化日志 → 写一条消息 → 检查文件是否生成
    logger, log_file = init_logging()
    logger.info("utils.py 自测：日志初始化成功。")
    print(f"日志文件: {log_file}")

    # 路径规范化测试
    p = normalize_path(".\\some\\path")
    print(f"规范化路径: {p}")

    # 托盘图标（仅测试导入是否正常）
    icon = create_tray_icon(None)
    if icon is None:
        print("系统托盘: 已跳过（依赖未安装）")
    else:
        destroy_tray_icon()
        print("系统托盘: 创建并销毁成功")
