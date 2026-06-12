"""
FileNest — 程序入口

初始化日志、配置、GUI 主窗口，组装所有模块。
"""

import logging
import sys
import traceback

# ---- 强制 UTF-8 输出（Python 3.7+），解决 Windows 中文乱码/卡死 ----
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import config_manager
from gui import FileNestApp
from utils import init_logging


def main() -> None:
    """应用程序主入口。"""
    # 1) 初始化日志
    logger, _ = init_logging()
    logger.info("FileNest 启动。")

    # 2) 初始化配置
    config = config_manager.load_config()
    logger.info("配置已加载，根目录: %s", config.get("root_directories"))

    # 3) 启动 GUI
    app = FileNestApp()
    app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger = logging.getLogger("FileNest")
        logger.info("用户通过键盘中断退出。")
        sys.exit(0)
    except Exception as e:
        logger = logging.getLogger("FileNest")
        logger.critical("程序异常退出: %s\n%s", e, traceback.format_exc())
        try:
            import tkinter.messagebox as mb
            mb.showerror("FileNest 错误",
                         f"程序发生未预期的错误：\n{e}\n\n请查看日志获取详细信息。")
        except Exception:
            pass
        sys.exit(1)
