"""
FileNest — 文件监控模块

负责使用 watchdog 监听指定目录的新文件事件，
通过队列安全地将事件传递给主线程，不直接操作任何 UI。
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Dict, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer



def _get_logger() -> logging.Logger:
    """延迟获取全局日志记录器。"""
    return logging.getLogger("FileNest")


# ---------------------------------------------------------------------------
# 事件数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FileEvent:
    """文件系统事件的数据封装。

    Attributes:
        file_path: 事件关联的文件路径。
        timestamp: 事件发生的时间戳（time.monotonic）。
    """
    file_path: Path
    timestamp: float


# ---------------------------------------------------------------------------
# 事件处理器
# ---------------------------------------------------------------------------

_DEBOUNCE_SECONDS: float = 3.0
"""同一文件的防抖间隔（秒）。"""


class FileEventHandler(FileSystemEventHandler):
    """watchdog 事件处理器，仅关注文件创建事件。

    将事件通过队列传递（而非直接操作），保证线程安全。
    """

    def __init__(self, queue: Queue) -> None:
        """初始化事件处理器。

        Args:
            queue: 主线程用于接收事件的 queue.Queue。
        """
        super().__init__()
        self._queue: Queue = queue
        self._debounce_map: Dict[Path, float] = {}
        self._debounce_lock: threading.Lock = threading.Lock()

    def on_created(self, event: FileSystemEvent) -> None:
        """处理文件创建事件。"""
        self._handle_event(event, "created")

    def on_modified(self, event: FileSystemEvent) -> None:
        """处理文件修改事件。"""
        self._handle_event(event, "modified")

    def _handle_event(self, event: FileSystemEvent, event_type: str) -> None:
        """统一处理文件系统事件的内部方法。

        Args:
            event: watchdog 原始事件。
            event_type: 事件类型描述（仅用于日志）。
        """
        logger = _get_logger()

        try:
            # 跳过目录事件
            if event.is_directory:
                return

            path = Path(event.src_path)

            # 文件可能已被删除，跳过
            if not path.exists():
                return

            # 防抖：同一文件在 _DEBOUNCE_SECONDS 内不重复入队
            now = time.monotonic()
            with self._debounce_lock:
                last_push = self._debounce_map.get(path, 0.0)
                if now - last_push < _DEBOUNCE_SECONDS:
                    logger.debug("防抖跳过: %s", path)
                    return
                self._debounce_map[path] = now

            event_data = FileEvent(file_path=path.resolve(), timestamp=now)
            self._queue.put(event_data)
            logger.info("事件入队 [%s]: %s", event_type, path)

        except Exception as e:
            logger.error("处理文件事件异常 [%s] %s: %s", event_type, event.src_path, e)


# ---------------------------------------------------------------------------
# 监控启停
# ---------------------------------------------------------------------------

def start_monitor(directory: Path, queue: Queue) -> Optional[Observer]:
    """启动 watchdog 文件监控。

    使用 ``recursive=False`` 只监控指定目录的直接子文件。

    Args:
        directory: 要监控的文件夹路径。
        queue: 用于接收事件的 queue.Queue。

    Returns:
        Observer 实例（用于后续停止）；目录无效时返回 ``None``。
    """
    logger = _get_logger()

    try:
        directory = directory.resolve(strict=False)
    except OSError as e:
        logger.error("无法解析监控目录 %s: %s", directory, e)
        return None

    if not directory.is_dir():
        logger.error("监控目录不存在: %s", directory)
        return None

    event_handler = FileEventHandler(queue)
    observer = Observer()
    observer.schedule(event_handler, str(directory), recursive=False)
    observer.start()

    logger.info("文件监控已启动: %s", directory)
    return observer


def stop_monitor(observer: Observer) -> None:
    """优雅停止 watchdog 文件监控。

    Args:
        observer: 由 ``start_monitor`` 返回的 Observer 实例。
    """
    logger = _get_logger()

    if observer is None:
        return

    try:
        observer.stop()
        observer.join(timeout=5)
        logger.info("文件监控已停止")
    except Exception as e:
        logger.error("停止监控异常: %s", e)


# ---------------------------------------------------------------------------
# 队列轮询（供主线程使用）
# ---------------------------------------------------------------------------

def drain_queue(queue: Queue) -> list[FileEvent]:
    """清空事件队列，返回所有待处理的事件列表。

    此函数由主线程（tkinter）定期调用（通过 ``root.after``），
    安全地从队列取出事件并处理。

    Args:
        queue: 事件队列。

    Returns:
        当前队列中所有 FileEvent 的列表（按入队顺序）。
    """
    events: list[FileEvent] = []
    while not queue.empty():
        try:
            events.append(queue.get_nowait())
        except Exception:
            break
    return events

