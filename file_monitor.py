"""
FileNest — 文件监控模块

负责使用 watchdog 监听指定目录的新文件事件，
通过队列安全地将事件传递给主线程，不直接操作任何 UI。
"""

import logging
import os
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


# ---------------------------------------------------------------------------
# 独立验证
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    import tempfile

    from utils import init_logging
    init_logging()
    log = _get_logger()
    log.info("file_monitor.py 自测开始。")

    tmp_dir = Path(tempfile.mkdtemp())
    print(f"测试目录: {tmp_dir}")

    event_queue: Queue = Queue()

    try:
        # ---- 测试 1: 启动监控 ----
        print("--- 测试 1: 启动监控 ---")
        observer = start_monitor(tmp_dir, event_queue)
        assert observer is not None, "监控应成功启动"
        assert observer.is_alive(), "observer 线程应运行中"
        print("  [OK] 测试 1 通过\n")

        # ---- 测试 2: 文件创建触发事件 ----
        print("--- 测试 2: 文件创建触发事件 ---")
        files_created = []
        for i in range(3):
            f = tmp_dir / f"test_file_{i}.txt"
            f.write_text(f"content_{i}", encoding="utf-8")
            files_created.append(f)
            time.sleep(1.1)  # 略大于防抖间隔

        # 等待事件传递
        time.sleep(2)

        events = drain_queue(event_queue)
        paths_received = {e.file_path for e in events}
        print(f"  收到 {len(events)} 个事件: {paths_received}")

        assert len(events) >= 3, f"应至少收到 3 个事件，实际 {len(events)}"
        for f in files_created:
            assert f.resolve() in paths_received, f"未收到 {f} 的事件"
        print("  [OK] 测试 2 通过\n")

        # ---- 测试 3: 防抖机制 ----
        print("--- 测试 3: 防抖机制 ---")
        before_count = len(drain_queue(event_queue))
        # 快速写入同一文件多次
        dup = tmp_dir / "debounce_test.txt"
        for _ in range(5):
            dup.write_text("data", encoding="utf-8")
            time.sleep(0.1)
        time.sleep(1)

        after_events = drain_queue(event_queue)
        # 防抖应让快速重复写入只生成少量事件（最好 1 个）
        print(f"  防抖后收到 {len(after_events)} 个事件（5 次快速写入）")
        assert len(after_events) <= 2, f"防抖失效：收到 {len(after_events)} 个事件"
        print("  [OK] 测试 3 通过\n")

        # ---- 测试 4: 目录事件过滤 ----
        print("--- 测试 4: 目录事件过滤 ---")
        sub = tmp_dir / "sub_folder"
        sub.mkdir()
        time.sleep(1)
        dir_events = drain_queue(event_queue)
        dir_paths = {e.file_path for e in dir_events}
        assert sub.resolve() not in dir_paths, "目录事件应被过滤"
        print("  [OK] 测试 4 通过\n")

        # ---- 测试 5: 停止监控 ----
        print("--- 测试 5: 停止监控 ---")
        stop_monitor(observer)
        assert not observer.is_alive(), "observer 应已停止"
        print("  [OK] 测试 5 通过\n")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("[OK] file_monitor.py 全部自测通过.")
