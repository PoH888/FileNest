"""
file_monitor.py 独立测试

从源文件 ``if __name__ == "__main__":`` 块迁移而来。
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path
from queue import Queue

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.utils import init_logging  # noqa: E402
from core.file_monitor import start_monitor, stop_monitor, drain_queue  # noqa: E402


def test_file_monitor() -> None:
    """file_monitor 所有自测。"""
    init_logging()

    tmp_dir = Path(tempfile.mkdtemp())
    event_queue: Queue = Queue()

    try:
        # ---- 测试 1: 启动监控 ----
        observer = start_monitor(tmp_dir, event_queue)
        assert observer is not None, "监控应成功启动"
        assert observer.is_alive(), "observer 线程应运行中"

        # ---- 测试 2: 文件创建触发事件 ----
        files_created = []
        for i in range(3):
            f = tmp_dir / f"test_file_{i}.txt"
            f.write_text(f"content_{i}", encoding="utf-8")
            files_created.append(f)
            time.sleep(1.1)  # 略大于防抖间隔

        time.sleep(2)
        events = drain_queue(event_queue)
        paths_received = {e.file_path for e in events}
        assert len(events) >= 3, f"应至少收到 3 个事件，实际 {len(events)}"
        for f in files_created:
            assert f.resolve() in paths_received, f"未收到 {f} 的事件"

        # ---- 测试 3: 防抖机制 ----
        drain_queue(event_queue)  # 清空
        dup = tmp_dir / "debounce_test.txt"
        for _ in range(5):
            dup.write_text("data", encoding="utf-8")
            time.sleep(0.1)
        time.sleep(1)
        after_events = drain_queue(event_queue)
        assert len(after_events) <= 2, f"防抖失效：收到 {len(after_events)} 个事件"

        # ---- 测试 4: 目录事件过滤 ----
        sub = tmp_dir / "sub_folder"
        sub.mkdir()
        time.sleep(1)
        dir_events = drain_queue(event_queue)
        dir_paths = {e.file_path for e in dir_events}
        assert sub.resolve() not in dir_paths, "目录事件应被过滤"

        # ---- 测试 5: 停止监控 ----
        stop_monitor(observer)
        assert not observer.is_alive(), "observer 应已停止"

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
