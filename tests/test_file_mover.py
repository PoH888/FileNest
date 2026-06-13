"""
file_mover.py 独立测试

从源文件 ``if __name__ == "__main__":`` 块迁移而来。
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.utils import init_logging  # noqa: E402
from core.file_mover import move_file, handle_collision, is_file_ready, rollback_file, undo_file  # noqa: E402


def test_file_mover() -> None:
    """file_mover 所有自测。"""
    init_logging()

    tmp_dir = Path(tempfile.mkdtemp())

    try:
        # ---- 测试 1: 正常移动 ----
        src = tmp_dir / "test_normal.txt"
        dst_dir = tmp_dir / "subdir"
        src.write_text("hello", encoding="utf-8")
        result = move_file(src, dst_dir)
        assert result is not None, "正常移动应成功"
        assert result.exists(), "移动后文件应存在于目标目录"
        assert result.parent == dst_dir.resolve(), "应位于目标文件夹"

        # ---- 测试 2: keep_both 同名冲突 ----
        dup = dst_dir / "test_normal.txt"
        dup.write_text("second", encoding="utf-8")
        collided = handle_collision(dup, strategy="keep_both")
        assert collided is not None, "keep_both 应返回新路径"
        assert collided != dup, "新路径不应与原路径相同"
        assert "_1" in collided.stem, "应包含序号 _1"

        # ---- 测试 3: is_file_ready 就绪检测 ----
        ready_src = tmp_dir / "ready_file.txt"
        ready_src.write_text("ready", encoding="utf-8")
        assert is_file_ready(ready_src) is True, "未占用文件应就绪"
        assert is_file_ready(tmp_dir / "nonexistent.txt") is False, "不存在文件不应就绪"

        # ---- 测试 4: rollback 回退 ----
        src_rb = dst_dir / "test_normal.txt"
        rollback_result = rollback_file(src_rb, tmp_dir)
        assert rollback_result is not None, "回退应成功"
        assert rollback_result.parent == tmp_dir.resolve(), "应回到上一层目录"

        # ---- 测试 5: undo 撤销 ----
        orig = tmp_dir / "undo_test.txt"
        orig.write_text("undo", encoding="utf-8")
        moved = move_file(orig, tmp_dir / "some_folder")
        assert moved is not None
        undid = undo_file(moved, orig)
        assert undid is not None, "撤销应成功"
        assert undid.parent == orig.parent.resolve(), "应回到原目录"

        # ---- 测试 6: 回退超出根目录保护 ----
        outside = tmp_dir / "outside"
        outside.mkdir()
        f = outside / "test.txt"
        f.write_text("data", encoding="utf-8")
        rb_fail = rollback_file(f, outside)
        assert rb_fail is None, "已处于根目录应拒绝回退"

        # ---- 测试 7: 覆盖策略 ----
        base_file = tmp_dir / "collide_overwrite.txt"
        base_file.write_text("original", encoding="utf-8")
        second_file = tmp_dir / "collide_overwrite.txt"
        second_file.write_text("replacement", encoding="utf-8")
        resolved = handle_collision(base_file, strategy="overwrite")
        assert resolved is not None, "覆盖应返回路径"

    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
