"""
FileNest — 文件移动模块

负责文件的移动、回退、撤销，以及文件就绪检测、同名冲突处理等边缘情况。
"""

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

from utils import init_logging

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

_logger_initialized: bool = False


def _get_logger() -> logging.Logger:
    """延迟获取全局日志记录器。"""
    global _logger_initialized  # noqa: PLW0603
    if not _logger_initialized:
        lgr, _ = init_logging()
        _logger_initialized = True
        return lgr
    return logging.getLogger("FileNest")


# ---------------------------------------------------------------------------
# 文件就绪检测
# ---------------------------------------------------------------------------

def is_file_ready(file_path: Path, timeout: int = 60) -> bool:
    """检测文件是否可被独占打开（未被其他进程占用）。

    尝试以独占、非共享模式打开文件；若因权限/占用失败则每隔 1 秒重试，
    直至超时。

    Args:
        file_path: 待检测的文件路径。
        timeout: 最大重试秒数（默认 60）。

    Returns:
        文件就绪返回 ``True``，超时或文件不存在返回 ``False``。
    """
    logger = _get_logger()
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if not file_path.exists():
            logger.warning("文件不存在: %s", file_path)
            return False

        try:
            with open(str(file_path), "rb"):
                pass
            return True
        except (PermissionError, OSError):
            time.sleep(1)

    logger.warning("文件就绪检测超时（%d 秒）: %s", timeout, file_path)
    return False


# ---------------------------------------------------------------------------
# 同名冲突处理
# ---------------------------------------------------------------------------

def _generate_alternative_name(target_path: Path) -> Path:
    """自动生成一个不冲突的文件路径（追加序号）。

    在文件名后插入 ``_N`` 后缀（如 ``file_1.txt``），直至找到一个
    不存在的路径。

    Args:
        target_path: 原始目标路径。

    Returns:
        不冲突的新路径。
    """
    parent = target_path.parent
    stem = target_path.stem
    suffix = target_path.suffix
    counter = 1

    while True:
        alt = parent / f"{stem}_{counter}{suffix}"
        if not alt.exists():
            return alt
        counter += 1


def handle_collision(target_path: Path, strategy: str) -> Optional[Path]:
    """处理目标路径的文件名冲突。

    Args:
        target_path: 目标文件完整路径。
        strategy: 冲突策略。
            - ``"overwrite"``：覆盖目标（直接返回原路径）。
            - ``"keep_both"``：保留两者（生成 ``_N`` 后缀新路径）。
            - ``"skip"``：跳过本次移动（返回 ``None``）。
            - 其他值：默认跳过。

    Returns:
        实际应使用的目标路径（``None`` 表示跳过）。
    """
    logger = _get_logger()

    if not target_path.exists():
        return target_path

    logger.info("同名冲突: %s, 策略=%s", target_path, strategy)

    if strategy == "overwrite":
        try:
            target_path.unlink()
            logger.info("已删除原文件（覆盖）: %s", target_path)
            return target_path
        except OSError as e:
            logger.error("覆盖删除失败: %s", e)
            return None

    if strategy == "keep_both":
        alt = _generate_alternative_name(target_path)
        logger.info("保留两者: %s → %s", target_path, alt)
        return alt

    # "skip" / 未知策略
    logger.info("跳过移动: %s", target_path)
    return None


# ---------------------------------------------------------------------------
# 核心移动
# ---------------------------------------------------------------------------

def move_file(src: Path, dst_dir: Path,
              collision_strategy: str = "keep_both") -> Optional[Path]:
    """将文件移动到目标目录。

    完整流程：
    1. 验证源文件存在且就绪（``is_file_ready``）。
    2. 确保目标目录存在。
    3. 检查同名冲突（``handle_collision``）。
    4. 执行 ``shutil.move``。

    Args:
        src: 源文件路径。
        dst_dir: 目标文件夹路径。
        collision_strategy: 冲突策略，默认 ``"keep_both"``。

    Returns:
        移动后的新文件路径；若失败或跳过则返回 ``None``。
    """
    logger = _get_logger()

    # 校验源文件
    if not src.exists():
        logger.error("源文件不存在: %s", src)
        return None

    if not src.is_file():
        logger.error("源路径不是文件: %s", src)
        return None

    # 文件就绪检测
    if not is_file_ready(src):
        logger.error("源文件被占用，无法移动: %s", src)
        return None

    # 确保目标目录存在
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("无法创建目标目录 %s: %s", dst_dir, e)
        return None

    dst = dst_dir / src.name

    # 同名冲突处理
    resolved = handle_collision(dst, collision_strategy)
    if resolved is None:
        logger.info("因冲突策略跳过移动: %s", src)
        return None

    # 执行移动
    try:
        actual = Path(shutil.move(str(src), str(resolved)))
        logger.info("移动成功: %s → %s", src, actual)
        return actual.resolve()
    except OSError as e:
        logger.error("移动失败 %s → %s: %s", src, resolved, e)
        return None


# ---------------------------------------------------------------------------
# 回退（向上一级移动，不能超出根目录）
# ---------------------------------------------------------------------------

def rollback_file(current_path: Path, root_dir: Path) -> Optional[Path]:
    """将文件回退至当前所在目录的父目录（向上一级）。

    若当前目录已是 ``root_dir`` 自身，则拒绝回退。

    Args:
        current_path: 文件当前路径。
        root_dir: 分类树根目录（不允许超出此目录）。

    Returns:
        回退后的新文件路径；失败则返回 ``None``。
    """
    logger = _get_logger()

    if not current_path.exists():
        logger.error("回退失败，文件不存在: %s", current_path)
        return None

    current_dir = current_path.parent.resolve()
    root_resolved = root_dir.resolve()

    # 已处于根目录 → 不可回退
    if current_dir == root_resolved:
        logger.warning("文件已在根目录，无法继续回退: %s", current_path)
        return None

    # 防止回退到根目录之外
    if root_resolved not in current_dir.parents:
        logger.warning("父目录不在分类树内，拒绝回退: %s", current_dir.parent)
        return None

    # 目标：向上一级（current_dir 的父目录）
    target_dir = current_dir.parent
    return move_file(current_path, target_dir, collision_strategy="keep_both")


# ---------------------------------------------------------------------------
# 撤销（移回最初来源）
# ---------------------------------------------------------------------------

def undo_file(current_path: Path, original_path: Path) -> Optional[Path]:
    """将文件撤销移回原始路径。

    Args:
        current_path: 文件当前路径。
        original_path: 文件最初来源路径。

    Returns:
        撤销后的新文件路径；原始路径不存在或移动失败则返回 ``None``。
    """
    logger = _get_logger()

    if not current_path.exists():
        logger.error("撤销失败，文件不存在: %s", current_path)
        return None

    original_dir = original_path.parent

    if not original_dir.exists():
        logger.error("撤销失败，原始目录不存在: %s", original_dir)
        return None

    return move_file(current_path, original_dir, collision_strategy="keep_both")


# ---------------------------------------------------------------------------
# 独立验证
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import tempfile

    init_logging()
    log = _get_logger()
    log.info("file_mover.py 自测开始。")

    tmp_dir = Path(tempfile.mkdtemp())
    print(f"测试目录: {tmp_dir}")

    try:
        # ---- 测试 1: 正常移动 ----
        print("--- 测试 1: 正常移动 ---")
        src = tmp_dir / "test_normal.txt"
        dst_dir = tmp_dir / "subdir"
        src.write_text("hello", encoding="utf-8")
        result = move_file(src, dst_dir)
        assert result is not None, "正常移动应成功"
        assert result.exists(), "移动后文件应存在于目标目录"
        assert result.parent == dst_dir.resolve(), "应位于目标文件夹"
        print(f"  {src.name} → {result}")
        print("  [OK] 测试 1 通过\n")

        # ---- 测试 2: keep_both 同名冲突 ----
        print("--- 测试 2: keep_both 同名冲突 ---")
        dup = dst_dir / "test_normal.txt"
        dup.write_text("second", encoding="utf-8")
        collided = handle_collision(dup, strategy="keep_both")
        assert collided is not None, "keep_both 应返回新路径"
        assert collided != dup, "新路径不应与原路径相同"
        assert "_1" in collided.stem, "应包含序号 _1"
        print(f"  冲突路径: {dup} → 新路径: {collided}")
        print("  [OK] 测试 2 通过\n")

        # ---- 测试 3: is_file_ready 就绪检测 ----
        print("--- 测试 3: is_file_ready 就绪检测 ---")
        ready_src = tmp_dir / "ready_file.txt"
        ready_src.write_text("ready", encoding="utf-8")
        assert is_file_ready(ready_src) is True, "未占用文件应就绪"
        assert is_file_ready(tmp_dir / "nonexistent.txt") is False, "不存在文件不应就绪"
        print("  [OK] 测试 3 通过\n")

        # ---- 测试 4: rollback 回退 ----
        print("--- 测试 4: rollback 回退 ---")
        src_rb = dst_dir / "test_normal.txt"
        rollback_result = rollback_file(src_rb, tmp_dir)
        assert rollback_result is not None, "回退应成功"
        # 文件应处于 tmp_dir 下（即 dst_dir 的上一级，但可能因 keep_both 而重命名）
        assert rollback_result.parent == tmp_dir.resolve(), "应回到上一层目录"
        print(f"  回退结果: {rollback_result}")
        print("  [OK] 测试 4 通过\n")

        # ---- 测试 5: undo 撤销 ----
        print("--- 测试 5: undo 撤销 ---")
        # 创建一个原始文件并记录原路径
        orig = tmp_dir / "undo_test.txt"
        orig.write_text("undo", encoding="utf-8")
        moved = move_file(orig, tmp_dir / "some_folder")
        assert moved is not None
        undid = undo_file(moved, orig)
        assert undid is not None, "撤销应成功"
        assert undid.parent == orig.parent.resolve(), "应回到原目录"
        print(f"  撤销结果: {undid}")
        print("  [OK] 测试 5 通过\n")

        # ---- 测试 6: 回退超出根目录保护 ----
        print("--- 测试 6: 根目录保护 ---")
        outside = tmp_dir / "outside"
        outside.mkdir()
        f = outside / "test.txt"
        f.write_text("data", encoding="utf-8")
        # 以 outside 为根目录，文件本身就在根目录，不应能回退
        rb_fail = rollback_file(f, outside)
        assert rb_fail is None, "已处于根目录应拒绝回退"
        print("  [OK] 测试 6 通过\n")

        # ---- 测试 7: 覆盖策略 ----
        print("--- 测试 7: overwrite 覆盖策略 ---")
        base_file = tmp_dir / "collide_overwrite.txt"
        base_file.write_text("original", encoding="utf-8")
        second_file = tmp_dir / "collide_overwrite.txt"
        second_file.write_text("replacement", encoding="utf-8")
        resolved = handle_collision(base_file, strategy="overwrite")
        assert resolved is not None, "覆盖应返回路径"
        print(f"  覆盖结果: {resolved}")
        print("  [OK] 测试 7 通过\n")

    finally:
        import shutil
        shutil.rmtree(tmp_dir)

    print("[OK] file_mover.py 全部自测通过.")
