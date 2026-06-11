"""
FileNest — 文件夹扫描模块

负责递归扫描目标根目录，获取所有子文件夹路径，
应用最大深度、忽略名单等过滤条件，安全跳过权限受限的目录。
"""

import fnmatch
import logging
from pathlib import Path
from typing import List, Tuple

import config_manager

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

_logger_initialized: bool = False


def _get_logger() -> logging.Logger:
    """延迟获取全局日志记录器。"""
    global _logger_initialized  # noqa: PLW0603
    if not _logger_initialized:
        # 从 config_manager 借 init_logging（它内部有单例保护）
        from utils import init_logging  # pylint: disable=import-outside-toplevel
        lgr, _ = init_logging()
        _logger_initialized = True
        return lgr
    return logging.getLogger("FileNest")


# ---------------------------------------------------------------------------
# 忽略模式匹配
# ---------------------------------------------------------------------------

def _parse_ignore_patterns(raw: str) -> List[str]:
    """将配置中的忽略字符串解析为模式列表。

    支持分号分隔及 ``*`` / ``?`` 通配符（由 ``fnmatch`` 处理）。

    Args:
        raw: 原始忽略字符串，如 ``"备份;临时;*缓存*"``。

    Returns:
        去除空白后的模式列表，空字符串时返回空列表。
    """
    if not raw or not raw.strip():
        return []
    return [p.strip() for p in raw.split(";") if p.strip()]


def _should_ignore(name: str, patterns: List[str]) -> bool:
    """检查文件夹名是否匹配任一忽略模式。

    支持三种匹配方式：
    - ``fnmatch`` 通配符（如 ``*缓存*``）。
    - 纯后缀匹配：以 ``.`` 开头的模式（如 ``.tmp``）视为 ``*.tmp``。
    - 精确关键字匹配。

    Args:
        name: 文件夹名称（不含路径）。
        patterns: 忽略模式列表。

    Returns:
        匹配任意模式返回 ``True``。
    """
    for pattern in patterns:
        # dot 后缀简写: ".tmp" -> "*.tmp"
        if pattern.startswith(".") and not pattern.startswith(".*"):
            if name.endswith(pattern):
                return True
        if fnmatch.fnmatch(name, pattern):
            return True
        # 精确关键字
        if pattern == name:
            return True
    return False


def _is_hidden_or_system(name: str) -> bool:
    """判断文件夹名是否为隐藏/系统目录。

    Args:
        name: 文件夹名称。

    Returns:
        以 ``.`` 开头返回 ``True``（Windows 下也适用）。
    """
    return name.startswith(".")


# ---------------------------------------------------------------------------
# 核心扫描器
# ---------------------------------------------------------------------------

def scan_directory(target_dir: Path, max_depth: int = 5,
                   ignore_patterns: List[str] | None = None) -> List[Path]:
    """递归扫描目标目录，返回所有符合条件的子文件夹路径。

    扫描规则：
    - 至多递归 ``max_depth`` 层（深度从 0 开始计数，0 表示仅返回直接子目录）。
    - 跳过名称匹配 ``ignore_patterns`` 的文件夹。
    - 跳过以 ``.`` 开头的隐藏/系统目录。
    - 单个目录权限问题不影响其他目录的扫描。

    Args:
        target_dir: 要扫描的根目录。
        max_depth: 最大递归深度，默认 5。
        ignore_patterns: 忽略名称模式列表，为 ``None`` 时从配置读取。

    Returns:
        排序后的子文件夹 Path 列表。
    """
    if ignore_patterns is None:
        cfg = config_manager.load_config()
        ignore_patterns = _parse_ignore_patterns(cfg.get("ignore_patterns", ""))

    logger = _get_logger()
    results: List[Path] = []

    try:
        target_dir = target_dir.resolve(strict=False)
    except OSError as e:
        logger.warning("无法解析目标目录 %s: %s", target_dir, e)
        return results

    if not target_dir.is_dir():
        logger.warning("目标路径不是有效目录: %s", target_dir)
        return results

    _scan_recursive(target_dir, 0, max_depth, ignore_patterns, results, logger)
    results.sort()
    return results


def _scan_recursive(current_dir: Path, current_depth: int, max_depth: int,
                    ignore_patterns: List[str], results: List[Path],
                    log: logging.Logger) -> None:
    """递归扫描的内部实现。

    Args:
        current_dir: 当前扫描的目录。
        current_depth: 当前递归深度（根目录为 0）。
        max_depth: 最大允许深度（1 表示仅直接子目录）。
        ignore_patterns: 忽略模式列表。
        results: 结果收集列表（就地修改）。
        log: 日志记录器。
    """
    next_depth = current_depth + 1
    if next_depth > max_depth:
        return

    try:
        iterator = current_dir.iterdir()
    except PermissionError:
        log.warning("权限不足，跳过目录: %s", current_dir)
        return
    except FileNotFoundError:
        log.warning("目录不存在，跳过: %s", current_dir)
        return
    except OSError as e:
        log.warning("无法访问目录 %s: %s", current_dir, e)
        return

    for entry in iterator:
        try:
            entry_path = Path(entry)
            name = entry_path.name

            # 跳过隐藏/系统目录
            if _is_hidden_or_system(name):
                continue

            # 检查是否为有效目录
            if not entry_path.is_dir():
                continue

            # 忽略名单检查
            if _should_ignore(name, ignore_patterns):
                log.debug("忽略匹配目录: %s", entry_path)
                continue

            # 当前条目符合深度要求，加入结果
            results.append(entry_path.resolve())

            # 递归深入（下一层）
            _scan_recursive(entry_path, next_depth, max_depth,
                            ignore_patterns, results, log)

        except PermissionError:
            log.warning("权限不足，跳过: %s", entry)
        except FileNotFoundError:
            log.warning("条目不存在，跳过: %s", entry)
        except OSError as e:
            log.warning("访问条目出错 %s: %s", entry, e)


# ---------------------------------------------------------------------------
# 获取所有根目录的子文件夹（便捷方法）
# ---------------------------------------------------------------------------

def scan_all_roots() -> List[Tuple[Path, List[Path]]]:
    """扫描所有配置中的根目录，返回每个根目录下的子文件夹。

    Returns:
        列表，每个元素为 ``(根目录, 子文件夹列表)`` 的二元组。
        若某根目录无效，其子文件夹列表为空。
    """
    cfg = config_manager.load_config()
    roots: List[str] = cfg.get("root_directories", [])
    max_depth: int = cfg.get("max_depth", 5)
    ignore_patterns = _parse_ignore_patterns(cfg.get("ignore_patterns", ""))

    result: List[Tuple[Path, List[Path]]] = []
    for root_str in roots:
        root = Path(root_str)
        folders = scan_directory(root, max_depth=max_depth,
                                 ignore_patterns=ignore_patterns)
        result.append((root, folders))
    return result


# ---------------------------------------------------------------------------
# 独立验证
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    import sys
    import tempfile

    from utils import init_logging
    init_logging()
    log = _get_logger()
    log.info("folder_scanner.py 自测开始。")

    # ---- 构建假目录树 ----
    tmp_root = Path(tempfile.mkdtemp())
    print(f"测试目录: {tmp_root}")

    def _makedirs(rel: str) -> Path:
        p = tmp_root / rel
        p.mkdir(parents=True, exist_ok=True)
        return p

    # 树结构：
    # tmp_root/
    #   ├── Music/
    #   │   ├── Rock/
    #   │   ├── Jazz/
    #   │   └── Classical/
    #   ├── Documents/
    #   │   ├── Work/
    #   │   │   └── Reports/         (depth 3)
    #   │   └── Personal/
    #   ├── .hidden_dir/             (应跳过)
    #   ├── __pycache__/             (应跳过 — 匹配 *.pyc 等, 但默认只检查 hidden)
    #   ├── .git/                    (应跳过 — hidden)
    #   └── backup.tmp/             (应跳过 — 匹配 .tmp 模式)

    _makedirs("Music/Rock")
    _makedirs("Music/Jazz")
    _makedirs("Music/Classical")
    _makedirs("Documents/Work/Reports")
    _makedirs("Documents/Personal")
    _makedirs(".hidden_dir/sub")
    _makedirs(".git/objects")
    _makedirs("backup.tmp/cache")

    # ---- 测试 1: 默认扫描 (depth=5, 无忽略) ----
    folders = scan_directory(tmp_root, max_depth=5, ignore_patterns=[])
    names = {p.name for p in folders}
    assert "Music" in names, "应包含 Music"
    assert "Documents" in names, "应包含 Documents"
    assert "Work" in names, "应包含 Work"
    assert "Reports" in names, "应包含 Reports"
    assert "Personal" in names, "应包含 Personal"
    assert ".hidden_dir" not in names, "应跳过隐藏目录"
    assert ".git" not in names, "应跳过 .git"
    assert "backup.tmp" in names, "无忽略模式时应包含 backup.tmp"
    print("[OK] 测试 1: 默认扫描(含隐藏跳过) -- 通过")

    # ---- 测试 2: max_depth 限制 ----
    folders_d1 = scan_directory(tmp_root, max_depth=1, ignore_patterns=[])
    names_d1 = {p.name for p in folders_d1}
    assert "Music" in names_d1
    assert "Documents" in names_d1
    assert "Work" not in names_d1, "depth=1 不应扫描到 Work"
    assert "Reports" not in names_d1, "depth=1 不应扫描到 Reports"
    print("[OK] 测试 2: max_depth=1 限制 -- 通过")

    # ---- 测试 3: ignore_patterns 过滤 ----
    ignore = [".tmp", "__pycache__"]
    folders_ig = scan_directory(tmp_root, max_depth=5,
                                ignore_patterns=ignore)
    names_ig = {p.name for p in folders_ig}
    assert "backup.tmp" not in names_ig, "应忽略 .tmp 目录"
    assert "Music" in names_ig, "正常目录应保留"
    print("[OK] 测试 3: ignore_patterns 过滤 -- 通过")

    # ---- 测试 4: 空目录 / 不存在路径 ----
    empty_result = scan_directory(tmp_root / "NotExists")
    assert empty_result == [], "不存在路径应返回空列表"
    print("[OK] 测试 4: 不存在的路径 -- 通过")

    # 清理
    shutil.rmtree(tmp_root)
    print("\n[OK] folder_scanner.py 全部自测通过.")
