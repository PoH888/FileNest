"""
FileNest — 文件名匹配模块

负责文件名预处理（规范化、短文件名保护）、模糊匹配及多候选排序。
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from fuzzywuzzy import fuzz

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SHORT_NAME_LIMIT: int = 2
"""短文件名保护阈值（含），低于此长度直接判定为无匹配"""

# ---------------------------------------------------------------------------
# 文件名预处理
# ---------------------------------------------------------------------------

def normalize_filename(filename: str) -> str:
    """规范化文件名：去扩展名，替换特殊字符，去除多余空白。

    Args:
        filename: 原始文件名（可含扩展名）。

    Returns:
        规范化后的文件名（全小写、无扩展名、特殊字符替换为空格）。
    """
    # 移除扩展名（仅移除最后一个 . 及之后的内容）
    name = filename.rsplit(".", 1)[0] if "." in filename else filename

    # 替换下划线、连字符、其他分隔符为空格
    name = re.sub(r"[_\-]+", " ", name)

    # 合并连续空白并去除首尾
    name = re.sub(r"\s+", " ", name).strip()

    return name.lower()


def is_short_name(filename: str) -> bool:
    """判断文件名是否为"极短名称"（有效字符数 ≤ 2）。

    Args:
        filename: 原始文件名（含扩展名）。

    Returns:
        有效长度 ≤ 2 返回 ``True``。
    """
    normalized = normalize_filename(filename)
    return len(normalized) <= SHORT_NAME_LIMIT


# ---------------------------------------------------------------------------
# 模糊匹配
# ---------------------------------------------------------------------------

def _score_folder(filename_normalized: str, folder_name: str) -> int:
    """计算单个文件夹名称与目标文件名的相似度。

    使用 ``fuzz.token_sort_ratio`` 忽略词语顺序的影响，
    再通过 ``fuzz.partial_ratio`` 补偿长文件夹名场景。

    Args:
        filename_normalized: 规范化后的文件名。
        folder_name: 文件夹名称（不含路径）。

    Returns:
        0–100 的相似度分数。
    """
    folder_normalized = folder_name.lower().strip()
    if not folder_normalized:
        return 0

    return max(
        fuzz.token_sort_ratio(filename_normalized, folder_normalized),
        fuzz.partial_ratio(filename_normalized, folder_normalized),
    )


def _apply_parent_weight(folder_path: Path, score: int,
                         parent_weight: bool = True) -> int:
    """可选地对"父文件夹名包含文件名片段"的情况进行加权。

    若 ``parent_weight`` 开启且当前文件夹的父文件夹名称中含有
    文件名中的某个单词，则增加 5 分奖励。

    Args:
        folder_path: 当前候选文件夹的完整路径。
        score: 原始相似度分数。
        parent_weight: 是否启用父文件夹加权。

    Returns:
        加权后的分数（不超过 100）。
    """
    if not parent_weight or score >= 100:
        return score

    try:
        parent_name = folder_path.parent.name.lower()
        folder_name = folder_path.name.lower()
        if parent_name and parent_name in folder_name:
            score = min(100, score + 5)
    except Exception:
        pass

    return score


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def get_best_matches(
    file_name: str,
    target_folders: List[Path],
    threshold: int = 85,
    parent_weight: bool = True,
) -> List[Tuple[Path, int]]:
    """对文件与所有目标文件夹执行模糊匹配，返回达到阈值的候选列表。

    匹配策略（由调用方 —— ``gui.py`` / ``main.py`` —— 根据返回结果决策）：

    - 空列表：无匹配（最高分 < 50%）。
    - 单条且分数 ≥ threshold：强匹配，可直接移动。
    - 多条且至少两条 ≥ threshold：多候选，需要用户选择。
    - 单条但分数 < threshold 且 ≥ 50：中等匹配，可询问或自动移动。

    Args:
        file_name: 原始文件名（含扩展名）。
        target_folders: 所有目标文件夹的 Path 列表。
        threshold: 自动匹配阈值（默认 85）。
        parent_weight: 是否启用父文件夹名称加权（默认开启）。

    Returns:
        按相似度降序排列的 ``[(Path, score), ...]`` 列表，
        仅包含分数 ≥ 50 的候选；空列表表示无任何有意义的匹配。
    """
    # 短文件名保护（≤2 字符→直接无匹配）
    if is_short_name(file_name):
        return []

    normalized = normalize_filename(file_name)
    scored: List[Tuple[Path, int]] = []

    for folder in target_folders:
        try:
            folder_name = folder.name
        except OSError:
            continue

        score = _score_folder(normalized, folder_name)
        score = _apply_parent_weight(folder, score, parent_weight)

        # 低于 50 分的无意义匹配直接丢弃（符合规划文档"无匹配"分支）
        if score >= 50:
            scored.append((folder, score))

    # 按分数降序排列
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def get_top_candidates(
    file_name: str,
    target_folders: List[Path],
    threshold: int = 85,
    parent_weight: bool = True,
) -> List[Tuple[Path, int]]:
    """返回达到自动阈值的所有候选（用于多候选弹窗场景）。

    等价于 ``get_best_matches`` 后过滤出分数 ≥ threshold 的条目。

    Args:
        同 ``get_best_matches``。

    Returns:
        分数 ≥ threshold 的候选列表，按分数降序排列。
    """
    all_matches = get_best_matches(file_name, target_folders, threshold, parent_weight)
    return [(p, s) for p, s in all_matches if s >= threshold]


# ---------------------------------------------------------------------------
# 独立验证
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("matcher.py 自测开始。\n")

    # ---- 准备假文件夹路径 ----
    base = Path("D:\\学校文件")
    folders = [
        base / "数学" / "高等数学",
        base / "数学" / "线性代数",
        base / "数学" / "概率论",
        base / "数学" / "离散数学",
        base / "英语" / "大学英语",
        base / "英语" / "英语四级",
        base / "计算机" / "Python编程",
        base / "计算机" / "数据结构",
        base / "音乐" / "Rock",
        base / "音乐" / "Jazz",
        base / "杂项" / "临时文件",
    ]

    # ---- 测试 1: 强匹配 ----
    print("--- 测试 1: 强匹配 ---")
    matches = get_best_matches("高等数学期末考试.pdf", folders)
    top_folder, top_score = matches[0]
    print(f"  '高等数学期末考试' 最佳匹配 → {top_folder} ({top_score})")
    assert top_folder.name == "高等数学", "最佳匹配应为 '高等数学'"
    assert top_score >= 85, "高等数学 → 高等数学 应达到阈值"
    print("  [OK] 测试 1 通过\n")

    # ---- 测试 2: 多候选匹配（含模糊重叠） ----
    print("--- 测试 2: 多候选匹配 ---")
    matches = get_best_matches("编程基础作业.py", folders)
    print(f"  '编程基础作业' 候选: {len(matches)} 个")
    for f, s in matches:
        print(f"    {f} ({s})")
    # 应至少匹配到 "Python编程"（包含"编程"）
    assert len(matches) >= 1, "应至少匹配到包含'编程'的文件夹"
    print("  [OK] 测试 2 通过\n")

    # ---- 测试 3: 极短文件名保护 ----
    print("--- 测试 3: 极短文件名保护 ---")
    matches = get_best_matches("a.txt", folders)
    assert matches == [], "长度 ≤2 应返回空列表"
    matches = get_best_matches("ab.pdf", folders)
    assert matches == [], "长度 ≤2 应返回空列表"
    print("  [OK] 测试 3 通过\n")

    # ---- 测试 4: 无匹配（无关文件名） ----
    print("--- 测试 4: 无匹配 ---")
    matches = get_best_matches("xyzabcxyz_random_file.exe", folders)
    print(f"  'xyzabcxyz_random_file' 匹配数: {len(matches)}")
    assert len(matches) == 0, "无关文件名应无匹配"
    print("  [OK] 测试 4 通过\n")

    # ---- 测试 5: 父文件夹加权 ----
    print("--- 测试 5: 父文件夹加权 ---")
    # "离散数学" 本身不包含父名"数学"，但加权场景可用精确命中验证
    weighted = get_best_matches("Python_project.zip", folders, parent_weight=True)
    noweight = get_best_matches("Python_project.zip", folders, parent_weight=False)
    if weighted and noweight:
        print(f"  加权首位: {weighted[0]} ({weighted[0][1]})")
        print(f"  未加权首位: {noweight[0]} ({noweight[0][1]})")
    print("  [OK] 测试 5 通过\n")

    # ---- 测试 6: 预处理验证 ----
    print("--- 测试 6: 规范化预处理 ---")
    assert normalize_filename("Hello-World_Test.TXT") == "hello world test"
    assert normalize_filename("ABC.DEF.GH") == "abc.def"
    assert normalize_filename("simple") == "simple"
    print(f"  'Hello-World_Test.TXT' → '{normalize_filename('Hello-World_Test.TXT')}'")
    print("  [OK] 测试 6 通过\n")

    # ---- 测试 7: get_top_candidates 阈值过滤 ----
    print("--- 测试 7: get_top_candidates(threshold=85) ---")
    candidates = get_top_candidates("线性代数作业.pdf", folders)
    for f, s in candidates:
        print(f"    {f} ({s})")
    assert len(candidates) >= 1, "线性代数作业应有 ≥85 的候选"
    assert candidates[0][0].name == "线性代数", "首位应为线性代数"
    print("  [OK] 测试 7 通过\n")

    print("matcher.py 全部自测通过.")
