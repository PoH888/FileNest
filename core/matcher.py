"""
FileNest — 文件名匹配模块

负责文件名预处理（规范化、短文件名保护）、模糊匹配及多候选排序。
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import Levenshtein
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
    return len(normalized) <= 1


# ---------------------------------------------------------------------------
# 模糊匹配
# ---------------------------------------------------------------------------

# ── 严格评分（max_len 归一化） —————————————————
# fuzzywuzzy 使用 len(s1)+len(s2) 做分母，短拉丁词会分数虚高：
#   "math" vs "music" → 67（实际应 ~40）
# 下面函数用 1 - dist / max_len 公式修正。

def _levenshtein_ratio(s1: str, s2: str) -> int:
    """Levenshtein 相似度，用 **最大长度** 归一化。

    Args:
        s1, s2: 两字符串。

    Returns:
        0–100 的整数分数。
    """
    if not s1 and not s2:
        return 100
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 100
    dist = Levenshtein.distance(s1, s2)
    return max(0, int(round((1 - dist / max_len) * 100)))


def _token_sort_strict(s1: str, s2: str) -> int:
    """分词排序后用 ``_levenshtein_ratio`` 比较。"""
    tokens1 = sorted(re.split(r"[\s_\-]+", s1))
    tokens2 = sorted(re.split(r"[\s_\-]+", s2))
    return _levenshtein_ratio(" ".join(tokens1).strip(),
                              " ".join(tokens2).strip())


def _partial_strict(s1: str, s2: str) -> int:
    """寻找最佳子串匹配，使用 ``_levenshtein_ratio``。"""
    shorter, longer = sorted([s1, s2], key=len)
    m = len(shorter)
    if m == 0:
        return 100
    best = 0
    for i in range(len(longer) - m + 1):
        ratio = _levenshtein_ratio(shorter, longer[i:i + m])
        if ratio > best:
            best = ratio
    return best


def _is_latin_sensitive(s: str) -> bool:
    """判断字符串是否为"短拉丁"——长度 ≤ 5 且包含 ≥ 2 个拉丁字母。

    fuzzywuzzy 的 sum-normalization 对此类字符串的匹配分数
    会虚假膨胀，需改用 ``_token_sort_strict`` / ``_partial_strict``。
    """
    if len(s) > 5:
        return False
    latin = sum(1 for c in s if 'a' <= c <= 'z')
    return latin >= 2


def _score_folder(filename_normalized: str, folder_name: str) -> int:
    """计算单个文件夹名称与目标文件名的相似度。

    对**短拉丁**字符串使用 max_len 归一化的严格评分，避免
    fuzzywuzzy 的 sum-normalization 导致 "math" → "Music" 等
    无关匹配分数虚高。中文及长字符串仍使用 fuzzywuzzy 以获得
    更柔和的模糊匹配效果。

    Args:
        filename_normalized: 规范化后的文件名。
        folder_name: 文件夹名称（不含路径）。

    Returns:
        0–100 的相似度分数。
    """
    folder_normalized = folder_name.lower().strip()
    if not folder_normalized:
        return 0

    # 判断是否有短拉丁字符串参与
    fn_latin = _is_latin_sensitive(filename_normalized)
    fld_latin = _is_latin_sensitive(folder_normalized)

    if fn_latin or fld_latin:
        token_sort = _token_sort_strict(filename_normalized, folder_normalized)
        partial = _partial_strict(filename_normalized, folder_normalized)
    else:
        token_sort = fuzz.token_sort_ratio(filename_normalized, folder_normalized)
        partial = fuzz.partial_ratio(filename_normalized, folder_normalized)

    # 文件夹名远超文件名时，子串匹配的价值降低
    fn_len = len(filename_normalized)
    folder_len = len(folder_normalized)
    if folder_len > fn_len * 2:
        ratio = fn_len / folder_len
        partial = int(partial * ratio)

    return max(token_sort, partial)


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


def _keyword_overlap_score(filename_normalized: str, folder_name: str) -> int:
    """计算文件名与文件夹名的关键词重叠度加分。

    对中英文混合场景使用三种策略取最大值：
    - **整词匹配**：适用于英文单词。
    - **bigram 匹配**：双字/双字母 n-gram，对中英文通用。
    - **连续子串匹配**：文件名中最长连续子串出现在文件夹名中的长度占比。

    按命中比例给予 0–15 分的额外加分。

    Args:
        filename_normalized: 规范化后的文件名。
        folder_name: 文件夹名称（不含路径）。

    Returns:
        0–15 的加分值。
    """
    folder_lower = folder_name.lower()
    fn = filename_normalized

    # 策略 1：整词重叠
    f_words = [w for w in fn.split() if len(w) > 1]
    word_hits = sum(1 for w in f_words if w in folder_lower)
    word_ratio = word_hits / max(len(f_words), 1) if f_words else 0

    # 策略 2：bigram 匹配（中英文通用）
    # 用双字/双字母 bigram 替代单字/单字母，避免单个高频字符（学、文、e、a 等）
    # 造成虚假匹配。例如"大学英语四级词汇"与"数学"共享"学"一个字符 → 不改前会加分
    fn_clean = fn.replace(" ", "")
    folder_clean = folder_lower.replace(" ", "")
    fn_bigrams = {fn_clean[i:i+2] for i in range(max(len(fn_clean) - 1, 0))}
    folder_bigrams = {folder_clean[i:i+2] for i in range(max(len(folder_clean) - 1, 0))}
    bigram_hits = len(fn_bigrams & folder_bigrams)
    bigram_ratio = bigram_hits / max(len(fn_bigrams), 1)

    # 策略 3：最长连续子串匹配（忽略单字符匹配）
    longest_sub = 0
    for i in range(len(fn)):
        for j in range(i + 2, len(fn) + 1):  # j=i+2 → 至少 2 字符
            if fn[i:j] in folder_lower:
                longest_sub = max(longest_sub, j - i)
    sub_ratio = longest_sub / max(len(fn), 1) if longest_sub >= 2 else 0

    # 取三种策略中最好的
    ratio = max(word_ratio, bigram_ratio, sub_ratio)

    if ratio >= 0.5:
        return 15
    if ratio >= 0.3:
        return 10
    if ratio >= 0.15:
        return 5
    return 0


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

    - 空列表：无匹配（最高分 < 55）。
    - 单条且分数 ≥ threshold：强匹配，可直接移动。
    - 多条且至少两条 ≥ threshold：多候选，需要用户选择。
    - 单条但分数 < threshold 且 ≥ 55：中等匹配，可询问或自动移动。

    Args:
        file_name: 原始文件名（含扩展名）。
        target_folders: 所有目标文件夹的 Path 列表。
        threshold: 自动匹配阈值（默认 85）。
        parent_weight: 是否启用父文件夹名称加权（默认开启）。

    Returns:
        按相似度降序排列的 ``[(Path, score), ...]`` 列表，
        仅包含分数 ≥ 55 的候选；空列表表示无任何有意义的匹配。
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
        score = min(100, score + _keyword_overlap_score(normalized, folder_name))

        # 低于 55 分的无意义匹配直接丢弃（符合规划文档"无匹配"分支）
        if score >= 55:
            scored.append((folder, score))

    # 按分数降序排列
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def has_clear_winner(matches: List[Tuple[Path, int]]) -> bool:
    """判断候选列表中第一名是否明显领先。

    规则：
    - 空候选 → False
    - 只有一个候选 → True
    - 第一名 ≥ 95 且与第二名不同分 → True
    - 第一名 ≥ 90 且领先第二名 ≥ 15 → True
    - 其余 → False

    Args:
        matches: ``get_best_matches`` 返回的候选列表。

    Returns:
        有明确赢家返回 ``True``。
    """
    if not matches:
        return False
    if len(matches) == 1:
        return True

    top_score = matches[0][1]
    second_score = matches[1][1]

    if top_score >= 95 and top_score > second_score:
        return True
    if top_score >= 90 and (top_score - second_score) >= 15:
        return True
    return False


def has_family_overlap(
    matches: List[Tuple[Path, int]],
) -> bool:
    """检查候选列表中是否存在父子文件夹同时出现的情况。

    Args:
        matches: ``get_best_matches`` 返回的候选列表。

    Returns:
        如果任意父文件夹和它的子文件夹同时出现在列表中返回 ``True``。
    """
    paths = {p for p, _ in matches}
    for p1 in paths:
        for p2 in paths:
            if p1 == p2:
                continue
            if p1 in p2.parents:
                return True
    return False


def extract_family_group(
    matches: List[Tuple[Path, int]],
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """从候选列表中分离出存在父子关系的"家族组"和独立的"非家族项"。

    当检测到家族重叠时，树形窗口只需要显示存在嵌套关系的文件夹；
    与家族无关的独立文件夹应该走普通候选逻辑。

    Args:
        matches: ``get_best_matches`` 返回的候选列表。

    Returns:
        ``(family_group, standalone)`` 二元组：
        - family_group：存在父子关系的所有文件夹（含父和子）。
        - standalone：与 family_group 无任何父子关系的文件夹。
    """
    paths = {p for p, _ in matches}

    # 找出参与家族关系的所有路径
    family_paths: set[Path] = set()
    for p1 in paths:
        for p2 in paths:
            if p1 == p2:
                continue
            if p1 in p2.parents:
                family_paths.add(p1)
                family_paths.add(p2)

    family_group = [(p, s) for p, s in matches if p in family_paths]
    standalone = [(p, s) for p, s in matches if p not in family_paths]

    # 按分数降序
    family_group.sort(key=lambda x: x[1], reverse=True)
    standalone.sort(key=lambda x: x[1], reverse=True)

    return family_group, standalone


def resolve_family(
    matches: List[Tuple[Path, int]],
    auto_threshold: int = 85,
    child_threshold: int = 65,
) -> Tuple[Optional[Path], List[Tuple[Path, int]]]:
    """在存在家族重叠时自动决策或返回需要用户选择的候选集。

    决策优先级：
    1. 如果有任意子文件夹分数 ≥ ``auto_threshold``，返回该子文件夹（自动归入）。
    2. 否则如果只有一个子文件夹且分数 ≥ ``child_threshold``，返回该子文件夹。
    3. 否则返回候选集（用户通过树形窗口选择）。

    Args:
        matches: ``get_best_matches`` 返回的候选列表。
        auto_threshold: 自动采纳阈值（默认 85）。
        child_threshold: 唯一子文件夹自动采纳阈值（默认 65）。

    Returns:
        ``(selected_path, remaining_candidates)`` 二元组：
        - 若自动决策成功，selected_path 为结果，remaining_candidates 为空列表。
        - 若需要用户选择，selected_path 为 None，remaining_candidates 为完整候选列表。
    """
    candidates = list(matches)
    paths = {p for p, _ in matches}

    # 收集所有子文件夹（有父文件夹也在候选中的）
    children = []
    for child_path, child_score in candidates:
        parent_dirs = set(child_path.parents)
        if parent_dirs & paths:
            children.append((child_path, child_score))

    # 有子文件夹存在 → 检查自动决策条件
    if children:
        # 规则 1：最佳子文件夹 ≥ auto_threshold
        children.sort(key=lambda x: x[1], reverse=True)
        best_child, best_child_score = children[0]
        if best_child_score >= auto_threshold:
            return (best_child, [])

        # 规则 2：唯一子文件夹 ≥ child_threshold
        if len(children) == 1 and best_child_score >= child_threshold:
            return (best_child, [])

    # 无法自动决策 → 返回完整候选列表让用户选择
    return (None, candidates)


def build_family_tree(
    matches: List[Tuple[Path, int]],
) -> List[Tuple[Path, Optional[List[Tuple[Path, int]]]]]:
    """将扁平候选列表构造成树形结构，供树形选择窗口渲染。

    Args:
        matches: ``get_best_matches`` 返回的候选列表。

    Returns:
        树形节点列表，每个节点为 ``(folder_path, children_or_none)``。
        顶层文件夹在最外层，其直接子文件夹嵌套在 children 列表中。
    """
    candidates = list(matches)
    paths = {p for p, _ in candidates}

    # 找出所有顶层节点（父文件夹不在候选中的节点）
    roots = []
    for p, _ in candidates:
        parent_in_candidates = False
        for parent in p.parents:
            if parent in paths:
                parent_in_candidates = True
                break
        if not parent_in_candidates:
            roots.append(p)

    def build_children(parent_path: Path) -> List[Tuple[Path, int]]:
        """递归构建指定父文件夹的直接子节点。"""
        result = []
        for p, s in candidates:
            if p.parent == parent_path and p in paths:
                result.append((p, s))
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def build_node(folder: Path) -> Tuple[Path, Optional[List[Tuple[Path, int]]]]:
        """递归构建单个节点及其子树。"""
        children = build_children(folder)
        score = dict(candidates).get(folder, 0)
        if children:
            return (folder, children)
        return (folder, None)

    result = [(build_node(root)) for root in roots]

    # 同时添加所有子节点中不在 trees 的嵌套子树
    def add_descendants(node_path: Path, node_children: Optional[List[Tuple[Path, int]]]) -> List[Tuple[Path, Optional[List[Tuple[Path, int]]]]]:
        result = [(node_path, node_children)]
        if node_children:
            for child_path, child_score in node_children:
                grand_children = build_children(child_path)
                if grand_children:
                    result.extend(add_descendants(child_path, grand_children))
        return result

    full_tree = []
    for root in roots:
        full_tree.extend(add_descendants(root, build_children(root)))

    return full_tree
