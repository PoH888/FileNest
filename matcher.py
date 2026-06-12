"""
FileNest — 文件名匹配模块

负责文件名预处理（规范化、短文件名保护）、模糊匹配及多候选排序。
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

def _score_folder(filename_normalized: str, folder_name: str) -> int:
    """计算单个文件夹名称与目标文件名的相似度。

    使用 ``fuzz.token_sort_ratio`` 忽略词语顺序的影响，
    再通过 ``fuzz.partial_ratio`` 补偿长文件夹名场景。

    当文件夹名长度远超文件名（> 2 倍）时，对 ``partial_ratio``
    按长度比例打折，避免因长路径包含短子串而误匹配。

    Args:
        filename_normalized: 规范化后的文件名。
        folder_name: 文件夹名称（不含路径）。

    Returns:
        0–100 的相似度分数。
    """
    folder_normalized = folder_name.lower().strip()
    if not folder_normalized:
        return 0

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

    # ---- 测试 2: keyword overlap 中文匹配（"高数"→"高等数学"） ----
    print("--- 测试 2: keyword overlap 中文匹配 ---")
    matches = get_best_matches("高数期末.pdf", folders)
    print(f"  '高数期末' 候选: {len(matches)} 个")
    for f, s in matches:
        print(f"    {f} ({s})")
    assert len(matches) >= 1, "应至少匹配到'高等数学'"
    assert matches[0][0].name == "高等数学", "首位应为高等数学"
    print("  [OK] 测试 2 通过\n")

    # ---- 测试 3: 无匹配（无关文件名） ----
    print("--- 测试 3: 无匹配 ---")
    matches = get_best_matches("xyzabcxyz_random_file.exe", folders)
    print(f"  'xyzabcxyz_random_file' 匹配数: {len(matches)}")
    assert len(matches) == 0, "无关文件名应无匹配"
    print("  [OK] 测试 3 通过\n")

    # ---- 测试 4: keyword overlap 跨目录（python 多目录） ----
    print("--- 测试 4: keyword overlap 跨目录 ---")
    code_folders = [
        base / "计算机" / "Python编程",
        Path("D:\\学校文件\\python课"),
        Path("D:\\学校文件\\python期末大作业"),
    ]
    code_matches = get_best_matches("Python_project_代码.zip", code_folders, threshold=50)
    assert len(code_matches) >= 2, "应找到多个 python 相关候选"
    for f, s in code_matches:
        print(f"    {f} ({s})")
    print("  [OK] 测试 4 通过\n")

    # ---- 测试 5: 预处理验证 ----
    print("--- 测试 5: 规范化预处理 ---")
    assert normalize_filename("Hello-World_Test.TXT") == "hello world test"
    assert normalize_filename("ABC.DEF.GH") == "abc.def"
    assert normalize_filename("simple") == "simple"
    print(f"  'Hello-World_Test.TXT' → '{normalize_filename('Hello-World_Test.TXT')}'")
    print("  [OK] 测试 5 通过\n")

    # ---- 测试 6: 极短文件名保护 ----
    print("--- 测试 6: 极短文件名保护 ---")
    assert is_short_name("a.txt") is True, "长度=1 应视为极短"
    assert is_short_name("ab.pdf") is False, "长度=2 不应拦截"
    print("  [OK] 测试 6 通过\n")

    # ---- 测试 7: 短文件名折中（长度=2 能匹配） ----
    print("--- 测试 8: 短文件名折中匹配 ---")
    short_matches = get_best_matches("高数.pdf", folders)
    if short_matches:
        print(f"  '高数' 最佳匹配 → {short_matches[0][0]} ({short_matches[0][1]})")
    else:
        print("  '高数' 无匹配（分数不足 50）")
    print("  [OK] 测试 8 通过\n")

    # ---- 测试 9: has_clear_winner — 领先 26 分 ----
    print("--- 测试 9: has_clear_winner 领先 26 分 ---")
    big_lead = [(Path("D:\\学校文件\\python课"), 100),
                (Path("D:\\学校文件\\python期末大作业"), 74)]
    assert has_clear_winner(big_lead) is True, "100 vs 74 → True"
    print("  100 vs 74 → True  [OK]")

    # ---- 测试 10: has_clear_winner — 接近 ----
    print("--- 测试 10: has_clear_winner 接近（88 vs 86）---")
    close = [(Path("D:\\学校文件\\英语"), 88),
             (Path("D:\\学校文件\\大学英语"), 86)]
    assert has_clear_winner(close) is False, "88 vs 86 → False"
    print("  88 vs 86 → False  [OK]")

    # ---- 测试 11: has_clear_winner — ≥95 直接赢 ----
    print("--- 测试 11: has_clear_winner ≥95 直接赢 ---")
    high = [(Path("D:\\学校文件\\数学"), 98),
            (Path("D:\\学校文件\\高等数学"), 90)]
    assert has_clear_winner(high) is True, "98 → True"
    print("  98 vs 90 → True  [OK]")

    # ---- 测试 12: has_clear_winner — 唯一候选 ----
    print("--- 测试 12: has_clear_winner 唯一候选 ---")
    single = [(Path("D:\\学校文件\\数学"), 75)]
    assert has_clear_winner(single) is True
    print("  唯一 75 分 → True  [OK]")

    # ---- 测试 13: has_clear_winner — 空列表 ----
    print("--- 测试 13: has_clear_winner 空列表 ---")
    assert has_clear_winner([]) is False
    print("  空列表 → False  [OK]")

    # ---- 测试 13b: has_clear_winner — 100 平局 ----
    print("--- 测试 13b: has_clear_winner 100 平局 ---")
    tied = [(Path("D:\\学校文件\\英语"), 100),
            (Path("D:\\学校文件\\英语\\大学英语"), 100)]
    assert has_clear_winner(tied) is False, "100 vs 100 → False"
    print("  100 vs 100 → False  [OK]\n")

    # ---- 测试 14: has_family_overlap — 父子同时出现 ----
    print("--- 测试 14: has_family_overlap 检测家族重叠 ---")
    eng_folders = [
        Path("D:\\学校文件\\英语"),
        Path("D:\\学校文件\\英语\\大学英语"),
        Path("D:\\学校文件\\英语\\中学英语"),
        Path("D:\\学校文件\\英语\\小学英语"),
    ]
    eng_matches = get_best_matches("英语四级词汇.docx", eng_folders)
    print(f"  候选列表:")
    for f, s in eng_matches:
        print(f"    {f} ({s})")
    assert has_family_overlap(eng_matches), "英语父子同时出现在列表中，应有重叠"
    print("  [OK] 测试 14 通过（检测到家族重叠）\n")

    # ---- 测试 15: extract_family_group ----
    print("--- 测试 15: extract_family_group 分离家族/非家族 ---")
    family, standalone = extract_family_group(eng_matches)
    assert all("英语" in str(p) for p, _ in family), "家族组应只含英语相关"
    assert len(standalone) == 0, "纯英语列表不应有非家族项"
    print("  [OK] 测试 15 通过\n")

    # ---- 测试 16: build_family_tree 树形结构 ----
    print("--- 测试 16: build_family_tree 树形结构 ---")
    tree_matches = get_best_matches("英语四级词汇.docx", eng_folders)
    tree = build_family_tree(tree_matches)
    print(f"  树形节点数: {len(tree)}")
    for node_path, children in tree:
        if children:
            print(f"    [{node_path.name}] ({dict(tree_matches).get(node_path, 0)}%)")
            for child_path, child_score in children:
                print(f"       +-- [{child_path.name}] ({child_score}%)")
        else:
            print(f"    [{node_path.name}] ({dict(tree_matches).get(node_path, 0)}%)")
    assert len(tree) >= 1, "应至少有一个根节点"
    root_names = {p.name for p, _ in tree}
    assert "英语" in root_names, "根节点应包含'英语'"
    eng_node = next((n for n in tree if n[0].name == "英语"), None)
    assert eng_node is not None
    assert eng_node[1] is not None and len(eng_node[1]) == 3, f"英语应有3个子节点，实际 {len(eng_node[1]) if eng_node[1] else 0}"
    print("  [OK] 测试 16 通过\n")

    # ---- 实景测试：Python_project_代码.zip — 100 vs 74 不应弹窗 ----
    print("--- 实景测试: Python_project_代码.zip 领先 26 分 ---")
    py_folders = [
        Path("D:\\学校文件\\python课"),
        Path("D:\\学校文件\\python期末大作业"),
        Path("D:\\学校文件\\python期末大作业\\report"),
    ]
    py_matches = get_best_matches("Python_project_代码.zip", py_folders)
    for f, s in py_matches:
        print(f"    {f} ({s})")
    assert has_clear_winner(py_matches), "100 vs 74 vs 60 → 有明显赢家"
    assert py_matches[0][0].name == "python课", "应自动归入 python课"
    print("  [OK] 实景测试通过（不弹窗，自动归入 python课）\n")

    print("matcher.py 全部自测通过.")

    # =================================================================
    # 英文测试集
    # =================================================================

    print("\n========== 英文测试集 ==========\n")

    eng_base = Path("D:\\EnglishTest")

    # ---- 测试 EN-1: Python_Project_Code.zip — Python vs Java/C++ ----
    print("--- EN-1: Python_Project_Code.zip ---")
    en1_folders = [
        eng_base / "Programming" / "Python",
        eng_base / "Programming" / "Java",
        eng_base / "Programming" / "C++",
    ]
    en1_matches = get_best_matches("Python_Project_Code.zip", en1_folders)
    for f, s in en1_matches:
        print(f"    {f} ({s})")
    assert en1_matches[0][0].name == "Python", "Python 应为第一名"
    # Java/C++ 不应达到阈值（≤55）
    java_in = any("Java" in str(p) for p, _ in en1_matches)
    cpp_in = any("C++" in str(p) for p, _ in en1_matches)
    assert not java_in, "Java 不应进入候选"
    assert not cpp_in, "C++ 不应进入候选"
    print("  [OK] EN-1 通过\n")

    # ---- 测试 EN-2: Machine_Learning_Notes.pdf — ML vs AI vs Web ----
    print("--- EN-2: Machine_Learning_Notes.pdf ---")
    en2_folders = [
        eng_base / "Artificial Intelligence",
        eng_base / "Machine Learning",
        eng_base / "Web Development",
    ]
    en2_matches = get_best_matches("Machine_Learning_Notes.pdf", en2_folders)
    for f, s in en2_matches:
        print(f"    {f} ({s})")
    assert en2_matches[0][0].name == "Machine Learning", "Machine Learning 应为第一名"
    # Web Development 不应进入候选
    web_dev = [p for p, _ in en2_matches if "Web Development" in str(p)]
    assert len(web_dev) == 0, "Web Development 不应进入候选列表"
    print("  [OK] EN-2 通过\n")

    # ---- 测试 EN-3: Tokyo_Travel_Plan.xlsx — Japan/Tokyo vs Finance ----
    print("--- EN-3: Tokyo_Travel_Plan.xlsx ---")
    en3_folders = [
        eng_base / "Japan Travel",
        eng_base / "Tokyo Trip",
        eng_base / "Finance",
    ]
    en3_matches = get_best_matches("Tokyo_Travel_Plan.xlsx", en3_folders)
    for f, s in en3_matches:
        print(f"    {f} ({s})")
    assert en3_matches[0][0].name == "Tokyo Trip", "Tokyo Trip 应为第一名"
    assert len(en3_matches) >= 2, "至少 Japan Travel 也应在候选"
    finance_in = any("Finance" in str(p) for p, _ in en3_matches)
    assert not finance_in, "Finance 不应进入候选"
    print("  [OK] EN-3 通过\n")

    # ---- 测试 EN-4: University_English_CET4_Vocabulary.docx — English vs Math ----
    print("--- EN-4: University_English_CET4_Vocabulary.docx ---")
    en4_folders = [
        eng_base / "English",
        eng_base / "College English",
        eng_base / "Mathematics",
    ]
    en4_matches = get_best_matches("University_English_CET4_Vocabulary.docx", en4_folders)
    for f, s in en4_matches:
        print(f"    {f} ({s})")
    assert en4_matches[0][0].name == "English", "English 应为第一名"
    assert len(en4_matches) >= 2, "至少 College English 也应在候选"
    math_in = any("Mathematics" in str(p) for p, _ in en4_matches)
    assert not math_in, "Mathematics 不应进入候选"
    print("  [OK] EN-4 通过\n")

    # ---- 测试 EN-5: Database_Final_Project.zip — DB vs OS vs Net ----
    print("--- EN-5: Database_Final_Project.zip ---")
    en5_folders = [
        eng_base / "Database",
        eng_base / "Operating Systems",
        eng_base / "Computer Networks",
    ]
    en5_matches = get_best_matches("Database_Final_Project.zip", en5_folders)
    for f, s in en5_matches:
        print(f"    {f} ({s})")
    assert en5_matches[0][0].name == "Database", "Database 应为第一名"
    assert has_clear_winner(en5_matches), "Database 应明显领先"
    print("  [OK] EN-5 通过\n")

    # ---- 核心验收：大学英语四级词汇.docx 不应包含数学 ----
    print("--- 核心验收: 大学英语四级词汇.docx ---")
    accept_folders = [
        Path("D:\\学校文件\\英语"),
        Path("D:\\学校文件\\英语\\大学英语"),
        Path("D:\\学校文件\\数学"),
    ]
    accept_matches = get_best_matches("大学英语四级词汇.docx", accept_folders)
    for p, s in accept_matches:
        print(f"    {p} ({s})")
    assert len(accept_matches) >= 2, "英语和大学英语都应匹配"
    math_accept = [p for p, _ in accept_matches if "数学" in str(p)]
    assert len(math_accept) == 0, "数学不应进入候选"
    print("  [OK] 核心验收通过\n")

    print("英文测试集全部通过.")
