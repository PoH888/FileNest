"""
matcher.py 独立测试

从源文件 ``if __name__ == "__main__":`` 块迁移而来。
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，以便导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.matcher import (  # noqa: E402
    get_best_matches,
    normalize_filename,
    is_short_name,
    has_clear_winner,
    has_family_overlap,
    extract_family_group,
    build_family_tree,
)


def test_cn_strong_match() -> None:
    """测试 1: 强匹配 —— 高等数学期末考试 → 高等数学"""
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

    matches = get_best_matches("高等数学期末考试.pdf", folders)
    top_folder, top_score = matches[0]
    assert top_folder.name == "高等数学", "最佳匹配应为 '高等数学'"
    assert top_score >= 85, "高等数学 → 高等数学 应达到阈值"


def test_cn_keyword_overlap() -> None:
    """测试 2: keyword overlap 中文匹配 —— 高数 → 高等数学"""
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

    matches = get_best_matches("高数期末.pdf", folders)
    assert len(matches) >= 1, "应至少匹配到'高等数学'"
    assert matches[0][0].name == "高等数学", "首位应为高等数学"


def test_no_match_irrelevant() -> None:
    """测试 3: 无匹配 —— 无关文件名"""
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

    matches = get_best_matches("xyzabcxyz_random_file.exe", folders)
    assert len(matches) == 0, "无关文件名应无匹配"


def test_keyword_overlap_multi_dirs() -> None:
    """测试 4: keyword overlap 跨目录 —— python 多目录"""
    base = Path("D:\\学校文件")
    code_folders = [
        base / "计算机" / "Python编程",
        Path("D:\\学校文件\\python课"),
        Path("D:\\学校文件\\python期末大作业"),
    ]
    code_matches = get_best_matches("Python_project_代码.zip", code_folders, threshold=50)
    assert len(code_matches) >= 2, "应找到多个 python 相关候选"


def test_normalize_filename() -> None:
    """测试 5: 规范化预处理"""
    assert normalize_filename("Hello-World_Test.TXT") == "hello world test"
    assert normalize_filename("ABC.DEF.GH") == "abc.def"
    assert normalize_filename("simple") == "simple"


def test_short_name_protection() -> None:
    """测试 6: 极短文件名保护"""
    assert is_short_name("a.txt") is True, "长度=1 应视为极短"
    assert is_short_name("ab.pdf") is False, "长度=2 不应拦截"


def test_cn_short_name() -> None:
    """测试 7: 短文件名折中 —— 高数 仍能匹配"""
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
    get_best_matches("高数.pdf", folders)  # 不报错即通过


# -----------------------------------------------------------------------
# has_clear_winner
# -----------------------------------------------------------------------

def test_clear_winner_big_lead() -> None:
    """测试 9: has_clear_winner — 领先 26 分"""
    big_lead = [
        (Path("D:\\学校文件\\python课"), 100),
        (Path("D:\\学校文件\\python期末大作业"), 74),
    ]
    assert has_clear_winner(big_lead) is True, "100 vs 74 → True"


def test_clear_winner_close() -> None:
    """测试 10: has_clear_winner — 接近 (88 vs 86)"""
    close = [
        (Path("D:\\学校文件\\英语"), 88),
        (Path("D:\\学校文件\\大学英语"), 86),
    ]
    assert has_clear_winner(close) is False, "88 vs 86 → False"


def test_clear_winner_95_direct() -> None:
    """测试 11: has_clear_winner — ≥95 直接赢"""
    high = [
        (Path("D:\\学校文件\\数学"), 98),
        (Path("D:\\学校文件\\高等数学"), 90),
    ]
    assert has_clear_winner(high) is True, "98 → True"


def test_clear_winner_single_candidate() -> None:
    """测试 12: has_clear_winner — 唯一候选"""
    single = [(Path("D:\\学校文件\\数学"), 75)]
    assert has_clear_winner(single) is True


def test_clear_winner_empty() -> None:
    """测试 13: has_clear_winner — 空列表"""
    assert has_clear_winner([]) is False


def test_clear_winner_tie_100() -> None:
    """测试 13b: has_clear_winner — 100 平局"""
    tied = [
        (Path("D:\\学校文件\\英语"), 100),
        (Path("D:\\学校文件\\英语\\大学英语"), 100),
    ]
    assert has_clear_winner(tied) is False, "100 vs 100 → False"


# -----------------------------------------------------------------------
# 家族重叠
# -----------------------------------------------------------------------

_ENG_FOLDERS = [
    Path("D:\\学校文件\\英语"),
    Path("D:\\学校文件\\英语\\大学英语"),
    Path("D:\\学校文件\\英语\\中学英语"),
    Path("D:\\学校文件\\英语\\小学英语"),
]


def test_family_overlap_detected() -> None:
    """测试 14: has_family_overlap — 父子同时出现"""
    eng_matches = get_best_matches("英语四级词汇.docx", _ENG_FOLDERS)
    assert has_family_overlap(eng_matches), "英语父子同时出现在列表中，应有重叠"


def test_extract_family_group() -> None:
    """测试 15: extract_family_group 分离家族/非家族"""
    eng_matches = get_best_matches("英语四级词汇.docx", _ENG_FOLDERS)
    family, standalone = extract_family_group(eng_matches)
    assert all("英语" in str(p) for p, _ in family), "家族组应只含英语相关"
    assert len(standalone) == 0, "纯英语列表不应有非家族项"


def test_build_family_tree() -> None:
    """测试 16: build_family_tree 树形结构"""
    tree_matches = get_best_matches("英语四级词汇.docx", _ENG_FOLDERS)
    tree = build_family_tree(tree_matches)
    assert len(tree) >= 1, "应至少有一个根节点"
    root_names = {p.name for p, _ in tree}
    assert "英语" in root_names, "根节点应包含'英语'"
    eng_node = next((n for n in tree if n[0].name == "英语"), None)
    assert eng_node is not None
    assert eng_node[1] is not None and len(eng_node[1]) == 3, (
        f"英语应有3个子节点，实际 {len(eng_node[1]) if eng_node[1] else 0}"
    )


# -----------------------------------------------------------------------
# 实景测试
# -----------------------------------------------------------------------

def test_python_project_clear_winner() -> None:
    """实景测试: Python_project_代码.zip 领先 26 分，不应弹窗"""
    py_folders = [
        Path("D:\\学校文件\\python课"),
        Path("D:\\学校文件\\python期末大作业"),
        Path("D:\\学校文件\\python期末大作业\\report"),
    ]
    py_matches = get_best_matches("Python_project_代码.zip", py_folders)
    assert has_clear_winner(py_matches), "100 vs 74 vs 60 → 有明显赢家"
    assert py_matches[0][0].name == "python课", "应自动归入 python课"


# -----------------------------------------------------------------------
# 英文测试集
# -----------------------------------------------------------------------

_ENG_BASE = Path("D:\\EnglishTest")


def test_en1_python_vs_java_cpp() -> None:
    """EN-1: Python_Project_Code.zip — Python vs Java/C++"""
    en1_folders = [
        _ENG_BASE / "Programming" / "Python",
        _ENG_BASE / "Programming" / "Java",
        _ENG_BASE / "Programming" / "C++",
    ]
    en1_matches = get_best_matches("Python_Project_Code.zip", en1_folders)
    assert en1_matches[0][0].name == "Python", "Python 应为第一名"
    java_in = any("Java" in str(p) for p, _ in en1_matches)
    cpp_in = any("C++" in str(p) for p, _ in en1_matches)
    assert not java_in, "Java 不应进入候选"
    assert not cpp_in, "C++ 不应进入候选"


def test_en2_machine_learning_vs_ai_web() -> None:
    """EN-2: Machine_Learning_Notes.pdf — ML vs AI vs Web"""
    en2_folders = [
        _ENG_BASE / "Artificial Intelligence",
        _ENG_BASE / "Machine Learning",
        _ENG_BASE / "Web Development",
    ]
    en2_matches = get_best_matches("Machine_Learning_Notes.pdf", en2_folders)
    assert en2_matches[0][0].name == "Machine Learning", "Machine Learning 应为第一名"
    web_dev = [p for p, _ in en2_matches if "Web Development" in str(p)]
    assert len(web_dev) == 0, "Web Development 不应进入候选列表"


def test_en3_tokyo_travel() -> None:
    """EN-3: Tokyo_Travel_Plan.xlsx — Japan/Tokyo vs Finance"""
    en3_folders = [
        _ENG_BASE / "Japan Travel",
        _ENG_BASE / "Tokyo Trip",
        _ENG_BASE / "Finance",
    ]
    en3_matches = get_best_matches("Tokyo_Travel_Plan.xlsx", en3_folders)
    assert en3_matches[0][0].name == "Tokyo Trip", "Tokyo Trip 应为第一名"
    assert len(en3_matches) >= 2, "至少 Japan Travel 也应在候选"
    finance_in = any("Finance" in str(p) for p, _ in en3_matches)
    assert not finance_in, "Finance 不应进入候选"


def test_en4_english_vocabulary() -> None:
    """EN-4: University_English_CET4_Vocabulary.docx — English vs Math"""
    en4_folders = [
        _ENG_BASE / "English",
        _ENG_BASE / "College English",
        _ENG_BASE / "Mathematics",
    ]
    en4_matches = get_best_matches("University_English_CET4_Vocabulary.docx", en4_folders)
    assert en4_matches[0][0].name == "English", "English 应为第一名"
    assert len(en4_matches) >= 2, "至少 College English 也应在候选"
    math_in = any("Mathematics" in str(p) for p, _ in en4_matches)
    assert not math_in, "Mathematics 不应进入候选"


def test_en5_database_final() -> None:
    """EN-5: Database_Final_Project.zip — DB vs OS vs Net"""
    en5_folders = [
        _ENG_BASE / "Database",
        _ENG_BASE / "Operating Systems",
        _ENG_BASE / "Computer Networks",
    ]
    en5_matches = get_best_matches("Database_Final_Project.zip", en5_folders)
    assert en5_matches[0][0].name == "Database", "Database 应为第一名"
    assert has_clear_winner(en5_matches), "Database 应明显领先"


def test_core_acceptance_no_math() -> None:
    """核心验收: 大学英语四级词汇.docx 不应包含数学"""
    accept_folders = [
        Path("D:\\学校文件\\英语"),
        Path("D:\\学校文件\\英语\\大学英语"),
        Path("D:\\学校文件\\数学"),
    ]
    accept_matches = get_best_matches("大学英语四级词汇.docx", accept_folders)
    assert len(accept_matches) >= 2, "英语和大学英语都应匹配"
    math_accept = [p for p, _ in accept_matches if "数学" in str(p)]
    assert len(math_accept) == 0, "数学不应进入候选"


# -----------------------------------------------------------------------
# 回归测试：短拉丁词不应虚假匹配
# -----------------------------------------------------------------------

def test_en_math_not_music() -> None:
    """Math 不应模糊匹配到 Music（fuzzywuzzy sum-normalization 虚高）"""
    folders = [
        Path("D:\\School") / "Music",
        Path("D:\\School") / "Math",
        Path("D:\\School") / "English",
    ]
    matches = get_best_matches("Math_Homework.pdf", folders)
    # Math 应排第一
    assert matches[0][0].name == "Math", "Math 应为第一名"
    # Music 不应出现在候选列表中
    music = [p for p, _ in matches if "Music" in str(p)]
    assert len(music) == 0, "Music 不应进入候选"


def test_en_short_no_false_positive() -> None:
    """短拉丁词不应模糊匹配无关文件夹"""
    folders = [
        Path("D:\\Code") / "Git",
        Path("D:\\Code") / "Go",
        Path("D:\\Code") / "DotNet",
    ]
    matches = get_best_matches("git_tutorial.pdf", folders)
    assert matches[0][0].name == "Git", "Git 应为第一名"
    dotnet = [p for p, _ in matches if "DotNet" in str(p)]
    assert len(dotnet) == 0, "DotNet 不应因 git 含字母 t 而虚高匹配"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
