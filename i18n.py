"""
FileNest — 国际化模块

用法：
    from i18n import tr, set_language, get_language, get_available_languages

    tr("status_ready")              # → "● 就绪" 或 "● Ready"
    tr("confirm_msg", fname, score)  # → format-string 传参
"""

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# 字符串字典
# ---------------------------------------------------------------------------

_STRINGS: Dict[str, Dict[str, str]] = {
    "zh": {
        # ---- 状态栏 ----
        "status_ready": "● 就绪",
        "status_monitoring": "● 监控中",
        "status_paused": "⏸ 已暂停",
        # ---- 投放区域 ----
        "drop_frame": "投放文件",
        "drop_hint": "拖拽文件到此处，自动归类",
        "browse_btn": "选择文件...",
        # ---- 监控 ----
        "monitor_checkbox": "监控下载文件夹",
        "monitor_choose_btn": "选择监控文件夹",
        "monitor_not_set": "（未设置）",
        # ---- 最近操作 ----
        "history_frame": "最近操作",
        "history_btn_open": "📂打开",
        "history_btn_up": "⬆上一级",
        "history_btn_undo": "↺撤销",
        # ---- 底部栏 ----
        "bottom_undo": "撤销上次移动",
        "bottom_settings": "设置",
        "bottom_exit": "退出",
        "bottom_guide": "📖 用户指南",
        # ---- 候选窗口 ----
        "candidate_title": "请选择存放位置",
        "candidate_desc": "找到多个可能位置，点击目标即可放入。",
        "candidate_more": "... 还有 {} 个低分候选未显示",
        "candidate_ok": "确定",
        "candidate_cancel": "取消",
        # ---- 树形候选窗口 ----
        "tree_title": "请选择具体存放位置",
        "tree_desc": "检测到父子文件夹同时匹配，请选择具体存放位置：",
        # ---- 通知窗口 ----
        "notif_title": "检测到新文件：",
        "notif_sort_btn": "立即归类",
        "notif_ignore_btn": "忽略",
        # ---- 同名冲突 ----
        "collision_title": "文件已存在",
        "collision_msg": "目标文件夹已存在同名文件：\n{}",
        "collision_overwrite": "覆盖",
        "collision_keep_both": "保留两者",
        "collision_skip": "跳过",
        # ---- 设置窗口 ----
        "settings_title": "设置",
        "settings_tab_categories": "分类目录",
        "settings_tab_matching": "匹配策略",
        "settings_tab_monitor": "监控目录",
        "settings_tab_ignore": "忽略名单",
        "settings_tab_appearance": "外观与行为",
        "settings_add_root": "添加根目录",
        "settings_add_root_title": "选择分类根目录",
        "settings_remove_root": "移除选中",
        "settings_depth": "扫描深度：",
        "settings_threshold": "自动移动阈值：",
        "settings_mid_range": "中等相似度处理：",
        "settings_always_ask": "总是询问",
        "settings_auto_prefix": "在",
        "settings_parent_weight": "启用父文件夹加权",
        "settings_short_name": "短文件名保护（≤2字符）",
        "settings_monitor_path": "当前监控路径：",
        "settings_monitor_change": "更改监控文件夹",
        "settings_monitor_enable": "开启监控",
        "settings_ignore_hint": "关键词用分号分隔，支持通配符 * 和 ?：",
        "settings_ignore_example": "示例：备份;临时;*缓存*",
        "settings_open_after": "移动后自动打开目标文件夹",
        "settings_log_path": "日志路径：{}",
        "settings_log_path_unknown": "（未知）",
        "settings_log_btn": "打开日志文件夹",
        "settings_save": "保存",
        "settings_cancel": "取消",
        # ---- 文件处理 ----
        "file_not_found": "文件不存在: {}",
        "file_name_too_short": "文件名过短，请手动归类: {}",
        "file_manual_title": "请手动选择目标文件夹",
        "no_folders": "未找到合适位置（无可用分类目录）",
        "no_match": "未找到合适位置: {}",
        "confirm_title": "确认移动",
        "confirm_msg": "将文件移动到：\n{}\n\n相似度: {}%",
        "error_title": "处理错误",
        "error_msg": "处理文件时发生异常:\n{}",
        "move_failed_title": "移动失败",
        "move_failed_msg": "无法移动文件:\n{}",
        "move_error_title": "移动异常",
        "move_error_msg": "移动文件时发生异常:\n{}",
        "skipped": "已跳过: {}",
        # ---- 错误/提示 ----
        "monitor_warning_title": "监控",
        "monitor_warning_msg": "请先选择监控文件夹",
        "monitor_error_title": "监控",
        "monitor_error_msg": "目录不存在:\n{}",
        "undo_title": "撤销",
        "nothing_to_undo": "没有可撤销的操作",
        "rollback_at_root": "文件已在根目录，无法继续回退",
        "rollback_not_in_tree": "文件不在分类树内",
        "rollback_failed": "回退失败",
        "undo_failed": "撤销失败",
        "file_gone": "文件不存在",
        # ---- 用户指南 ----
        "guide_placeholder": (
            "FileNest - 文件归巢  使用说明\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "📂 投放文件\n"
            "  拖拽文件到「投放文件」区域，或点击「选择文件」按钮，"
            "程序会自动匹配分类目录并移动。\n\n"
            "🔄 归类流程\n"
            "  1. 选择/拖入文件\n"
            "  2. 程序按文件名匹配最佳分类\n"
            "  3. 确认后自动移动到对应文件夹\n\n"
            "⚙ 设置\n"
            "  · 分类目录 — 添加/删除分类根目录\n"
            "  · 匹配策略 — 调整相似度阈值、父文件夹加权等\n"
            "  · 监控目录 — 监控下载文件夹自动归类\n"
            "  · 忽略名单 — 设置不想自动处理的文件名\n\n"
            "🌐 语言切换\n"
            "  点击底部 Language 按钮，随时切换中/英文\n\n"
            "↩ 最近操作\n"
            "  每条操作记录都有「打开」「上一级」和「撤销」按钮\n\n"
            "📁 监控模式\n"
            "  开启后自动处理监控目录中的新文件并分类"
        ),
    },
    "en": {
        # ---- Status bar ----
        "status_ready": "● Ready",
        "status_monitoring": "● Monitoring",
        "status_paused": "⏸ Paused",
        # ---- Drop zone ----
        "drop_frame": "Drop File",
        "drop_hint": "Drag files here to auto-sort",
        "browse_btn": "Browse...",
        # ---- Monitor ----
        "monitor_checkbox": "Monitor Download Folder",
        "monitor_choose_btn": "Choose Folder",
        "monitor_not_set": "(Not set)",
        # ---- History ----
        "history_frame": "History",
        "history_btn_open": "📂Open",
        "history_btn_up": "⬆Up",
        "history_btn_undo": "↺Undo",
        # ---- Bottom bar ----
        "bottom_undo": "Undo Last Move",
        "bottom_settings": "Settings",
        "bottom_exit": "Exit",
        "bottom_guide": "📖 User Guide",
        # ---- CandidateWindow ----
        "candidate_title": "Select a destination",
        "candidate_desc": "Multiple folders matched, click one to move.",
        "candidate_more": "... {} more low-score candidates hidden",
        "candidate_ok": "OK",
        "candidate_cancel": "Cancel",
        # ---- TreeCandidateWindow ----
        "tree_title": "Select a location",
        "tree_desc": "Parent and child folders both matched, please select:",
        # ---- NotificationWindow ----
        "notif_title": "New file detected:",
        "notif_sort_btn": "Sort Now",
        "notif_ignore_btn": "Ignore",
        # ---- CollisionDialog ----
        "collision_title": "File Exists",
        "collision_msg": "A file with the same name already exists:\n{}",
        "collision_overwrite": "Overwrite",
        "collision_keep_both": "Keep Both",
        "collision_skip": "Skip",
        # ---- Settings ----
        "settings_title": "Settings",
        "settings_tab_categories": "Categories",
        "settings_tab_matching": "Matching",
        "settings_tab_monitor": "Monitor",
        "settings_tab_ignore": "Ignore List",
        "settings_tab_appearance": "Appearance",
        "settings_add_root": "Add Root",
        "settings_add_root_title": "Select category root",
        "settings_remove_root": "Remove",
        "settings_depth": "Scan Depth:",
        "settings_threshold": "Auto-move Threshold:",
        "settings_mid_range": "Medium Similarity:",
        "settings_always_ask": "Always ask",
        "settings_auto_prefix": "between",
        "settings_parent_weight": "Enable Parent Weight",
        "settings_short_name": "Short Name Protection (≤2 chars)",
        "settings_monitor_path": "Current Monitor Path:",
        "settings_monitor_change": "Change Folder",
        "settings_monitor_enable": "Enable Monitoring",
        "settings_ignore_hint": "Separate keywords by semicolons, support * and ?:",
        "settings_ignore_example": "Example: backup;temp;*cache*",
        "settings_open_after": "Open target folder after move",
        "settings_log_path": "Log Path: {}",
        "settings_log_path_unknown": "(Unknown)",
        "settings_log_btn": "Open Log Folder",
        "settings_save": "Save",
        "settings_cancel": "Cancel",
        # ---- File processing ----
        "file_not_found": "File not found: {}",
        "file_name_too_short": "File name too short, please sort manually: {}",
        "file_manual_title": "Please select a destination folder",
        "no_folders": "No suitable folder found (no categories available)",
        "no_match": "No match found: {}",
        "confirm_title": "Confirm Move",
        "confirm_msg": "Move file to:\n{}\n\nMatch: {}%",
        "error_title": "Error",
        "error_msg": "Error processing file:\n{}",
        "move_failed_title": "Move Failed",
        "move_failed_msg": "Cannot move file:\n{}",
        "move_error_title": "Move Error",
        "move_error_msg": "Error moving file:\n{}",
        "skipped": "Skipped: {}",
        # ---- Error/info dialogs ----
        "monitor_warning_title": "Monitor",
        "monitor_warning_msg": "Please select a monitor folder first",
        "monitor_error_title": "Monitor",
        "monitor_error_msg": "Directory not found:\n{}",
        "undo_title": "Undo",
        "nothing_to_undo": "Nothing to undo",
        "rollback_at_root": "File is already at root, cannot rollback further",
        "rollback_not_in_tree": "File is not inside the category tree",
        "rollback_failed": "Rollback Failed",
        "undo_failed": "Undo Failed",
        "file_gone": "File not found",
        # ---- User guide ----
        "guide_placeholder": (
            "FileNest — User Guide\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "📂 Drop Zone\n"
            '  Drag files onto the drop area, or click "Browse..." '
            "to select files. The app will auto-match and move them.\n\n"
            "🔄 Workflow\n"
            "  1. Drop / select a file\n"
            "  2. App matches by file name\n"
            "  3. Confirm and auto-move to the target folder\n\n"
            "⚙ Settings\n"
            "  · Categories — add/remove root folders\n"
            "  · Matching — adjust threshold, parent weight, etc.\n"
            "  · Monitor — watch a download folder for new files\n"
            "  · Ignore List — skip certain file names\n\n"
            "🌐 Language\n"
            "  Click the Language button in the bottom bar\n\n"
            "↩ History\n"
            '  Each record has "Open", "Up" and "Undo" buttons\n\n'
            "📁 Monitor Mode\n"
            "  Auto-detects and sorts new files in the monitored folder"
        ),
    },
}

# ---------------------------------------------------------------------------
# 当前语言
# ---------------------------------------------------------------------------

_current_language: str = "zh"
_available_languages: List[str] = ["zh", "en"]


def get_available_languages() -> List[str]:
    return list(_available_languages)


def get_language() -> str:
    return _current_language


def set_language(lang: str) -> None:
    global _current_language
    if lang in _STRINGS:
        _current_language = lang


def tr(key: str, *args: str) -> str:
    """获取当前语言下 key 对应的翻译文本。

    若 key 不存在则返回 ``⧼key⧽`` 作为占位。
    支持 ``args`` 对 ``{}`` 占位符做顺序格式化。
    """
    value = _STRINGS.get(_current_language, {}).get(key)
    if value is None:
        return f"⧼{key}⧽"
    if args:
        return value.format(*args)
    return value
