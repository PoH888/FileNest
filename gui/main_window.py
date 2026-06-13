"""
FileNest — 主窗口模块

包含拖拽支持函数、FileNestApp 主窗口类。
"""

import logging
import os
import re
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core import config_manager
from core import file_monitor
from core import file_mover
from core import folder_scanner
from core import matcher
from gui.dialogs import CollisionDialog, CandidateWindow, TreeCandidateWindow
from gui.notification import NotificationWindow
from gui.settings_window import SettingsWindow
from core.i18n import tr, set_language, get_language, get_available_languages
from core.utils import create_tray_icon, destroy_tray_icon, ensure_icon_ico, get_big_ico_path, get_ico_path

# ---------------------------------------------------------------------------
# TkinterDnD — 原生拖拽支持
# ---------------------------------------------------------------------------

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _HAS_DND = True
except ImportError:
    _HAS_DND = False

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------


def _get_logger() -> logging.Logger:
    return logging.getLogger("FileNest")


# ---------------------------------------------------------------------------
# 拖拽支持
# ---------------------------------------------------------------------------

def _register_drop_target(widget: tk.Widget, callback: Callable[[str], None]) -> None:
    """使用 TkinterDnD 将某个 widget 注册为拖拽目标。"""
    if not _HAS_DND:
        _get_logger().warning("TkinterDnD 未安装，拖拽功能不可用")
        return

    widget.drop_target_register(DND_FILES)
    widget.dnd_bind("<<Drop>>", lambda e: _on_dnd_drop(e, callback))


def _on_dnd_drop(event: tk.Event, callback: Callable[[str], None]) -> None:
    """解析 TkinterDnD 拖拽事件中的文件路径。"""
    raw = event.data
    if not raw:
        return
    # tkinterdnd2 返回的路径用 {} 包裹带空格路径，可传多个文件
    files = re.findall(r'\{([^}]*)\}|(\S+)', raw)
    for matched in files:
        path = matched[0] or matched[1]
        if path:
            callback(path.strip())


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

WINDOW_TITLE: str = "FileNest - 文件归巢"
WINDOW_WIDTH: int = 450
WINDOW_HEIGHT: int = 350
HISTORY_MAX: int = 10
HISTORY_DISPLAY: int = 5


# ===================================================================
# 主窗口
# ===================================================================

class FileNestApp:
    """FileNest 主窗口。

    包含标题与状态栏、投放区域、监控开关、最近操作记录、底部按钮栏。
    """

    def __init__(self) -> None:
        # ---- 根窗口 ----
        if _HAS_DND:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()

        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(400, 320)

        # 居中
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - WINDOW_WIDTH) // 2
        y = (sh - WINDOW_HEIGHT) // 2
        self.root.geometry(f"+{x}+{y}")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ---- 窗口图标 ----
        self._set_window_icon()

        # ---- 配置 ----
        self.config = config_manager.load_config()
        set_language(self.config.get("language", "zh"))

        # ---- 状态 ----
        self.history: List[Dict[str, str]] = []
        self.monitor_active: bool = False
        self._title_click_count: int = 0
        self._observer: Optional[Any] = None
        self._tray_shown: bool = False
        self.event_queue: "queue.Queue" = __import__("queue").Queue()

        # ---- 构建 UI ----
        self._build_ui()

        # ---- 定时轮询队列 ----
        self._poll_queue()

        # ---- 开机自动监控 ----
        self._auto_start_monitor()

    # ================================================================
    # 界面构建
    # ================================================================

    def _build_ui(self) -> None:
        """构建完整界面布局。"""
        self._build_status_bar()
        self._build_drop_zone()
        self._build_monitor_section()
        self._build_bottom_bar()
        self._build_history_section()

    # ---- 状态栏 ----

    def _build_status_bar(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=10, pady=(8, 0))

        self._title_label = ttk.Label(bar, text=WINDOW_TITLE, font=("", 11, "bold"))
        self._title_label.pack(side=tk.LEFT)
        self._title_label.bind("<Button-1>", self._on_title_click)

        self.status_label = ttk.Label(bar, font=("", 9))
        self.status_label.pack(side=tk.RIGHT)
        self._update_status("idle")

    def _update_status(self, state: str) -> None:
        """更新状态栏显示。"""
        if state == "running":
            self.status_label.config(text=tr("status_monitoring"), foreground="green")
        elif state == "paused":
            self.status_label.config(text=tr("status_paused"), foreground="red")
        else:
            self.status_label.config(text=tr("status_ready"), foreground="green")

    # ---- 标题彩蛋 ----

    def _on_title_click(self, event: tk.Event) -> None:
        """🎄 隐藏彩蛋：连续点击标题 7 次。"""
        self._title_click_count += 1
        if self._title_click_count >= 7:
            self._title_click_count = 0
            messagebox.showinfo("🎉", "感谢女朋友大人的支持！")

    # ---- 投放区域 ----

    def _build_drop_zone(self) -> None:
        self._drop_frame = ttk.LabelFrame(self.root, text=tr("drop_frame"), padding=10)
        frame = self._drop_frame
        frame.pack(fill=tk.X, padx=10, pady=(8, 0))

        self.drop_area = tk.Frame(frame, relief="solid", bd=2,
                                  bg="#f5f5f5", height=60)
        self.drop_area.pack(fill=tk.X)
        self.drop_area.pack_propagate(False)

        if _HAS_DND:
            _register_drop_target(self.drop_area, self._on_drop_path)

        self._drop_hint_label = ttk.Label(self.drop_area, text=tr("drop_hint"),
                  background="#f5f5f5", foreground="gray")
        self._drop_hint_label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

        btn_f = ttk.Frame(frame)
        btn_f.pack(pady=(5, 0))
        self._browse_btn = ttk.Button(btn_f, text=tr("browse_btn"),
                   command=self._select_file)
        self._browse_btn.pack()

    def _on_drop_path(self, file_path_str: str) -> None:
        """处理拖拽投放。"""
        try:
            p = Path(file_path_str.strip())
            if p.is_file():
                self.root.after(0, self._process_file, p)
        except Exception as e:
            _get_logger().error("拖拽处理异常: %s", e)

    # ---- 监控开关 ----

    def _build_monitor_section(self) -> None:
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.X, padx=10, pady=(6, 0))

        self.monitor_var = tk.BooleanVar(value=self.config.get("monitor_enabled", False))
        self._monitor_checkbox = ttk.Checkbutton(frame, text=tr("monitor_checkbox"),
                        variable=self.monitor_var,
                        command=self._toggle_monitor)
        self._monitor_checkbox.pack(side=tk.LEFT)

        self._monitor_choose_btn = ttk.Button(frame, text=tr("monitor_choose_btn"),
                   command=self._select_monitor_dir)
        self._monitor_choose_btn.pack(side=tk.LEFT, padx=(5, 0))

        monitor_dir = self.config.get("monitor_directory", "")
        self.monitor_path_label = ttk.Label(
            frame, text=monitor_dir if monitor_dir else tr("monitor_not_set"),
            foreground="gray")
        self.monitor_path_label.pack(side=tk.LEFT, padx=(5, 0))

    # ---- 最近操作记录 ----

    def _build_history_section(self) -> None:
        self._history_frame = ttk.LabelFrame(self.root, text=tr("history_frame"), padding=5)
        frame = self._history_frame
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 0))

        self._history_canvas = tk.Canvas(frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL,
                                  command=self._history_canvas.yview)
        self._history_inner = ttk.Frame(self._history_canvas)

        self._history_inner.bind("<Configure>",
                                 lambda e: self._history_canvas.configure(
                                     scrollregion=self._history_canvas.bbox("all")))

        self._history_canvas.create_window((0, 0), window=self._history_inner,
                                            anchor=tk.NW, tags="inner")
        self._history_canvas.configure(yscrollcommand=scrollbar.set)

        self._history_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._history_canvas.bind("<Enter>", self._bind_mousewheel)
        self._history_canvas.bind("<Leave>", self._unbind_mousewheel)

    def _bind_mousewheel(self, event: tk.Event) -> None:
        try:
            self._history_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        except Exception:
            pass

    def _unbind_mousewheel(self, event: tk.Event) -> None:
        try:
            self._history_canvas.unbind_all("<MouseWheel>")
        except Exception:
            pass

    def _on_mousewheel(self, event: tk.Event) -> None:
        try:
            self._history_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    # ---- 底部按钮栏 ----

    def _build_bottom_bar(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(6, 8))

        self._undo_btn = ttk.Button(bar, text=tr("bottom_undo"),
                   command=self._undo_last_move)
        self._undo_btn.pack(side=tk.LEFT, padx=(0, 5))
        self._settings_btn = ttk.Button(bar, text=tr("bottom_settings"),
                   command=self._show_settings)
        self._settings_btn.pack(side=tk.LEFT, padx=5)

        current_lang_label = "简体中文" if get_language() == "zh" else "English"
        self._lang_btn = ttk.Menubutton(bar, text="Language",
                                        direction="above")
        lang_menu = tk.Menu(self._lang_btn, tearoff=0)
        lang_menu.add_command(label="简体中文",
                              command=lambda: self._switch_language("zh"))
        lang_menu.add_command(label="English",
                              command=lambda: self._switch_language("en"))
        self._lang_btn.config(menu=lang_menu)
        self._lang_btn.pack(side=tk.RIGHT, padx=(5, 0))

        self._guide_btn = ttk.Button(bar, text=tr("bottom_guide"),
                                     command=self._show_guide)
        self._guide_btn.pack(side=tk.RIGHT, padx=5)

        self._exit_btn = ttk.Button(bar, text=tr("bottom_exit"),
                   command=self._on_close)
        self._exit_btn.pack(side=tk.RIGHT, padx=(5, 0))

    # ================================================================
    # 文件选择
    # ================================================================

    def _select_file(self) -> None:
        path_str = filedialog.askopenfilename(title=tr("browse_btn"))
        if path_str:
            self._process_file(Path(path_str))

    # ================================================================
    # 文件处理流程
    # ================================================================

    def _process_file(self, file_path: Path, original_path: Optional[Path] = None) -> None:
        """完整文件处理流程。"""
        logger = _get_logger()
        orig = original_path or file_path

        try:
            if not file_path.exists():
                self._append_record(tr("file_not_found", file_path.name), None)
                return

            file_name = file_path.name

            # 短文件名保护
            if matcher.is_short_name(file_name):
                self._append_record(tr("file_name_too_short", file_name), None)
                folder_str = filedialog.askdirectory(title=tr("file_manual_title"))
                if folder_str:
                    self._do_move(file_path, Path(folder_str), orig)
                return

            # 扫描根目录
            roots_data = folder_scanner.scan_all_roots()
            all_folders: List[Path] = []
            for _root, folders in roots_data:
                all_folders.extend(folders)

            if not all_folders:
                self._append_record(tr("no_folders"), None)
                folder_str = filedialog.askdirectory(title=tr("file_manual_title"))
                if folder_str:
                    self._do_move(file_path, Path(folder_str), orig)
                return

            # 模糊匹配
            threshold = self.config.get("auto_threshold", 85)
            parent_wt = self.config.get("parent_weight_enabled", True)
            matches = matcher.get_best_matches(
                file_name, all_folders, threshold=threshold, parent_weight=parent_wt)

            # 明显领先 → 直接自动归类
            if matcher.has_clear_winner(matches):
                self._do_move(file_path, matches[0][0], orig)
                return

            # 家族重叠 → 树形选择窗口
            if matcher.has_family_overlap(matches):
                family_group, standalone = matcher.extract_family_group(matches)
                tree_matches = family_group + standalone
                w = TreeCandidateWindow(self.root, file_name, tree_matches)
                if w.selected:
                    self._do_move(file_path, w.selected, orig)
                return

            # 分支决策
            if not matches:
                self._append_record(tr("no_match", file_name), None)
                folder_str = filedialog.askdirectory(title=tr("file_manual_title"))
                if folder_str:
                    self._do_move(file_path, Path(folder_str), orig)

            elif len(matches) == 1 and matches[0][1] >= threshold:
                self._do_move(file_path, matches[0][0], orig)

            else:
                high_candidates = [(p, s) for p, s in matches if s >= threshold]
                if len(high_candidates) >= 2:
                    w = CandidateWindow(self.root, file_name, high_candidates)
                    if w.selected:
                        self._do_move(file_path, w.selected, orig)
                else:
                    best_path, best_score = matches[0]
                    mid_auto = self.config.get("mid_range_auto", False)
                    if mid_auto:
                        self._do_move(file_path, best_path, orig)
                    else:
                        ok = messagebox.askyesno(
                            tr("confirm_title"),
                            tr("confirm_msg", str(best_path), str(best_score)),
                            parent=self.root)
                        if ok:
                            self._do_move(file_path, best_path, orig)

        except Exception as e:
            logger.error("文件处理异常: %s", e)
            messagebox.showerror(tr("error_title"),
                                 tr("error_msg", str(e)), parent=self.root)

    # ================================================================
    # 执行移动
    # ================================================================

    def _do_move(self, src: Path, dst_dir: Path, original_path: Path) -> bool:
        """执行文件移动，处理冲突。"""
        logger = _get_logger()

        try:
            dst = dst_dir / src.name
            strategy = "keep_both"
            if dst.exists():
                dialog = CollisionDialog(self.root, src.name)
                chosen = dialog.result
                if chosen is None or chosen == "skip":
                    self._append_record(tr("skipped", src.name), None)
                    return False
                strategy = chosen

            result = file_mover.move_file(src, dst_dir,
                                          collision_strategy=strategy)
            if result is None:
                messagebox.showerror(tr("move_failed_title"),
                                     tr("move_failed_msg", str(src)), parent=self.root)
                return False

            self._add_history(src, result, original_path)

            if self.config.get("open_folder_after_move", False):
                try:
                    os.startfile(str(result.parent))
                except Exception:
                    pass

            return True

        except Exception as e:
            logger.error("移动异常: %s", e)
            messagebox.showerror(tr("move_error_title"),
                                 tr("move_error_msg", str(e)), parent=self.root)
            return False

    # ================================================================
    # 历史记录
    # ================================================================

    def _add_history(self, src: Path, dst: Path,
                     original_path: Path) -> None:
        """添加一条操作记录并刷新显示。"""
        record: Dict[str, str] = {
            "src": str(src),
            "current": str(dst),
            "original_path": str(original_path),
        }
        self.history.insert(0, record)
        if len(self.history) > HISTORY_MAX:
            self.history.pop()
        self._refresh_history()

    def _append_record(self, message: str, _unused: Any = None) -> None:
        """在最近操作区域追加一条纯文本消息。"""
        record_frame = ttk.Frame(self._history_inner)
        record_frame.pack(fill=tk.X, padx=2, pady=1,
                          before=self._history_inner.winfo_children()[0]
                          if self._history_inner.winfo_children() else None)

        ttk.Label(record_frame, text=f"  {message}",
                  font=("Microsoft YaHei", 8)).pack(anchor=tk.W)

    def _refresh_history(self) -> None:
        """刷新最近操作记录显示区域。"""
        for w in self._history_inner.winfo_children():
            w.destroy()

        display = self.history[:HISTORY_DISPLAY]
        for idx, rec in enumerate(display):
            current_path = Path(rec["current"])
            original_path = Path(rec["original_path"])

            record_frame = ttk.Frame(self._history_inner)
            record_frame.pack(fill=tk.X, padx=2, pady=3)

            line1 = ttk.Frame(record_frame)
            line1.pack(fill=tk.X)
            ttk.Label(line1, text=f"✅ {current_path.name} →",
                      font=("Microsoft YaHei", 8)).pack(anchor=tk.W)

            path_label = ttk.Label(record_frame, text=rec['current'],
                                   font=("Microsoft YaHei", 8),
                                   anchor=tk.W)
            path_label.pack(fill=tk.X)

            line2 = ttk.Frame(record_frame)
            line2.pack(anchor=tk.W, pady=(2, 0))

            ttk.Button(line2, text=tr("history_btn_open"),
                       command=lambda p=current_path: self._open_in_explorer(p),
                       width=8).pack(side=tk.LEFT, padx=(0, 4))
            ttk.Button(line2, text=tr("history_btn_up"),
                       command=lambda i=idx: self._rollback(i),
                       width=8).pack(side=tk.LEFT, padx=4)
            ttk.Button(line2, text=tr("history_btn_undo"),
                       command=lambda i=idx: self._undo(i),
                       width=8).pack(side=tk.LEFT, padx=4)

    # ================================================================
    # 回退 / 撤销
    # ================================================================

    def _rollback(self, index: int) -> None:
        """回退：将文件向上一级文件夹移动。"""
        try:
            record = self.history[index]
            current_path = Path(record["current"])

            if not current_path.exists():
                messagebox.showerror(tr("rollback_failed"), tr("file_gone"), parent=self.root)
                return

            roots = [Path(r).resolve()
                     for r in self.config.get("root_directories", [])]
            parent = current_path.parent.resolve()

            root_dir: Optional[Path] = None
            for root in roots:
                if parent == root:
                    messagebox.showinfo(tr("undo_title"), tr("rollback_at_root"),
                                        parent=self.root)
                    return
                if root in parent.parents:
                    root_dir = root
                    break

            if root_dir is None:
                messagebox.showerror(tr("rollback_failed"), tr("rollback_not_in_tree"),
                                     parent=self.root)
                return

            result = file_mover.rollback_file(current_path, root_dir)
            if result:
                record["current"] = str(result)
                _get_logger().info("回退成功: %s → %s", current_path, result)
                self._refresh_history()

        except Exception as e:
            _get_logger().error("回退异常: %s", e)
            messagebox.showerror(tr("rollback_failed"), str(e), parent=self.root)

    def _undo(self, index: int) -> None:
        """撤销：将文件移回原始来源。"""
        try:
            record = self.history[index]
            current_path = Path(record["current"])
            original_path = Path(record["original_path"])

            if not current_path.exists():
                messagebox.showerror(tr("undo_failed"), tr("file_gone"), parent=self.root)
                return

            result = file_mover.undo_file(current_path, original_path)
            if result:
                _get_logger().info("撤销成功: %s → %s", current_path, result)
                self.history.pop(index)
                self._refresh_history()

        except Exception as e:
            _get_logger().error("撤销异常: %s", e)
            messagebox.showerror(tr("move_error_title"), str(e), parent=self.root)

    def _undo_last_move(self) -> None:
        """撤销最近一次移动。"""
        if not self.history:
            messagebox.showinfo(tr("undo_title"), tr("nothing_to_undo"), parent=self.root)
            return
        self._undo(0)

    # ================================================================
    # 监控开关
    # ================================================================

    def _select_monitor_dir(self) -> None:
        d = filedialog.askdirectory(title=tr("monitor_choose_btn"))
        if d:
            self.config["monitor_directory"] = d
            self.monitor_path_label.config(text=d, foreground="black")
            config_manager.save_config(self.config)

    def _toggle_monitor(self) -> None:
        if self.monitor_var.get():
            self._start_monitor()
        else:
            self._stop_monitor()

    def _auto_start_monitor(self) -> None:
        if (self.config.get("monitor_enabled", False)
                and self.config.get("monitor_directory")):
            self.monitor_var.set(True)
            self._start_monitor()

    def _start_monitor(self) -> None:
        monitor_dir = self.config.get("monitor_directory", "")
        if not monitor_dir:
            messagebox.showwarning(tr("monitor_warning_title"),
                                   tr("monitor_warning_msg"), parent=self.root)
            self.monitor_var.set(False)
            return

        dir_path = Path(monitor_dir)
        if not dir_path.is_dir():
            messagebox.showerror(tr("monitor_error_title"),
                                 tr("monitor_error_msg", monitor_dir), parent=self.root)
            self.monitor_var.set(False)
            return

        self._observer = file_monitor.start_monitor(dir_path, self.event_queue)
        if self._observer:
            self.monitor_active = True
            self._update_status("running")
            _get_logger().info("监控已启动: %s", monitor_dir)
        else:
            self.monitor_var.set(False)
            self._update_status("paused")

    def _stop_monitor(self, update_status: bool = True) -> None:
        if self._observer:
            file_monitor.stop_monitor(self._observer)
            self._observer = None
        self.monitor_active = False
        if update_status:
            self._update_status("paused")
        _get_logger().info("监控已停止")

    # ================================================================
    # 队列轮询
    # ================================================================

    def _poll_queue(self) -> None:
        """定时从事件队列取出文件事件并弹出通知窗口。"""
        try:
            events = file_monitor.drain_queue(self.event_queue)
            for ev in events:
                if ev.file_path.exists():
                    NotificationWindow(
                        self.root, ev.file_path,
                        on_classify=lambda p=ev.file_path: self._process_file(p, p),
                    )
        except Exception as e:
            _get_logger().error("队列轮询异常: %s", e)

        self.root.after(100, self._poll_queue)

    # ================================================================
    # 辅助功能
    # ================================================================

    def _set_window_icon(self) -> None:
        """设置窗口/任务栏图标为 FileNest_big.ico（Picture_big.png 图案），
        系统托盘仍保持 Picture_small.png。"""
        # 优先使用 FileNest_big.ico（大图标，任务栏 + Alt+Tab 清晰）
        try:
            big_ico = get_big_ico_path()
            if big_ico and big_ico.exists():
                self.root.iconbitmap(str(big_ico))
                _get_logger().info("窗口图标已设置 (big .ico): %s", big_ico)
                return
        except Exception as e:
            _get_logger().warning("big iconbitmap 失败: %s", e)

        # 回退：使用 FileNest.ico（小图标）
        try:
            ico_path = ensure_icon_ico()
            if ico_path and ico_path.exists():
                self.root.iconbitmap(str(ico_path))
                _get_logger().info("窗口图标已设置 (small .ico): %s", ico_path)
                return
        except Exception as e:
            _get_logger().warning("small iconbitmap 失败: %s", e)

        # 最终回退：使用 PNG 通过 iconphoto
        try:
            from PIL import Image, ImageTk
            png_path = get_ico_path().parent / "Picture_small.png"
            if png_path.exists():
                img = ImageTk.PhotoImage(Image.open(png_path))
                self.root.iconphoto(True, img)
                self._icon_ref = img
                _get_logger().info("窗口图标已设置 (PNG): %s", png_path)
        except Exception as e:
            _get_logger().warning("无法设置窗口图标: %s", e)

    def _open_in_explorer(self, file_path: Path) -> None:
        try:
            import subprocess
            subprocess.Popen(['explorer', '/select,', str(file_path)])
        except Exception as e:
            _get_logger().error("打开资源管理器失败: %s", e)

    # ================================================================
    # 语言切换与用户指南
    # ================================================================

    def _switch_language(self, code: str) -> None:
        set_language(code)
        self.config["language"] = code
        config_manager.save_config(self.config)
        self._refresh_texts()

    def _refresh_texts(self) -> None:
        """即时刷新主窗口所有可见文字。"""
        self._update_status("running" if self.monitor_active else "idle")

        self._drop_frame.config(text=tr("drop_frame"))
        self._drop_hint_label.config(text=tr("drop_hint"))
        self._browse_btn.config(text=tr("browse_btn"))

        self._monitor_checkbox.config(text=tr("monitor_checkbox"))
        self._monitor_choose_btn.config(text=tr("monitor_choose_btn"))
        monitor_dir = self.config.get("monitor_directory", "")
        self.monitor_path_label.config(
            text=monitor_dir if monitor_dir else tr("monitor_not_set"))

        self._history_frame.config(text=tr("history_frame"))
        self._refresh_history()

        self._undo_btn.config(text=tr("bottom_undo"))
        self._settings_btn.config(text=tr("bottom_settings"))
        self._guide_btn.config(text=tr("bottom_guide"))
        self._exit_btn.config(text=tr("bottom_exit"))

    def _show_guide(self) -> None:
        win = tk.Toplevel(self.root)
        win.title(tr("bottom_guide"))
        win.geometry("480x400")
        win.resizable(False, False)
        win.transient(self.root)
        # 与主窗口保持一致的图标
        try:
            big_ico = get_big_ico_path()
            if big_ico and big_ico.exists():
                win.iconbitmap(str(big_ico))
        except Exception:
            pass

        text = tk.Text(win, wrap=tk.WORD, font=("Microsoft YaHei", 10),
                       padx=16, pady=12, borderwidth=0)
        text.insert("1.0", tr("guide_placeholder"))
        text.config(state=tk.DISABLED)
        text.pack(fill=tk.BOTH, expand=True)

        ttk.Button(win, text="OK", command=win.destroy).pack(pady=(0, 10))

    # ================================================================
    # 设置窗口
    # ================================================================

    def _show_settings(self) -> None:
        SettingsWindow(self.root, self.config, self._on_settings_saved)

    def _on_settings_saved(self, new_config: Dict[str, Any]) -> None:
        old_monitor_dir = self.config.get("monitor_directory")
        new_monitor_dir = new_config.get("monitor_directory")

        self.config = new_config
        config_manager.save_config(self.config)

        md = self.config.get("monitor_directory", "")
        self.monitor_path_label.config(
            text=md if md else tr("monitor_not_set"),
            foreground="black" if md else "gray")

        if old_monitor_dir != new_monitor_dir:
            if self.monitor_active:
                self._stop_monitor(update_status=False)
                self._start_monitor()

        _get_logger().info("配置已更新")

    # ================================================================
    # 关闭 / 系统托盘
    # ================================================================

    def _on_close(self) -> None:
        if not self._tray_shown:
            create_tray_icon(self.root, self._quit_app)
            self._tray_shown = True
        self.root.withdraw()

    def _quit_app(self) -> None:
        try:
            if self.monitor_active:
                self._stop_monitor(update_status=False)
            destroy_tray_icon()
        except Exception:
            pass
        self.root.destroy()
        sys.exit(0)

    # ================================================================
    # 启动
    # ================================================================

    def run(self) -> None:
        self.root.mainloop()
