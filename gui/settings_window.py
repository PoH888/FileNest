"""
FileNest — 设置窗口

包含所有配置分区：分类目录、匹配策略、监控目录、忽略名单、外观与行为。
"""

import logging
import os
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from typing import Any, Callable, Dict

import config_manager
from i18n import tr
from utils import init_logging


def _get_logger() -> logging.Logger:
    return logging.getLogger("FileNest")


class SettingsWindow:
    """独立模态设置窗口，含所有配置分区。"""

    def __init__(self, parent: tk.Tk, config: Dict[str, Any],
                 on_save: Callable[[Dict[str, Any]], None]) -> None:
        self.config = dict(config)
        self.on_save = on_save
        self._vars: Dict[str, Any] = {}

        self.dialog = tk.Toplevel(parent)
        self.dialog.title(tr("settings_title"))
        self.dialog.geometry("520x480")
        self.dialog.resizable(True, True)
        self.dialog.minsize(480, 400)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Notebook 选项卡
        self.notebook = ttk.Notebook(self.dialog)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        self._build_categories_tab()
        self._build_matching_tab()
        self._build_monitor_tab()
        self._build_ignore_tab()
        self._build_appearance_tab()

        btn_f = ttk.Frame(self.dialog)
        btn_f.pack(pady=(5, 10))
        ttk.Button(btn_f, text=tr("settings_save"), command=self._save).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_f, text=tr("settings_cancel"), command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)

        self.dialog.protocol("WM_DELETE_WINDOW", self.dialog.destroy)
        parent.wait_window(self.dialog)

    # ---- 分类目录 ----

    def _build_categories_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=tr("settings_tab_categories"))

        self._root_listbox = tk.Listbox(tab, height=6)
        self._root_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        for r in self.config.get("root_directories", []):
            self._root_listbox.insert(tk.END, r)

        btn_f = ttk.Frame(tab)
        btn_f.pack(pady=(0, 5))
        ttk.Button(btn_f, text=tr("settings_add_root"),
                   command=self._add_root).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_f, text=tr("settings_remove_root"),
                   command=self._remove_root).pack(side=tk.LEFT, padx=5)

        depth_f = ttk.Frame(tab)
        depth_f.pack(pady=(0, 10))
        ttk.Label(depth_f, text=tr("settings_depth")).pack(side=tk.LEFT)
        self._depth_var = tk.IntVar(value=self.config.get("max_depth", 5))
        ttk.Spinbox(depth_f, from_=1, to=20, textvariable=self._depth_var,
                    width=5).pack(side=tk.LEFT, padx=5)

    def _add_root(self) -> None:
        d = filedialog.askdirectory(title=tr("settings_add_root_title"))
        if d:
            self._root_listbox.insert(tk.END, d)

    def _remove_root(self) -> None:
        sel = self._root_listbox.curselection()
        if sel:
            self._root_listbox.delete(sel[0])

    # ---- 匹配策略 ----

    def _build_matching_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=tr("settings_tab_matching"))

        ttk.Label(tab, text=tr("settings_threshold")).pack(anchor=tk.W, padx=10, pady=(10, 0))
        th_f = ttk.Frame(tab)
        th_f.pack(fill=tk.X, padx=10)
        self._threshold_var = tk.IntVar(value=self.config.get("auto_threshold", 85))
        ttk.Scale(th_f, from_=50, to=100, variable=self._threshold_var,
                  command=lambda v: self._threshold_label.config(text=f"{int(float(v))}%")
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._threshold_label = ttk.Label(th_f, text=f"{self._threshold_var.get()}%", width=5)
        self._threshold_label.pack(side=tk.LEFT, padx=(5, 0))

        sep = ttk.Separator(tab, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(tab, text=tr("settings_mid_range")).pack(anchor=tk.W, padx=10)
        self._mid_auto_var = tk.BooleanVar(value=self.config.get("mid_range_auto", False))
        ttk.Radiobutton(tab, text=tr("settings_always_ask"), variable=self._mid_auto_var,
                        value=False).pack(anchor=tk.W, padx=20)
        rb_f = ttk.Frame(tab)
        rb_f.pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(rb_f, text=tr("settings_auto_prefix"),
                        variable=self._mid_auto_var,
                        value=True).pack(side=tk.LEFT)
        self._mid_min_var = tk.IntVar(value=self.config.get("mid_range_min", 60))
        ttk.Spinbox(rb_f, from_=0, to=100, textvariable=self._mid_min_var,
                    width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(rb_f, text="% – ").pack(side=tk.LEFT)
        self._mid_max_var = tk.IntVar(value=self.config.get("mid_range_max", 85))
        ttk.Spinbox(rb_f, from_=0, to=100, textvariable=self._mid_max_var,
                    width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(rb_f, text="%").pack(side=tk.LEFT)

        sep2 = ttk.Separator(tab, orient=tk.HORIZONTAL)
        sep2.pack(fill=tk.X, padx=10, pady=10)
        self._parent_wt_var = tk.BooleanVar(value=self.config.get("parent_weight_enabled", True))
        ttk.Checkbutton(tab, text=tr("settings_parent_weight"),
                        variable=self._parent_wt_var).pack(anchor=tk.W, padx=10)

        short_f = ttk.Frame(tab)
        short_f.pack(anchor=tk.W, padx=10, pady=(5, 0))
        short_cb = ttk.Checkbutton(short_f, text=tr("settings_short_name"),
                                    state=tk.DISABLED)
        short_cb.state(["!alternate", "selected"])
        short_cb.pack(side=tk.LEFT)

    # ---- 监控目录 ----

    def _build_monitor_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=tr("settings_tab_monitor"))

        ttk.Label(tab, text=tr("settings_monitor_path")).pack(anchor=tk.W, padx=10, pady=(10, 0))
        self._monitor_path_var = tk.StringVar(value=self.config.get("monitor_directory", ""))
        ttk.Entry(tab, textvariable=self._monitor_path_var, width=60).pack(padx=10, fill=tk.X)

        ttk.Button(tab, text=tr("settings_monitor_change"),
                   command=self._change_monitor_dir).pack(anchor=tk.W, padx=10, pady=5)

        self._monitor_enabled_var = tk.BooleanVar(
            value=self.config.get("monitor_enabled", False))
        ttk.Checkbutton(tab, text=tr("settings_monitor_enable"),
                        variable=self._monitor_enabled_var).pack(anchor=tk.W, padx=10)

    def _change_monitor_dir(self) -> None:
        d = filedialog.askdirectory(title=tr("settings_monitor_change"))
        if d:
            self._monitor_path_var.set(d)

    # ---- 忽略名单 ----

    def _build_ignore_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=tr("settings_tab_ignore"))

        ttk.Label(tab, text=tr("settings_ignore_hint")).pack(
            anchor=tk.W, padx=10, pady=(10, 5))
        self._ignore_var = tk.StringVar(value=self.config.get("ignore_patterns", ""))
        ttk.Entry(tab, textvariable=self._ignore_var, width=60).pack(
            padx=10, fill=tk.X)
        ttk.Label(tab, text=tr("settings_ignore_example"),
                  foreground="gray").pack(anchor=tk.W, padx=10, pady=(2, 0))

    # ---- 外观与行为 ----

    def _build_appearance_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=tr("settings_tab_appearance"))

        self._open_folder_var = tk.BooleanVar(
            value=self.config.get("open_folder_after_move", False))
        ttk.Checkbutton(tab, text=tr("settings_open_after"),
                        variable=self._open_folder_var).pack(anchor=tk.W, padx=10, pady=(10, 5))

        sep = ttk.Separator(tab, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, padx=10, pady=10)

        log_path = self._get_log_path()
        ttk.Label(tab, text=tr("settings_log_path", log_path),
                  wraplength=400).pack(anchor=tk.W, padx=10)
        ttk.Button(tab, text=tr("settings_log_btn"),
                   command=lambda: self._open_log_folder(log_path)).pack(anchor=tk.W, padx=10, pady=5)

    def _get_log_path(self) -> str:
        try:
            _, log_file = init_logging()
            return log_file
        except Exception:
            return tr("settings_log_path_unknown")

    def _open_log_folder(self, log_path: str) -> None:
        try:
            folder = os.path.dirname(log_path)
            if os.path.isdir(folder):
                os.startfile(folder)
        except Exception as e:
            _get_logger().error("打开日志文件夹失败: %s", e)

    # ---- 保存 ----

    def _save(self) -> None:
        self.config["root_directories"] = list(self._root_listbox.get(0, tk.END))
        self.config["max_depth"] = self._depth_var.get()
        self.config["auto_threshold"] = self._threshold_var.get()
        self.config["mid_range_auto"] = self._mid_auto_var.get()
        self.config["mid_range_min"] = self._mid_min_var.get()
        self.config["mid_range_max"] = self._mid_max_var.get()
        self.config["parent_weight_enabled"] = self._parent_wt_var.get()
        self.config["monitor_directory"] = self._monitor_path_var.get()
        self.config["monitor_enabled"] = self._monitor_enabled_var.get()
        self.config["ignore_patterns"] = self._ignore_var.get()
        self.config["open_folder_after_move"] = self._open_folder_var.get()

        self.on_save(self.config)
        self.dialog.destroy()
