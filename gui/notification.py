"""
FileNest — 通知窗口

右下角非模态通知，检测到新文件时弹出，10 秒自动消失。
"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Callable, Optional

from i18n import tr


class NotificationWindow:
    """右下角非模态通知窗口。

    检测到新文件时弹出，10 秒无操作自动消失。
    """

    def __init__(self, parent: tk.Tk, file_path: Path,
                 on_classify: Callable[[], None],
                 on_ignore: Optional[Callable[[], None]] = None) -> None:
        self.win = tk.Toplevel(parent)
        self.win.title("")
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.geometry("320x100")

        # 右下角定位
        self.win.update_idletasks()
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        x = sw - 340
        y = sh - 140
        self.win.geometry(f"+{x}+{y}")

        frame = ttk.Frame(self.win, relief=tk.RAISED, borderwidth=2)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=tr("notif_title"), font=("", 9, "bold")).pack(anchor=tk.W, padx=10, pady=(8, 0))
        ttk.Label(frame, text=file_path.name, wraplength=280).pack(anchor=tk.W, padx=10)

        btn_f = ttk.Frame(frame)
        btn_f.pack(pady=6)

        self._on_classify = on_classify
        self._on_ignore_cb = on_ignore

        ttk.Button(btn_f, text=tr("notif_sort_btn"), command=self._classify).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_f, text=tr("notif_ignore_btn"), command=self._ignore).pack(side=tk.LEFT, padx=5)

        # 10 秒后自动消失
        self._auto_close_id = parent.after(10000, self._auto_close)

        self.win.protocol("WM_DELETE_WINDOW", self._ignore)

    def _classify(self) -> None:
        if hasattr(self, '_auto_close_id'):
            try:
                self.win.master.after_cancel(self._auto_close_id)  # type: ignore[union-attr]
            except Exception:
                pass
        self.win.destroy()
        self._on_classify()

    def _ignore(self) -> None:
        if hasattr(self, '_auto_close_id'):
            try:
                self.win.master.after_cancel(self._auto_close_id)  # type: ignore[union-attr]
            except Exception:
                pass
        self.win.destroy()
        if self._on_ignore_cb:
            self._on_ignore_cb()

    def _auto_close(self) -> None:
        self.win.destroy()
