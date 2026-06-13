"""
FileNest — 对话框组件

包含同名冲突对话框、多候选选择窗口、树形候选选择窗口。
"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core import matcher
from core.i18n import tr


# ===================================================================
# 同名文件处理对话框
# ===================================================================

class CollisionDialog:
    """同名文件处理模态对话框。

    提供"覆盖"、"保留两者"、"跳过"三个选项。
    """

    def __init__(self, parent: tk.Widget, filename: str) -> None:
        self.result: Optional[str] = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title(tr("collision_title"))
        self.dialog.geometry("380x150")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # 居中于父窗口
        self.dialog.update_idletasks()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        dw = self.dialog.winfo_width()
        dh = self.dialog.winfo_height()
        x = px + (pw - dw) // 2
        y = py + (ph - dh) // 2
        self.dialog.geometry(f"+{x}+{y}")

        ttk.Label(
            self.dialog,
            text=tr("collision_msg", filename),
            wraplength=350,
        ).pack(pady=(15, 10))

        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text=tr("collision_overwrite"), width=10,
                   command=lambda: self._on_choice("overwrite")).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text=tr("collision_keep_both"), width=10,
                   command=lambda: self._on_choice("keep_both")).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text=tr("collision_skip"), width=10,
                   command=lambda: self._on_choice("skip")).pack(side=tk.LEFT, padx=5)

        self.dialog.protocol("WM_DELETE_WINDOW", lambda: self._on_choice("skip"))
        parent.wait_window(self.dialog)

    def _on_choice(self, choice: str) -> None:
        self.result = choice
        self.dialog.destroy()


# ===================================================================
# 候选选择窗口
# ===================================================================

class CandidateWindow:
    """多候选文件夹选择窗口。

    模态窗口，最多显示 6 行完整路径，超出可滚动。
    """

    def __init__(self, parent: tk.Widget,
                 file_name: str,
                 matches: List[Tuple[Path, int]]) -> None:
        self.selected: Optional[Path] = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title(tr("candidate_title"))
        self.dialog.geometry("480x280")
        self.dialog.resizable(True, True)
        self.dialog.minsize(400, 220)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        ttk.Label(self.dialog, text=tr("candidate_desc"),
                  wraplength=460).pack(pady=(10, 5))

        frame = ttk.Frame(self.dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set,
                                  font=("Consolas", 9))
        scrollbar.config(command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 填充列表（最多显示 10 个候选，太多则提示）
        max_display = 10
        display_matches = matches[:max_display]
        for folder, score in display_matches:
            self.listbox.insert(tk.END, f"{folder}  ({score}%)")
        if len(matches) > max_display:
            self.listbox.insert(tk.END, tr("candidate_more", str(len(matches) - max_display)))

        self.listbox.bind("<Double-Button-1>", self._on_select)
        self.listbox.bind("<Return>", self._on_select)

        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(pady=(0, 10))

        ttk.Button(btn_frame, text=tr("candidate_ok"), command=self._on_confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text=tr("candidate_cancel"), command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)

        self.dialog.protocol("WM_DELETE_WINDOW", self.dialog.destroy)
        parent.wait_window(self.dialog)

    def _on_select(self, event: Optional[tk.Event] = None) -> None:
        self._on_confirm()

    def _on_confirm(self) -> None:
        sel = self.listbox.curselection()
        if not sel:
            return
        text: str = self.listbox.get(sel[0])
        path_str = text.rsplit("  (", 1)[0]
        self.selected = Path(path_str)
        self.dialog.destroy()


# ===================================================================
# 树形候选选择窗口（家族重叠场景）
# ===================================================================

class TreeCandidateWindow:
    """家族重叠时的树形文件夹选择窗口。

    以 ttk.Treeview 展示父子层级，用户可点选任意节点（含父文件夹）。
    """

    def __init__(self, parent: tk.Widget,
                 file_name: str,
                 matches: List[Tuple[Path, int]]) -> None:
        self.selected: Optional[Path] = None
        self._path_map: Dict[str, Path] = {}

        self.dialog = tk.Toplevel(parent)
        self.dialog.title(tr("tree_title"))
        self.dialog.geometry("520x320")
        self.dialog.resizable(True, True)
        self.dialog.minsize(400, 240)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        ttk.Label(self.dialog,
                  text=tr("tree_desc"),
                  wraplength=500).pack(pady=(10, 5))

        # Treeview
        tree_frame = ttk.Frame(self.dialog)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        self.tree = ttk.Treeview(tree_frame, yscrollcommand=scrollbar.set,
                                  show="tree", selectmode="browse")
        scrollbar.config(command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 构建树
        tree_data = matcher.build_family_tree(matches)
        for root_path, children in tree_data:
            root_score = dict(matches).get(root_path, 0)
            root_id = self._insert_node("", root_path, root_score)
            if children:
                self._build_subtree(root_id, children, matches)

        # 默认展开所有节点
        self._expand_all()

        self.tree.bind("<Double-Button-1>", self._on_select)
        self.tree.bind("<Return>", self._on_select)

        # 底部按钮
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(pady=(0, 10))

        ttk.Button(btn_frame, text=tr("candidate_ok"), command=self._on_confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text=tr("candidate_cancel"), command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)

        self.dialog.protocol("WM_DELETE_WINDOW", self.dialog.destroy)
        parent.wait_window(self.dialog)

    def _insert_node(self, parent_id: str, path: Path, score: int) -> str:
        """插入一个节点，返回其 treeview item id。"""
        display = f"{path}  ({score}%)"
        iid = self.tree.insert(parent_id, tk.END, text=display, open=True)
        self._path_map[iid] = path
        return iid

    def _build_subtree(self, parent_iid: str,
                       children: List[Tuple[Path, int]],
                       matches: List[Tuple[Path, int]]) -> None:
        """递归构建子树。"""
        for child_path, child_score in children:
            child_iid = self._insert_node(parent_iid, child_path, child_score)
            # 递归检查更深层级
            grand_children = []
            for p, s in matches:
                if p.parent == child_path and p in {m[0] for m in matches}:
                    grand_children.append((p, s))
            if grand_children:
                self._build_subtree(child_iid, grand_children, matches)

    def _expand_all(self) -> None:
        """递归展开所有节点。"""
        def _expand(item: str) -> None:
            self.tree.item(item, open=True)
            for child in self.tree.get_children(item):
                _expand(child)
        for root in self.tree.get_children(""):
            _expand(root)

    def _on_select(self, event: Optional[tk.Event] = None) -> None:
        self._on_confirm()

    def _on_confirm(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        self.selected = self._path_map.get(sel[0])
        self.dialog.destroy()
