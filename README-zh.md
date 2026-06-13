# FileNest 📃 → 🪹

> 让文件自己回家
>
> 把文件拖进来
>
> 或者让 FileNest 在后台帮你盯着
>
> 剩下的交给 FileNest
>
> 让每个文件都找到自己的巢

---

```
　　项目周报.docx　→　工作/周报/
　　东京旅行照片.jpg　→　照片/旅行/
　　设计稿_v2.fig　→　项目A/UI设计/
```

---

FileNest 会根据文件名和你的目录结构，

自动推断文件应该存放的位置。

**无需搜索。**

**无需手动整理。**

**无需记住文件应该放在哪里。**

---

除了拖拽整理，

FileNest 还可以持续监控 Downloads 或任意文件夹。

当新文件出现时，

自动推荐目标目录，

由你选择「归类」或「忽略」。

---

✅ 智能目录匹配

✅ 自动监控 Downloads 或任意文件夹

✅ 新文件实时归类建议

✅ 一键撤销

✅ 完全本地运行

[📥 Download](https://github.com/PoH888/FileNest/releases)

---

## 演示

![拖拽操作](assets/中文-操作.gif)

*拖拽文件自动归类到匹配目录*

![监控演示](assets/中文-监控.gif)

*后台监控新文件出现，实时推荐目标目录*

## 环境要求

- Python 3.10+
- Windows

## 安装

```bash
# 克隆仓库
git clone https://github.com/PoH888/FileNest.git
cd FileNest

# 安装依赖
pip install -r requirements.txt
```

## 运行

```bash
python main.py
```

## 依赖

| 依赖 | 用途 |
|------|------|
| [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy) | 文件名模糊匹配 |
| [python-Levenshtein](https://github.com/rapidfuzz/python-Levenshtein) | 匹配算法 C 加速 |
| [watchdog](https://github.com/gorakhargosh/watchdog) | 文件系统监控 |
| [pystray](https://github.com/moses-palmer/pystray) | 系统托盘图标 |
| [Pillow](https://python-pillow.org) | 托盘图标图像处理 |

> tkinter 为 Python 标准库，无需额外安装。

## 项目结构

```
FileNest/
├── main.py              # 程序入口
├── core/                # 核心逻辑
│   ├── __init__.py
│   ├── matcher.py       # 文件名匹配引擎
│   ├── config_manager.py# 配置读写、容错
│   ├── file_mover.py    # 文件移动、回退、撤销
│   ├── file_monitor.py  # watchdog 文件监控
│   ├── folder_scanner.py# 目录树扫描
│   ├── utils.py         # 日志、路径、托盘图标
│   └── i18n.py          # 中/英文语言包
├── gui/
│   ├── main_window.py   # 主窗口
│   ├── dialogs.py       # 候选选择、冲突处理对话框
│   ├── settings_window.py  # 设置窗口
│   └── notification.py  # 新文件通知弹窗
├── tests/               # 测试用例
│   ├── test_matcher.py
│   ├── test_config_manager.py
│   ├── test_file_mover.py
│   ├── test_file_monitor.py
│   ├── test_folder_scanner.py
│   └── test_utils.py
├── assets/              # 图标、图片、演示 GIF
├── requirements.txt
└── README.md
```

## 打包

```bash
pip install pyinstaller
```

所有依赖（assets/ 图标、tkinterdnd2 拖拽 DLL、Python 运行时）均被打入单个 exe 内部，没有零散文件。分发结构：

```
FileNest.zip
  └── 解压 → FileNest/
                └── FileNest.exe    ← 首次运行后同级自动生成 settings.json
```

```bash
# 1) 清除旧的构建产物
rm -rf dist build

# 2) 打包为单文件 exe（--onefile）
pyinstaller --noconsole --onefile --name FileNest --add-data "assets;assets" --icon "assets/FileNest_big.ico" main.py

# 3) 创建分发文件夹，把 exe 放进去
mkdir -p dist/FileNest
mv dist/FileNest.exe dist/FileNest/

# 4) 压缩为 .zip（注意：exe 放在 zip 根级，不要文件夹前缀，
#         Windows "Extract All…" 会自动创建 FileNest/ 目录）
python -c "import zipfile; zipfile.ZipFile('FileNest.zip','w',zipfile.ZIP_DEFLATED).write('dist/FileNest/FileNest.exe','FileNest.exe')"
```

> **注意**：`--onefile` 模式下 `settings.json` 首次运行时会自动生成在 exe 所在目录（即 `FileNest/` 下），随文件夹整体移动即可带走配置。

## 配置

首次运行自动生成 `settings.json`，可通过设置窗口修改：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| 分类目录 | 文件归类的根目录列表 | `[]` |
| 扫描深度 | 子文件夹扫描层级 | 5 |
| 自动阈值 | 高于此分数自动移动 | 85 |
| 父文件夹加权 | 父目录名匹配加分 | 开启 |
| 忽略名单 | 跳过的文件名关键词 | — |
| 语言 | zh / en | zh |

## 运行测试

```bash
python -m pytest tests/ -v
```

## 技术要点

### 匹配策略

1. 文件名规范化 → 去扩展名、替换分隔符、全小写
2. `fuzzywuzzy.token_sort_ratio` + `partial_ratio` 计算基础分
3. 关键词重叠加分（中英文 bigram + 连续子串）
4. 父文件夹加权（父目录名命中加分）
5. 低于 55 分的候选直接丢弃

### 匹配结果决策

- **无匹配**（0 候选）→ 提示手动处理
- **强匹配**（1 候选且 ≥85 分）→ 自动移动
- **多候选**（≥2 个）→ 弹出选择窗口
- **家族重叠**（父子目录同时在候选）→ 树形选择窗口

## License

MIT

# 🐱