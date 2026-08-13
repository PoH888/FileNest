# FileNest 📃 → 🪹

> 本地优先的文件整理项目，以真实证据为基础建设 V2 后端。

本仓库包含两条相关路径：

- **旧版 Windows 桌面程序**：拖拽整理、模糊目录匹配、可选文件夹监控和本地撤销。
- **FileNest V2 后端**：FastAPI 应用，将文件整理拆成可查看、可审批、可安全撤销的链路：查询 → 计划预览 → 审批 → 执行 → undo。

下面关于 V2 的描述以当前代码为准，不把计划中的能力写成当前能力。

---

## 当前 V2 能力

V2 后端当前提供：

- 已授权工作区注册、显式工作区扫描、文件搜索和文件元数据查看；
- 文档加载、可追踪片段、关键词知识搜索，以及离线实验性向量路径；
- 只读 Agent Run、持久化运行/工具状态和 SSE 状态投影；
- 整理计划预览、持久化审批决定、安全执行、执行历史和受保护的 undo；
- 本机 stdio MCP 服务，当前暴露 `search_files`、`knowledge_search` 和 `create_operation_proposal`。

V2 的安全整理主链是显式的：审批只改变工作流状态，不写入文件；执行时重新读取已批准的 checkpoint，并再次校验授权、计划和文件前置条件；undo 依赖执行历史及当前文件元数据校验，不会无条件覆盖路径。

## 现有桌面演示媒体

以下视频展示的是旧版 Windows 桌面程序，不是 V2 后端已经接入旧版文件监控的证据。

[📥 Download](https://github.com/PoH888/FileNest/releases)

---

[![拖拽操作演示](assets/中文-操作-thumb.png)](https://cdn.jsdelivr.net/gh/PoH888/FileNest@main/assets/%E4%B8%AD%E6%96%87-%E6%93%8D%E4%BD%9C.mp4)

*旧版桌面程序的拖拽整理*

[![监控归类演示](assets/中文-监控-thumb.png)](https://cdn.jsdelivr.net/gh/PoH888/FileNest@main/assets/%E4%B8%AD%E6%96%87-%E7%9B%91%E6%8E%A7.mp4)

*旧版桌面程序的文件夹监控*

## 代表性 V2 演示

可复现的本地演示使用专用工作区和一个 187 字节的固定 ZIP 文件。已有记录中的 SHA-256 为：

```text
02394A5D0DB905108D69D44ADCCAE3036AB02727CF456848688DAF28C30C16C3
```

演示的状态顺序为：

```text
waiting / WAITING_APPROVAL
        ↓ 批准
ready / APPROVED
        ↓ 执行
COMPLETED
        ↓ undo
UNDONE
```

已有运行记录检查了 undo 后源文件恢复且 SHA-256 不变。确定性 API 路径使用已建立的文件索引，不需要外部模型。浏览器中的 Agent 查询需要有效的本地模型配置；它不被当作模型质量证据。

## 环境要求

- V2 后端本地验证使用 Python 3.13。
- 旧版桌面程序需要 Windows。
- 仓库包含 V2 API 的 Docker Compose 配置，但当前环境未验证本地 Docker 构建或容器运行。

## 安装

在 PowerShell 中进入存放仓库的目录，执行：

```powershell
git clone https://github.com/PoH888/FileNest.git
Set-Location .\FileNest

py -3.13 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

CI 等价的测试命令还需要单独安装 `pytest` 和 `httpx`：

```powershell
& .\.venv\Scripts\python.exe -m pip install pytest httpx
```

## 运行

### 桌面应用

```powershell
& .\.venv\Scripts\python.exe .\main.py
```

### 本地运行 V2 API

先在 `backend` 目录执行迁移，再从仓库根目录启动现有 FastAPI 应用：

```powershell
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path

Push-Location .\backend
& $python -m alembic upgrade head
Pop-Location

& $python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

在另一个 PowerShell 窗口检查 API，并在浏览器打开
<http://127.0.0.1:8000/docs> 查看交互式文档：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/health
```

预期健康结果为 `status: ok`。打开
<http://127.0.0.1:8000/> 查看当前最小 V2 界面，打开
<http://127.0.0.1:8000/docs> 查看 OpenAPI 交互文档。

界面可以选择工作区、提交只读 Agent 请求、查看出处、生成计划、批准或拒绝计划、执行已批准计划，并撤销已完成执行。Agent 请求需要有效的模型配置；没有模型配置时，使用 E39-03 本地演示脚本中的确定性 HTTP 主链。

### 使用 Docker Compose 运行 V2 API

仓库包含 API 和 SQLite 数据卷的 Compose 定义。当前环境未验证本地 Docker 构建或容器运行，以下命令只是配置参考，不是已记录的执行结果：

```powershell
docker compose up --build --detach
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/health
docker compose down
```

如果需要模型配置，Compose 会读取 `.env` 做变量替换。直接本地运行 Uvicorn 时，程序使用当前 PowerShell 会话中的环境变量，不会自动加载 `.env`。

### 确定性 HTTP 主链

无模型演示使用当前 API 返回的动态 ID，接口顺序为：

1. `GET` 或 `POST /api/v1/workspaces` ——查找或注册专用工作区；
2. `POST /api/v1/workspaces/{workspace_id}/scan` ——同步文件索引；
3. `GET /api/v1/workspaces/{workspace_id}/files` ——查询固定 ZIP；
4. `POST /api/v1/workflows` ——创建等待审批的计划；
5. `POST /api/v1/workflows/{workflow_id}/decisions` ——批准当前展示的计划；
6. `POST /api/v1/workflows/{workflow_id}/execute` ——执行已批准计划；
7. `POST /api/v1/workflows/{workflow_id}/undo` ——依据执行历史恢复文件。

从当前检出目录运行时，使用 `docs/E29-04-可复现运行说明与工程调用链.md` 中的固定文件准备命令和完整 PowerShell 主链。不要硬编码旧的工作区、文件、工作流或计划 ID。如果目标文件已经存在，应先检查上一次运行并执行 undo，不要用宽泛的重置命令删除文件。

### 本机 stdio MCP 服务

当前 MCP 入口为：

```powershell
& .\.venv\Scripts\python.exe -m backend.app.mcp_server
```

它只暴露当前的只读搜索工具和待审批操作提案，不暴露批准、执行或 undo 工具，也不开放网络传输。

## 依赖

| 依赖 | 用途 |
|------|------|
| [thefuzz](https://github.com/seatgeek/thefuzz) / [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) | 文件名模糊匹配 |
| [watchdog](https://github.com/gorakhargosh/watchdog) | 文件系统监控 |
| [pystray](https://github.com/moses-palmer/pystray) / [Pillow](https://python-pillow.org) | 系统托盘和图标支持 |
| [tkinterdnd2](https://github.com/pmgagne/tkinterdnd2) | 桌面拖放支持 |
| [FastAPI](https://fastapi.tiangolo.com) / [Uvicorn](https://www.uvicorn.org) | V2 HTTP API 和 ASGI 服务 |
| [SQLAlchemy](https://www.sqlalchemy.org) / [Alembic](https://alembic.sqlalchemy.org) | SQLite 持久化和迁移 |
| [Pydantic](https://docs.pydantic.dev) / `pydantic-settings` | 数据契约和环境配置 |
| [OpenAI Python SDK](https://github.com/openai/openai-python) | OpenAI 兼容模型客户端 |
| [LangGraph](https://github.com/langchain-ai/langgraph) / `langgraph-checkpoint-sqlite` | 工作流执行和检查点 |
| `pypdf` / `python-docx` | PDF 和 DOCX 文本提取 |
| `pytest` / `httpx` | 本地和 CI 测试；不随运行时依赖安装，需单独安装 |

> tkinter 属于 Python 标准库，无需额外安装。

## 项目结构

```
FileNest/
├── main.py                    # 旧版桌面程序入口
├── backend/
│   ├── app/                   # V2 FastAPI 应用和服务
│   ├── alembic/               # 数据库迁移
│   ├── evaluation/            # 已记录的评测和规模结果
│   └── alembic.ini
├── core/                      # 旧版桌面/文件逻辑
├── gui/                       # 旧版桌面界面
├── tests/                     # 单元、集成、安全和评测测试
├── assets/                    # 图标、图片和演示媒体
├── .github/workflows/ci.yml   # 精简 CI 测试分组
├── Dockerfile                 # V2 API 镜像
├── compose.yaml               # API 和 SQLite 数据卷
├── .env.example               # 本地/Compose 环境变量模板
└── requirements.txt           # 运行时依赖
```

## 打包

以下命令打包现有的 Windows 桌面入口（`main.py`）。V2 API 使用 Uvicorn
或 Docker Compose 运行，不会打入桌面 exe。请在仓库根目录的 PowerShell
中执行：

```powershell
& .\.venv\Scripts\python.exe -m pip install pyinstaller
Remove-Item -Recurse -Force .\dist, .\build -ErrorAction SilentlyContinue
& .\.venv\Scripts\pyinstaller.exe --noconsole --onefile --name FileNest --add-data "assets;assets" --icon "assets\FileNest_big.ico" .\main.py
New-Item -ItemType Directory -Force .\dist\FileNest | Out-Null
Move-Item -Force .\dist\FileNest.exe .\dist\FileNest\
& .\.venv\Scripts\python.exe -c "import zipfile; zipfile.ZipFile('FileNest.zip','w',zipfile.ZIP_DEFLATED).write('dist/FileNest/FileNest.exe','FileNest.exe')"
```

分发结构：

```
FileNest.zip
  └── 解压 → FileNest/
                └── FileNest.exe    ← 首次运行后同级自动生成 settings.json
```

> **注意**：`--onefile` 模式下 `settings.json` 首次运行时会自动生成在 exe 所在目录（即 `FileNest/` 下），随文件夹整体移动即可带走配置。

## 配置

### 旧版桌面程序配置

桌面程序首次运行自动生成 `settings.json`，可通过设置窗口修改：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| 分类目录 | 文件归类的根目录列表 | `[]` |
| 扫描深度 | 子文件夹扫描层级 | 5 |
| 自动阈值 | 高于此分数自动移动 | 85 |
| 父文件夹加权 | 父目录名匹配加分 | 开启 |
| 忽略名单 | 跳过的文件名关键词 | — |
| 语言 | zh / en | zh |

### V2 API 环境变量

V2 API 支持以下环境变量：

| 变量 | 说明 | 默认值或示例 |
|------|------|--------------|
| `FILENEST_DATABASE_URL` | SQLAlchemy 数据库 URL | 本地默认为 `sqlite:///./backend/filenest.db` |
| `FILENEST_MODEL_PROVIDER` | 模型提供方名称 | 参见 `.env.example` |
| `FILENEST_MODEL_NAME` | 模型名称 | 参见 `.env.example` |
| `FILENEST_MODEL_API_KEY` | 模型 API 密钥 | 仅保存在本地，禁止提交真实密钥 |

使用 Docker Compose 时，以 `.env.example` 为 `.env` 模板。直接本地运行时，
请在启动 Uvicorn 前于当前 PowerShell 会话设置这些变量。

## 运行测试

先安装两个测试专用包，再运行一组代表性的本地冒烟测试：

```powershell
& .\.venv\Scripts\python.exe -m pip install pytest httpx
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m pytest `
  .\tests\test_matcher.py `
  .\tests\test_backend_migrations.py `
  .\tests\test_path_policy.py `
  .\tests\test_agent_evaluation.py `
  -q -p no:cacheprovider
```

GitHub Actions 会执行 `.github/workflows/ci.yml` 中定义的完整且互不重复的
单元、集成、安全和评测分组，触发方式包括 push、pull request 和手动运行。
CI 不设置覆盖率门槛。

## 技术要点

### 旧版桌面匹配策略

1. 文件名规范化 → 去扩展名、替换分隔符、全小写
2. `thefuzz.token_sort_ratio` + `partial_ratio` 计算基础分
3. 关键词重叠加分（中英文 bigram + 连续子串）
4. 父文件夹加权（父目录名命中加分）
5. 低于 55 分的候选直接丢弃

### 旧版桌面匹配结果决策

- **无匹配**（0 候选）→ 提示手动处理
- **强匹配**（1 候选且 ≥85 分）→ 自动移动
- **多候选**（≥2 个）→ 弹出选择窗口
- **家族重叠**（父子目录同时在候选）→ 树形选择窗口

## 当前 V2 边界

- 旧版桌面监控尚未接入 V2 的 Job/Attempt 或文档索引路径。
- V2 Agent POST 当前为同步运行；SSE 只投影已持久化的 Agent 状态，本身不执行或取消业务任务。
- 大规模扫描和文档索引已有本地长任务证据，但 Job/Attempt 基线尚未完整接入 HTTP、扫描或文档索引。
- 关键词检索仍是默认路径；向量检索仅作实验，没有引入外部向量数据库。
- Redis、Celery、分布式 worker、用户认证、多租户隔离和远程 MCP 传输不是当前 V2 能力。
- 当前没有经过验证的生产部署、生产用户量、QPS、SLA、真实模型准确率或真实 Embedding 质量结论。

## License

MIT

# 🐱
