# FileNest 📃 → 🪹

> Let files find their way home
>
> Drag files in
>
> Or let FileNest watch in the background
>
> Leave the rest to FileNest
>
> Every file finds its own nest

---

```
　　　Report_Q4.docx　→　Work/Reports/
　　　Travel_Alaska.jpg　→　Photos/Travel/
　　　UI_Homepage.fig　→　Project A/UI Design/
```

---

FileNest analyzes file names and your directory structure

to automatically infer where each file belongs.

**No searching.**

**No manual organizing.**

**No need to remember where files go.**

---

Beyond drag-and-drop sorting,

FileNest can continuously monitor Downloads or any folder.

When new files appear,

it recommends a target directory

and you choose to **Categorize** or **Ignore**.

---

✅ Smart directory matching

✅ Auto-monitor Downloads or any folder

✅ Real-time categorization suggestions for new files

✅ One-click undo

✅ Fully local, no internet required

[📥 Download](https://github.com/PoH888/FileNest/releases)

---

## Demo

[![Drag & Drop](assets/English-Operate-thumb.png)](https://cdn.jsdelivr.net/gh/PoH888/FileNest@main/assets/English-Operate.mp4)

*Drag files to automatically sort them into matching directories*

[![Auto Monitor](assets/English-Monitor-thumb.png)](https://cdn.jsdelivr.net/gh/PoH888/FileNest@main/assets/English-Monitor.mp4)

*Monitor folders for new files and get real-time categorization suggestions*

## Requirements

- Python 3.11+ (Python 3.13 is recommended for the V2 backend)
- Windows for the desktop application
- Docker Desktop with Docker Compose, if you want to run the V2 API in a container

## Installation

Run the following commands in PowerShell from the directory where you keep the repository:

```powershell
git clone https://github.com/PoH888/FileNest.git
Set-Location .\FileNest

py -3.13 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

The CI-equivalent test commands additionally require `pytest` and `httpx`:

```powershell
& .\.venv\Scripts\python.exe -m pip install pytest httpx
```

## Usage

### Desktop application

```powershell
& .\.venv\Scripts\python.exe .\main.py
```

### V2 API locally

Run these commands from the repository root. The default local database is
`backend/filenest.db`.

```powershell
& .\.venv\Scripts\python.exe -m alembic -c .\backend\alembic.ini upgrade head
& .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In another PowerShell window, check the API and open the interactive docs at
<http://127.0.0.1:8000/docs>:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/health
```

### V2 API with Docker Compose

The Compose file contains only the API and its SQLite data volume. Start it
from the repository root:

```powershell
docker compose up --build --detach
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/health
docker compose down
```

If model settings are needed, copy `.env.example` to `.env`, edit the local
values, and then run `docker compose up --build --detach` again. Compose reads
`.env` for interpolation; a direct local Uvicorn run reads environment
variables from the current PowerShell session and does not load `.env`
automatically.

## Dependencies

| Dependency | Purpose |
|------------|---------|
| [thefuzz](https://github.com/seatgeek/thefuzz) / [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) | Fuzzy file name matching |
| [watchdog](https://github.com/gorakhargosh/watchdog) | File system monitoring |
| [pystray](https://github.com/moses-palmer/pystray) / [Pillow](https://python-pillow.org) | System tray and icon support |
| [tkinterdnd2](https://github.com/pmgagne/tkinterdnd2) | Desktop drag-and-drop support |
| [FastAPI](https://fastapi.tiangolo.com) / [Uvicorn](https://www.uvicorn.org) | V2 HTTP API and ASGI server |
| [SQLAlchemy](https://www.sqlalchemy.org) / [Alembic](https://alembic.sqlalchemy.org) | SQLite persistence and migrations |
| [Pydantic](https://docs.pydantic.dev) / `pydantic-settings` | Contracts and environment configuration |
| [OpenAI Python SDK](https://github.com/openai/openai-python) | OpenAI-compatible model client |
| [LangGraph](https://github.com/langchain-ai/langgraph) / `langgraph-checkpoint-sqlite` | Workflow execution and checkpoints |
| `pypdf` / `python-docx` | PDF and DOCX text extraction |
| `pytest` / `httpx` | Local and CI test execution; installed separately from runtime dependencies |

> tkinter is part of the Python standard library — no extra installation needed.

## Project Structure

```
FileNest/
├── main.py                    # Legacy desktop entrypoint
├── backend/
│   ├── app/                   # V2 FastAPI application and services
│   ├── alembic/               # Database migrations
│   └── alembic.ini
├── core/                      # Shared desktop/file logic
├── gui/                       # Desktop UI
├── tests/                     # Unit, integration, security, and evaluation tests
├── assets/                    # Icons, images, and demo media
├── .github/workflows/ci.yml   # Curated CI test groups
├── Dockerfile                 # V2 API image
├── compose.yaml               # API plus SQLite volume
├── .env.example               # Local/Compose environment template
└── requirements.txt           # Runtime dependencies
```

## Packaging

The commands below package the existing Windows desktop entrypoint (`main.py`).
The V2 API is run with Uvicorn or Docker Compose and is not bundled into the
desktop executable. Run these commands from the repository root in PowerShell:

```powershell
& .\.venv\Scripts\python.exe -m pip install pyinstaller
Remove-Item -Recurse -Force .\dist, .\build -ErrorAction SilentlyContinue
& .\.venv\Scripts\pyinstaller.exe --noconsole --onefile --name FileNest --add-data "assets;assets" --icon "assets\FileNest_big.ico" .\main.py
New-Item -ItemType Directory -Force .\dist\FileNest | Out-Null
Move-Item -Force .\dist\FileNest.exe .\dist\FileNest\
& .\.venv\Scripts\python.exe -c "import zipfile; zipfile.ZipFile('FileNest.zip','w',zipfile.ZIP_DEFLATED).write('dist/FileNest/FileNest.exe','FileNest.exe')"
```

Distribution structure:

```
FileNest.zip
  └── extract → FileNest/
                   └── FileNest.exe    ← settings.json auto-generated here on first run
```

> **Note**: With `--onefile`, `settings.json` is auto-generated on first run in the same directory as the exe (i.e., inside `FileNest/`). Moving the whole folder carries the config with it.

## Configuration

`settings.json` is auto-generated on first run. It can be modified through the settings window:

| Config Key | Description | Default |
|------------|-------------|---------|
| Classification directories | Root directories for file sorting | `[]` |
| Scan depth | Subfolder scanning level | 5 |
| Auto threshold | Auto-move above this score | 85 |
| Parent folder weighting | Bonus for matching parent dir name | On |
| Ignore list | File name keywords to skip | — |
| Language | zh / en | zh |

The V2 API accepts these environment variables:

| Variable | Description | Default or example |
|----------|-------------|--------------------|
| `FILENEST_DATABASE_URL` | SQLAlchemy database URL | `sqlite:///./backend/filenest.db` locally |
| `FILENEST_MODEL_PROVIDER` | Model provider name | See `.env.example` |
| `FILENEST_MODEL_NAME` | Model name | See `.env.example` |
| `FILENEST_MODEL_API_KEY` | Model API key | Keep local; never commit a real key |

For Docker Compose, use `.env.example` as the template for `.env`. For a
direct local run, set the variables in the current PowerShell session before
starting Uvicorn.

## Running Tests

Install the two test-only packages first, then run a representative local smoke test:

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

GitHub Actions runs the complete curated, non-overlapping unit, integration,
security, and evaluation groups defined in `.github/workflows/ci.yml` on pushes,
pull requests, and manual dispatch. It does not require a coverage threshold.

## How It Works

### Matching Strategy

1. File name normalization → strip extension, replace separators, lowercase
2. `thefuzz.token_sort_ratio` + `partial_ratio` for base score
3. Keyword overlap bonus (EN/CN bigrams + continuous substrings)
4. Parent folder weighting (parent dir name match bonus)
5. Candidates below 55 are discarded

### Match Result Decision

- **No match** (0 candidates) → Prompt manual handling
- **Strong match** (1 candidate ≥ 85) → Auto-move
- **Multiple candidates** (≥2) → Selection window
- **Nested overlap** (parent + child dirs in candidates) → Tree selection window

## License

MIT

# 🐱
