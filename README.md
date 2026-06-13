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

![Drag & Drop](assets/English-Operate.gif)

*Drag files to automatically sort them into matching directories*

![Monitoring](assets/English-Monitor.gif)

*Monitor folders for new files and get real-time categorization suggestions*

## Requirements

- Python 3.10+
- Windows

## Installation

```bash
# Clone the repository
git clone https://github.com/PoH888/FileNest.git
cd FileNest

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Dependencies

| Dependency | Purpose |
|------------|---------|
| [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy) | Fuzzy file name matching |
| [python-Levenshtein](https://github.com/rapidfuzz/python-Levenshtein) | C-accelerated matching |
| [watchdog](https://github.com/gorakhargosh/watchdog) | File system monitoring |
| [pystray](https://github.com/moses-palmer/pystray) | System tray icon |
| [Pillow](https://python-pillow.org) | Tray icon image processing |

> tkinter is part of the Python standard library — no extra installation needed.

## Project Structure

```
FileNest/
├── main.py              # Entry point
├── core/                # Core logic
│   ├── __init__.py
│   ├── matcher.py       # File name matching engine
│   ├── config_manager.py# Config read/write with fault tolerance
│   ├── file_mover.py    # File move, rollback, undo
│   ├── file_monitor.py  # watchdog file monitoring
│   ├── folder_scanner.py# Directory tree scanner
│   ├── utils.py         # Logging, paths, tray icon
│   └── i18n.py          # zh/en language pack
├── gui/
│   ├── main_window.py   # Main window
│   ├── dialogs.py       # Candidate selection & conflict dialogs
│   ├── settings_window.py  # Settings window
│   └── notification.py  # New file notification popup
├── tests/               # Test cases
│   ├── test_matcher.py
│   ├── test_config_manager.py
│   ├── test_file_mover.py
│   ├── test_file_monitor.py
│   ├── test_folder_scanner.py
│   └── test_utils.py
├── assets/              # Icons, images, demo GIFs
├── requirements.txt
└── README.md
```

## Packaging

```bash
pip install pyinstaller
pyinstaller --noconsole --onefile --name FileNest --add-data "assets;assets" --icon "assets/FileNest_big.ico" main.py
```

After packaging, place the generated `dist/FileNest.exe` together with the following files:
- `assets/` — Icons, images, and demo GIFs
- `setup_shortcut.bat` — One-click desktop shortcut & folder icon installation

Then compress the entire directory into `.zip` for distribution. Users just need to unzip and run `setup_shortcut.bat` to complete installation.

```bash
# Full packaging command
tar -acf FileNest.zip dist/FileNest.exe assets/ setup_shortcut.bat
```

> **Note**: `settings.json` is auto-generated on first run. Users can copy their existing config alongside the exe — paths remain unchanged.

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

## Running Tests

```bash
python -m pytest tests/ -v
```

## How It Works

### Matching Strategy

1. File name normalization → strip extension, replace separators, lowercase
2. `fuzzywuzzy.token_sort_ratio` + `partial_ratio` for base score
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
