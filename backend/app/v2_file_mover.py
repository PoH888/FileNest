"""V2 专用的精确目标文件移动原语。"""

import shutil
from pathlib import Path


def move_file(source: Path, target: Path) -> Path | None:
    """将已授权的普通文件移动到确定目标，不继承 V1 的改名策略。"""

    if not source.exists() or not source.is_file():
        return None
    if not target.parent.is_dir() or target.exists():
        return None

    try:
        actual = Path(shutil.move(str(source), str(target)))
    except OSError:
        return None
    return actual.resolve()
