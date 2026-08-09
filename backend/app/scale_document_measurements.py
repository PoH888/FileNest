"""第 35 课 E35-03 的文档索引、内存和 SQLite 大小测量。"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
import tracemalloc
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from .document_chunker import chunk_document
from .document_parser import load_document
from .filesystem_adapter import FileSystemAdapter
from .models import ChunkRecord, DocumentRecord
from .repositories import find_file_entries
from .scale_test_workspaces import (
    DEFAULT_SEED,
    ScaleWorkspaceManifest,
    generate_scale_workspace,
    get_scale_profile,
)
from .database import Base
from .services import create_workspace, scan_workspace


DEFAULT_REPEATS = 3
DOCUMENT_EXTENSIONS = frozenset({".md", ".markdown", ".txt"})


@dataclass(frozen=True, slots=True)
class NumericSummary:
    """同一规模同一指标的少量可复核样本。"""

    samples: tuple[int, ...]
    median: int
    minimum: int
    maximum: int


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """同一规模文档索引耗时的 wall-clock 样本。"""

    samples_ms: tuple[float, ...]
    median_ms: float
    minimum_ms: float
    maximum_ms: float


@dataclass(frozen=True, slots=True)
class DocumentScaleMeasurement:
    """一个规模的文档索引、内存和数据库大小证据。"""

    scale: str
    seed: int
    repeats: int
    file_count: int
    source_document_file_count: int
    fixture_bytes: int
    fixture_content_sha256: str
    indexed_document_count: int
    indexed_chunk_count: int
    index: TimingSummary
    peak_python_allocations: NumericSummary
    database_bytes: NumericSummary


class DocumentMeasurementError(ValueError):
    """文档测量参数或结果不符合固定测量契约。"""


class MeasurementOutputError(DocumentMeasurementError):
    """测量报告输出路径不安全或不可写。"""


def measure_document_scale(
    scale: str,
    *,
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
) -> DocumentScaleMeasurement:
    """在临时工作区和独立 SQLite 中测量一个文档规模。"""

    profile = get_scale_profile(scale)
    _validate_repeats(repeats)

    with TemporaryDirectory(prefix="filenest-e35-03-") as temporary_root:
        temporary_path = Path(temporary_root)
        workspace_root = temporary_path / "workspace"
        manifest = generate_scale_workspace(
            workspace_root,
            profile.name,
            seed=seed,
        )
        return _measure_generated_documents(
            manifest,
            workspace_root,
            temporary_path / "databases",
            repeats=repeats,
        )


def run_document_measurements(
    *,
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
) -> dict[str, object]:
    """按固定顺序运行 small、medium、large 三档文档测量。"""

    _validate_repeats(repeats)
    measurements = tuple(
        measure_document_scale(scale, seed=seed, repeats=repeats)
        for scale in ("small", "medium", "large")
    )
    return {
        "schema_version": "1.0",
        "task": "E35-03",
        "method": {
            "timing": "time.perf_counter wall-clock milliseconds",
            "repeats": repeats,
            "workspace_seed": seed,
            "index": (
                "load_document -> chunk_document -> "
                "DocumentRecord/ChunkRecord commit"
            ),
            "memory": (
                "tracemalloc peak Python allocations during document indexing"
            ),
            "database": (
                "main SQLite file plus -wal/-shm sidecars after engine disposal"
            ),
            "scope": "Markdown/TXT documents only; no embeddings",
        },
        "measurements": [
            asdict(measurement) for measurement in measurements
        ],
    }


def write_document_measurement_report(
    report: dict[str, object],
    output_path: Path,
) -> None:
    """只写入尚不存在的 JSON 报告，避免覆盖既有证据。"""

    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise MeasurementOutputError(
            "measurement report output must not already exist"
        )

    try:
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except OSError as error:
        raise MeasurementOutputError(
            "measurement report output could not be written"
        ) from error


def _measure_generated_documents(
    manifest: ScaleWorkspaceManifest,
    workspace_root: Path,
    database_root: Path,
    *,
    repeats: int,
) -> DocumentScaleMeasurement:
    """在同一工作区上用独立数据库重复测量文档索引。"""

    database_root.mkdir()
    index_samples: list[float] = []
    memory_samples: list[int] = []
    database_samples: list[int] = []
    document_count: int | None = None
    chunk_count: int | None = None

    for repeat_index in range(repeats):
        database_path = database_root / f"repeat-{repeat_index}.db"
        engine = create_engine(
            f"sqlite:///{database_path.as_posix()}"
        )
        try:
            Base.metadata.create_all(bind=engine)
            with Session(engine) as session:
                workspace = create_workspace(
                    session,
                    f"E35-03 {manifest.scale}",
                    str(workspace_root),
                )
                scan_result = scan_workspace(session, workspace.id)
                if scan_result.created != manifest.file_count:
                    raise DocumentMeasurementError(
                        "scan did not create the expected file index count"
                    )

                _start_memory_measurement()
                index_started = perf_counter()
                indexed_documents, indexed_chunks = _index_documents(
                    session,
                    workspace.id,
                    workspace_root,
                    seed=manifest.seed,
                )
                index_samples.append(_elapsed_ms(index_started))
                memory_samples.append(_stop_memory_measurement())

                if document_count is None:
                    document_count = indexed_documents
                    chunk_count = indexed_chunks
                elif (
                    indexed_documents != document_count
                    or indexed_chunks != chunk_count
                ):
                    raise DocumentMeasurementError(
                        "indexed document or chunk count changed between repeats"
                    )
        finally:
            engine.dispose()

        database_samples.append(_database_size_bytes(database_path))

    if document_count is None or chunk_count is None:
        raise DocumentMeasurementError(
            "document measurement produced no indexed documents"
        )

    return DocumentScaleMeasurement(
        scale=manifest.scale,
        seed=manifest.seed,
        repeats=repeats,
        file_count=manifest.file_count,
        source_document_file_count=manifest.document_file_count,
        fixture_bytes=manifest.total_bytes,
        fixture_content_sha256=manifest.content_sha256,
        indexed_document_count=document_count,
        indexed_chunk_count=chunk_count,
        index=_timing_summary(index_samples),
        peak_python_allocations=_numeric_summary(memory_samples),
        database_bytes=_numeric_summary(database_samples),
    )


def _index_documents(
    session: Session,
    workspace_id: int,
    workspace_root: Path,
    *,
    seed: int,
) -> tuple[int, int]:
    """复用现有解析、切分和 ORM 持久化链路，不引入新的索引行为。"""

    adapter = FileSystemAdapter(workspace_root)
    document_count = 0
    chunk_count = 0
    for file_entry in find_file_entries(session, workspace_id):
        if file_entry.extension not in DOCUMENT_EXTENSIONS:
            continue

        document = load_document(
            adapter,
            workspace_id=workspace_id,
            file_entry_id=file_entry.id,
            source_relative_path=file_entry.relative_path,
            document_id=uuid5(
                NAMESPACE_URL,
                f"filenest-e35-03:{seed}:{file_entry.relative_path}",
            ),
        )
        chunks = chunk_document(document)
        session.add(DocumentRecord.from_contract(document))
        session.add_all(ChunkRecord.from_contract(chunk) for chunk in chunks)
        document_count += 1
        chunk_count += len(chunks)

    session.commit()
    return document_count, chunk_count


def _start_memory_measurement() -> None:
    """开始当前索引段的 Python 分配峰值测量。"""

    if tracemalloc.is_tracing():
        tracemalloc.clear_traces()
    else:
        tracemalloc.start()


def _stop_memory_measurement() -> int:
    """读取并停止当前索引段的 Python 分配峰值测量。"""

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak_bytes


def _database_size_bytes(database_path: Path) -> int:
    """计算主库及 SQLite 旁车文件的总大小。"""

    paths = (
        database_path,
        Path(f"{database_path}-wal"),
        Path(f"{database_path}-shm"),
    )
    return sum(path.stat().st_size for path in paths if path.is_file())


def _timing_summary(samples: list[float]) -> TimingSummary:
    if not samples:
        raise DocumentMeasurementError("index timing produced no samples")

    return TimingSummary(
        samples_ms=tuple(round(sample, 3) for sample in samples),
        median_ms=round(float(median(samples)), 3),
        minimum_ms=round(min(samples), 3),
        maximum_ms=round(max(samples), 3),
    )


def _numeric_summary(samples: list[int]) -> NumericSummary:
    if not samples:
        raise DocumentMeasurementError("numeric measurement produced no samples")

    return NumericSummary(
        samples=tuple(samples),
        median=int(median(samples)),
        minimum=min(samples),
        maximum=max(samples),
    )


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1_000


def _validate_repeats(repeats: int) -> None:
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise DocumentMeasurementError("repeats must be a positive integer")


def build_parser() -> argparse.ArgumentParser:
    """构建 E35-03 测量命令。"""

    parser = argparse.ArgumentParser(
        description=(
            "测量 FileNest 第 35 课的文档索引、Python 内存和 SQLite 大小。"
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="必须尚不存在的 JSON 报告路径。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行三档文档测量并写出 JSON 报告。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_document_measurements()
        write_document_measurement_report(report, args.output)
    except (MeasurementOutputError, DocumentMeasurementError) as error:
        parser.error(str(error))

    print(f"document measurement report written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
