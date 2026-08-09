"""第 35 课 E35-02 的扫描、搜索和只读 Agent 查询测量。"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from .agent_loop import AgentLoop
from .database import Base
from .fake_model_client import FakeModelClient
from .model_client import ModelMessage, ModelResponse, ModelToolCall
from .scale_test_workspaces import (
    DEFAULT_SEED,
    ScaleWorkspaceManifest,
    generate_scale_workspace,
    get_scale_profile,
)
from .services import create_workspace, scan_workspace, search_files
from .tool_registry import build_read_tool_registry


DEFAULT_REPEATS = 3
SEARCH_KEYWORD = "item-"
AGENT_PROMPT = "查询当前规模工作区中的文件。"


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """同一规模同一行动的少量 wall-clock 样本。"""

    samples_ms: tuple[float, ...]
    median_ms: float
    minimum_ms: float
    maximum_ms: float


@dataclass(frozen=True, slots=True)
class ScaleMeasurement:
    """一个规模的扫描、搜索和 Agent 查询证据。"""

    scale: str
    seed: int
    repeats: int
    file_count: int
    document_file_count: int
    fixture_bytes: int
    fixture_content_sha256: str
    search_keyword: str
    search_total: int
    scan: TimingSummary
    search: TimingSummary
    agent: TimingSummary
    agent_statuses: tuple[str, ...]
    agent_model_turns: tuple[int, ...]
    agent_tool_calls: tuple[int, ...]


class ScaleMeasurementError(ValueError):
    """规模测量参数或结果不符合固定测量契约。"""


class MeasurementOutputError(ScaleMeasurementError):
    """测量报告输出路径不安全或不可写。"""


def measure_scale(
    scale: str,
    *,
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
) -> ScaleMeasurement:
    """在临时工作区和临时 SQLite 中测量一个规模。"""

    profile = get_scale_profile(scale)
    _validate_repeats(repeats)

    with TemporaryDirectory(prefix="filenest-e35-02-") as temporary_root:
        temporary_path = Path(temporary_root)
        workspace_root = temporary_path / "workspace"
        manifest = generate_scale_workspace(
            workspace_root,
            profile.name,
            seed=seed,
        )
        return _measure_generated_scale(
            manifest,
            workspace_root,
            temporary_path / "databases",
            repeats=repeats,
        )


def run_scale_measurements(
    *,
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
) -> dict[str, object]:
    """按固定顺序运行 small、medium、large 三档测量。"""

    _validate_repeats(repeats)
    measurements = tuple(
        measure_scale(scale, seed=seed, repeats=repeats)
        for scale in ("small", "medium", "large")
    )
    return {
        "schema_version": "1.0",
        "task": "E35-02",
        "method": {
            "timing": "time.perf_counter wall-clock milliseconds",
            "repeats": repeats,
            "workspace_seed": seed,
            "search_keyword": SEARCH_KEYWORD,
            "agent_client": "FakeModelClient",
            "database": "fresh temporary SQLite database per repeat",
            "scan": "scan_workspace including FileEntry persistence",
            "search": "services.search_files with page_size=20",
            "agent": "AgentLoop with one search_files tool call and one final turn",
        },
        "measurements": [asdict(measurement) for measurement in measurements],
    }


def write_measurement_report(
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


def _measure_generated_scale(
    manifest: ScaleWorkspaceManifest,
    workspace_root: Path,
    database_root: Path,
    *,
    repeats: int,
) -> ScaleMeasurement:
    """在同一工作区上用独立数据库重复测量，避免重复生成文件。"""

    database_root.mkdir()
    scan_samples: list[float] = []
    search_samples: list[float] = []
    agent_samples: list[float] = []
    agent_statuses: list[str] = []
    agent_model_turns: list[int] = []
    agent_tool_calls: list[int] = []
    search_total: int | None = None

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
                    f"E35-02 {manifest.scale}",
                    str(workspace_root),
                )

                scan_started = perf_counter()
                scan_result = scan_workspace(session, workspace.id)
                scan_samples.append(_elapsed_ms(scan_started))
                if scan_result.created != manifest.file_count:
                    raise ScaleMeasurementError(
                        "scan did not create the expected file index count"
                    )

                search_started = perf_counter()
                search_result = search_files(
                    session,
                    workspace.id,
                    keyword=SEARCH_KEYWORD,
                    sort_by="relative_path",
                    sort_order="asc",
                    page=1,
                    page_size=20,
                )
                search_samples.append(_elapsed_ms(search_started))
                if search_total is None:
                    search_total = search_result.total
                elif search_result.total != search_total:
                    raise ScaleMeasurementError(
                        "search total changed between repeats"
                    )

                model_client = FakeModelClient(
                    _agent_responses(
                        workspace.id,
                        call_id=f"scale-search-{repeat_index}",
                    )
                )
                loop = AgentLoop(
                    model_client=model_client,
                    tool_registry=build_read_tool_registry(session),
                )
                agent_started = perf_counter()
                agent_result = loop.run(
                    [ModelMessage(role="user", content=AGENT_PROMPT)],
                    max_steps=3,
                    timeout_seconds=60.0,
                    max_model_retries=0,
                )
                agent_samples.append(_elapsed_ms(agent_started))
                agent_statuses.append(agent_result.status)
                agent_model_turns.append(agent_result.model_turns)
                agent_tool_calls.append(1)
                if agent_result.status != "completed":
                    raise ScaleMeasurementError(
                        "fake Agent query did not complete"
                    )
                if len(model_client.calls) != 2:
                    raise ScaleMeasurementError(
                        "fake Agent query did not use two model turns"
                    )
        finally:
            engine.dispose()

    if search_total is None:
        raise ScaleMeasurementError("scale measurement produced no search result")

    return ScaleMeasurement(
        scale=manifest.scale,
        seed=manifest.seed,
        repeats=repeats,
        file_count=manifest.file_count,
        document_file_count=manifest.document_file_count,
        fixture_bytes=manifest.total_bytes,
        fixture_content_sha256=manifest.content_sha256,
        search_keyword=SEARCH_KEYWORD,
        search_total=search_total,
        scan=_timing_summary(scan_samples),
        search=_timing_summary(search_samples),
        agent=_timing_summary(agent_samples),
        agent_statuses=tuple(agent_statuses),
        agent_model_turns=tuple(agent_model_turns),
        agent_tool_calls=tuple(agent_tool_calls),
    )


def _agent_responses(workspace_id: int, *, call_id: str) -> tuple[ModelResponse, ...]:
    return (
        ModelResponse(
            message=ModelMessage(
                role="assistant",
                tool_calls=(
                    ModelToolCall(
                        id=call_id,
                        name="search_files",
                        arguments={
                            "workspace_id": workspace_id,
                            "keyword": SEARCH_KEYWORD,
                            "limit": 20,
                        },
                    ),
                ),
            ),
            finish_reason="tool_calls",
        ),
        ModelResponse(
            message=ModelMessage(
                role="assistant",
                content="规模查询完成。",
            ),
            finish_reason="stop",
        ),
    )


def _timing_summary(samples: list[float]) -> TimingSummary:
    if not samples:
        raise ScaleMeasurementError("timing measurement produced no samples")

    return TimingSummary(
        samples_ms=tuple(round(sample, 3) for sample in samples),
        median_ms=round(float(median(samples)), 3),
        minimum_ms=round(min(samples), 3),
        maximum_ms=round(max(samples), 3),
    )


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1_000


def _validate_repeats(repeats: int) -> None:
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ScaleMeasurementError("repeats must be a positive integer")


def build_parser() -> argparse.ArgumentParser:
    """构建 E35-02 测量命令。"""

    parser = argparse.ArgumentParser(
        description="测量 FileNest 第 35 课的扫描、搜索和只读 Agent 查询。"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="必须尚不存在的 JSON 报告路径。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行三档测量并写出 JSON 报告。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_scale_measurements()
        write_measurement_report(report, args.output)
    except (MeasurementOutputError, ScaleMeasurementError) as error:
        parser.error(str(error))

    print(f"measurement report written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
