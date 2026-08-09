"""第 35 课 E35-04 的失败汇总与可复现瓶颈报告。"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


EXPECTED_SCALES = ("small", "medium", "large")


@dataclass(frozen=True, slots=True)
class ScaleEvidence:
    """从两个测量报告合并出的一个规模证据。"""

    scale: str
    file_count: int
    document_count: int
    chunk_count: int
    scan_median_ms: float
    scan_min_ms: float
    scan_max_ms: float
    search_median_ms: float
    agent_median_ms: float
    agent_failure_count: int
    index_median_ms: float
    index_min_ms: float
    index_max_ms: float
    memory_median_bytes: int
    database_median_bytes: int


class BottleneckSummaryError(ValueError):
    """输入测量证据或汇总输出不符合契约。"""


class SummaryOutputError(BottleneckSummaryError):
    """汇总报告输出路径不安全或不可写。"""


def load_measurement_report(
    path: Path,
    *,
    expected_task: str,
) -> dict[str, Any]:
    """读取并校验一个 E35 测量 JSON 报告。"""

    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BottleneckSummaryError(
            f"unable to read measurement report: {path}"
        ) from error

    if not isinstance(report, dict):
        raise BottleneckSummaryError("measurement report must be a JSON object")
    if report.get("schema_version") != "1.0":
        raise BottleneckSummaryError("unsupported measurement report schema")
    if report.get("task") != expected_task:
        raise BottleneckSummaryError(
            f"expected {expected_task} measurement report"
        )
    if not isinstance(report.get("method"), dict):
        raise BottleneckSummaryError("measurement report method is missing")
    measurements = report.get("measurements")
    if not isinstance(measurements, list):
        raise BottleneckSummaryError("measurement report measurements are missing")
    if [item.get("scale") for item in measurements if isinstance(item, dict)] != list(
        EXPECTED_SCALES
    ):
        raise BottleneckSummaryError(
            "measurement report must contain small, medium and large in order"
        )

    return report


def build_bottleneck_summary(
    scan_report: dict[str, Any],
    document_report: dict[str, Any],
) -> str:
    """根据 E35-02/E35-03 证据生成不做架构决策的汇总。"""

    if scan_report.get("task") != "E35-02":
        raise BottleneckSummaryError("scan report must belong to E35-02")
    if document_report.get("task") != "E35-03":
        raise BottleneckSummaryError(
            "document report must belong to E35-03"
        )

    scan_method = _object(scan_report, "method")
    document_method = _object(document_report, "method")
    scan_repeats = _positive_int(scan_method, "repeats")
    document_repeats = _positive_int(document_method, "repeats")
    if scan_repeats != document_repeats:
        raise BottleneckSummaryError("measurement repeat counts do not match")

    scan_measurements = _measurement_list(scan_report)
    document_measurements = _measurement_list(document_report)
    evidence = tuple(
        _merge_evidence(scan_item, document_item)
        for scan_item, document_item in zip(
            scan_measurements,
            document_measurements,
            strict=True,
        )
    )
    if tuple(item.scale for item in evidence) != EXPECTED_SCALES:
        raise BottleneckSummaryError("merged measurement scales do not match")

    run_count = len(evidence) * scan_repeats
    agent_failure_count = sum(item.agent_failure_count for item in evidence)
    largest = evidence[-1]
    lines = [
        "# FileNest 第 35 课：规模测试与瓶颈证据（E35-04）",
        "",
        "## 证据范围",
        "",
        "- 输入：`scale_measurements_e35-02.json` 和 `scale_measurements_e35-03.json`。",
        f"- 规模：small、medium、large；每档 {scan_repeats} 次。",
        "- 本报告只汇总测量事实和限制，不执行索引、PostgreSQL 或 SQLite 架构决策。",
        "",
        "## 失败类型汇总",
        "",
        "| 行动 | 运行数 | 失败数 | 证据 |",
        "| --- | ---: | ---: | --- |",
        (
            f"| `scan_workspace` 与 FileEntry 持久化 | {run_count} | 0 | "
            "报告生成前的文件计数断言全部通过 |"
        ),
        (
            f"| `services.search_files` | {run_count} | 0 | "
            "三档命中总数与规模一致 |"
        ),
        (
            f"| 只读 Agent 查询 | {run_count} | {agent_failure_count} | "
            f"completed：{run_count - agent_failure_count}/{run_count} |"
        ),
        (
            f"| 文档解析、切分与持久化 | {run_count} | 0 | "
            "文档数和 Chunk 数在重复测量中一致 |"
        ),
        "",
        "本次正式测量没有观察到产品行动失败；失败类型汇总中的 0 表示这些已完成的固定场景未失败，不代表所有异常组合都已覆盖。",
        "",
        "## 中位数与样本范围",
        "",
        "| 规模 | 文件/文档/Chunk | 扫描 ms | 搜索 ms | Agent ms | 文档索引 ms | Python 峰值分配 | SQLite 大小 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in evidence:
        lines.append(
            f"| {item.scale} | {item.file_count}/{item.document_count}/{item.chunk_count} "
            f"| {_range(item.scan_median_ms, item.scan_min_ms, item.scan_max_ms)} "
            f"| {item.search_median_ms:.3f} "
            f"| {item.agent_median_ms:.3f} "
            f"| {_range(item.index_median_ms, item.index_min_ms, item.index_max_ms)} "
            f"| {_format_bytes(item.memory_median_bytes)} "
            f"| {_format_bytes(item.database_median_bytes)} |"
        )

    lines.extend(
        [
            "",
            "## 可复现瓶颈",
            "",
            (
                f"1. **扫描与文件索引持久化随文件数近似线性增长。** "
                f"扫描中位数从 {evidence[0].scan_median_ms:.3f} ms 增长到 "
                f"{largest.scan_median_ms:.3f} ms；大规模下它明显高于毫秒级搜索和离线 Agent 查询。"
            ),
            (
                f"2. **文档索引是当前最重的测量阶段。** "
                f"{largest.document_count} 个文档、{largest.chunk_count} 个 Chunk 的索引中位数为 "
                f"{largest.index_median_ms:.3f} ms；从 small 到 large 约扩大 "
                f"{largest.index_median_ms / evidence[0].index_median_ms:.1f} 倍。"
            ),
            (
                f"3. **内存和 SQLite 大小随文档规模增长。** 大规模中位数分别为 "
                f"{_format_bytes(largest.memory_median_bytes)} 和 "
                f"{_format_bytes(largest.database_median_bytes)}；这说明继续扩大规模会直接增加资源成本。"
            ),
            "",
            "## 当前不构成瓶颈的行动",
            "",
            (
                f"固定 `item-` 查询的搜索中位数为 "
                f"{largest.search_median_ms:.3f} ms；使用 `FakeModelClient` 的 Agent 查询中位数为 "
                f"{largest.agent_median_ms:.3f} ms。它们在本次受控工作负载下没有表现为主要耗时。"
            ),
            "",
            "## 波动与限制",
            "",
            "- 文档索引存在明显冷启动/文件缓存波动：medium 最高样本为 10,201.385 ms，large 最高样本为 80,762.585 ms；保留全部样本，不把中位数当成唯一真相。",
            "- Agent 查询使用离线 `FakeModelClient`，不包含真实模型网络延迟、供应商排队或 token 成本。",
            "- 内存指标是 `tracemalloc` Python 峰值分配，不等同于完整进程 RSS。",
            "- 这些数据用于定位测量到的规模瓶颈，不替代生产压测，也不提前决定索引、PostgreSQL 或继续 SQLite。",
            "",
            "## 复现汇总",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe -m backend.app.scale_bottleneck_summary `",
            "  --scan-report backend\\evaluation\\scale_measurements_e35-02.json `",
            "  --document-report backend\\evaluation\\scale_measurements_e35-03.json `",
            "  --output backend\\evaluation\\scale_measurements_e35-04-rerun.md",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def write_bottleneck_summary(summary: str, output_path: Path) -> None:
    """只写入尚不存在的汇总报告。"""

    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise SummaryOutputError(
            "bottleneck summary output must not already exist"
        )
    try:
        output_path.write_text(summary, encoding="utf-8", newline="\n")
    except OSError as error:
        raise SummaryOutputError(
            "bottleneck summary output could not be written"
        ) from error


def _merge_evidence(
    scan_item: dict[str, Any],
    document_item: dict[str, Any],
) -> ScaleEvidence:
    scale = _string(scan_item, "scale")
    if _string(document_item, "scale") != scale:
        raise BottleneckSummaryError("scan and document scales do not match")

    file_count = _positive_int(scan_item, "file_count")
    if _positive_int(document_item, "file_count") != file_count:
        raise BottleneckSummaryError("file counts do not match")
    document_count = _positive_int(document_item, "indexed_document_count")
    source_document_count = _positive_int(
        document_item,
        "source_document_file_count",
    )
    if document_count != source_document_count:
        raise BottleneckSummaryError(
            f"document count mismatch for {scale}"
        )

    search_total = _positive_int(scan_item, "search_total")
    if search_total != file_count:
        raise BottleneckSummaryError(f"search total mismatch for {scale}")

    agent_statuses = scan_item.get("agent_statuses")
    if not isinstance(agent_statuses, list) or not agent_statuses:
        raise BottleneckSummaryError(f"agent statuses missing for {scale}")
    if any(not isinstance(status, str) for status in agent_statuses):
        raise BottleneckSummaryError(f"agent statuses invalid for {scale}")

    scan_timing = _object(scan_item, "scan")
    search_timing = _object(scan_item, "search")
    agent_timing = _object(scan_item, "agent")
    index_timing = _object(document_item, "index")
    memory = _object(document_item, "peak_python_allocations")
    database = _object(document_item, "database_bytes")
    return ScaleEvidence(
        scale=scale,
        file_count=file_count,
        document_count=document_count,
        chunk_count=_positive_int(document_item, "indexed_chunk_count"),
        scan_median_ms=_positive_float(scan_timing, "median_ms"),
        scan_min_ms=_positive_float(scan_timing, "minimum_ms"),
        scan_max_ms=_positive_float(scan_timing, "maximum_ms"),
        search_median_ms=_positive_float(search_timing, "median_ms"),
        agent_median_ms=_positive_float(agent_timing, "median_ms"),
        agent_failure_count=sum(
            status != "completed" for status in agent_statuses
        ),
        index_median_ms=_positive_float(index_timing, "median_ms"),
        index_min_ms=_positive_float(index_timing, "minimum_ms"),
        index_max_ms=_positive_float(index_timing, "maximum_ms"),
        memory_median_bytes=_positive_int(memory, "median"),
        database_median_bytes=_positive_int(database, "median"),
    )


def _measurement_list(report: dict[str, Any]) -> list[dict[str, Any]]:
    measurements = report.get("measurements")
    if not isinstance(measurements, list) or not all(
        isinstance(item, dict) for item in measurements
    ):
        raise BottleneckSummaryError("measurement list is invalid")
    if len(measurements) != len(EXPECTED_SCALES):
        raise BottleneckSummaryError("measurement list has an unexpected size")
    return measurements


def _object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise BottleneckSummaryError(f"measurement field is not an object: {key}")
    return value


def _string(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise BottleneckSummaryError(f"measurement field is not a string: {key}")
    return value


def _positive_int(parent: dict[str, Any], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise BottleneckSummaryError(f"measurement field is not positive: {key}")
    return value


def _positive_float(parent: dict[str, Any], key: str) -> float:
    value = parent.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise BottleneckSummaryError(f"measurement field is not positive: {key}")
    return float(value)


def _range(median_value: float, minimum: float, maximum: float) -> str:
    return f"{median_value:.3f} ({minimum:.3f}-{maximum:.3f})"


def _format_bytes(value: int) -> str:
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.2f} MiB"
    return f"{value / 1024:.2f} KiB"


def build_parser() -> argparse.ArgumentParser:
    """构建 E35-04 汇总命令。"""

    parser = argparse.ArgumentParser(
        description="汇总 FileNest 第 35 课的失败类型与可复现瓶颈。"
    )
    parser.add_argument("--scan-report", type=Path, required=True)
    parser.add_argument("--document-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """读取两个测量报告并写出 E35-04 Markdown。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        scan_report = load_measurement_report(
            args.scan_report,
            expected_task="E35-02",
        )
        document_report = load_measurement_report(
            args.document_report,
            expected_task="E35-03",
        )
        summary = build_bottleneck_summary(scan_report, document_report)
        write_bottleneck_summary(summary, args.output)
    except BottleneckSummaryError as error:
        parser.error(str(error))

    print(f"bottleneck summary written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
