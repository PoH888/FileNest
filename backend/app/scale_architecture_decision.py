"""第 35 课 E35-05 的 SQLite、索引和 PostgreSQL 架构决定。"""

import argparse
from hashlib import sha256
from pathlib import Path
from typing import Any

from .scale_bottleneck_summary import (
    BottleneckSummaryError,
    load_measurement_report,
)


class ArchitectureDecisionError(ValueError):
    """架构决定输入证据或输出路径不符合契约。"""


class DecisionOutputError(ArchitectureDecisionError):
    """架构决定报告输出路径不安全或不可写。"""


def load_bottleneck_report(path: Path) -> str:
    """读取并确认 E35-04 瓶颈汇总包含必要边界。"""

    try:
        report = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise ArchitectureDecisionError(
            f"unable to read bottleneck report: {path}"
        ) from error

    required_markers = (
        "E35-04",
        "## 可复现瓶颈",
        "## 当前不构成瓶颈的行动",
        "不提前决定索引、PostgreSQL 或继续 SQLite",
    )
    if any(marker not in report for marker in required_markers):
        raise ArchitectureDecisionError(
            "bottleneck report is missing required evidence boundaries"
        )
    return report


def build_architecture_decision(
    scan_report: dict[str, Any],
    document_report: dict[str, Any],
    *,
    evidence_hashes: dict[str, str],
) -> str:
    """依据 E35 测量结果生成不修改数据库实现的架构决定。"""

    if scan_report.get("task") != "E35-02":
        raise ArchitectureDecisionError("scan report must belong to E35-02")
    if document_report.get("task") != "E35-03":
        raise ArchitectureDecisionError(
            "document report must belong to E35-03"
        )
    if not evidence_hashes or any(
        len(digest) != 64 for digest in evidence_hashes.values()
    ):
        raise ArchitectureDecisionError("evidence SHA-256 values are invalid")

    scan_large = _large_measurement(scan_report)
    document_large = _large_measurement(document_report)
    if _positive_int(scan_large, "file_count") != _positive_int(
        document_large,
        "file_count",
    ):
        raise ArchitectureDecisionError(
            "large-scale file counts do not match"
        )

    file_count = _positive_int(scan_large, "file_count")
    document_count = _positive_int(
        document_large,
        "indexed_document_count",
    )
    chunk_count = _positive_int(document_large, "indexed_chunk_count")
    scan_ms = _median_ms(scan_large, "scan")
    search_ms = _median_ms(scan_large, "search")
    agent_ms = _median_ms(scan_large, "agent")
    index_ms = _median_ms(document_large, "index")
    memory_bytes = _median_int(
        document_large,
        "peak_python_allocations",
    )
    database_bytes = _median_int(document_large, "database_bytes")

    agent_statuses = scan_large.get("agent_statuses")
    if not isinstance(agent_statuses, list) or any(
        status != "completed" for status in agent_statuses
    ):
        raise ArchitectureDecisionError(
            "large-scale Agent measurements were not all completed"
        )

    lines = [
        "# FileNest 第 35 课：规模证据架构决定（E35-05）",
        "",
        "## 决定",
        "",
        "1. **当前继续使用 SQLite。**",
        "2. **本课不新增数据库索引。**",
        "3. **本课不迁移 PostgreSQL。**",
        "4. 后续若优化，优先分析扫描同步和文档索引的批处理/对象生命周期，而不是先更换数据库。",
        "",
        "## 决定所依据的证据",
        "",
        f"- large：{file_count} 个文件、{document_count} 个文档、{chunk_count} 个 Chunk。",
        f"- 文件扫描与 FileEntry 持久化中位数：{scan_ms:.3f} ms。",
        f"- 固定文件搜索中位数：{search_ms:.3f} ms。",
        f"- 离线 Fake Agent 查询中位数：{agent_ms:.3f} ms，所有重复均 completed。",
        f"- 文档索引中位数：{index_ms:.3f} ms。",
        f"- Python 峰值分配中位数：{_format_bytes(memory_bytes)}。",
        f"- SQLite 主库及旁车文件中位数：{_format_bytes(database_bytes)}。",
        "- 正式固定测量没有观察到数据库锁、写入失败或查询失败。",
        "",
        "## 为什么现在不新增索引",
        "",
        "- 当前文件搜索使用工作区过滤加 `icontains` 子串匹配；在 10,000 个文件下中位数仍低于 4 ms，没有形成测量瓶颈。",
        "- `file_entries` 已有 `(workspace_id, relative_path)` 唯一约束，文档和 Chunk 也已有身份/关联索引；没有证据支持再增加普通 B-tree 索引。",
        "- 以 `%keyword%` 形式进行的前置通配符子串查询通常不能直接从普通 B-tree 获益；在没有查询计划和退化证据时新增索引只会增加写入与存储成本。",
        "- 规模化 `knowledge_search`/全文检索未在 E35-02 中测量，因此本决定不宣称文档全文检索无需专门索引；该问题需要独立证据。",
        "",
        "## 为什么现在不迁移 PostgreSQL",
        "",
        "- 当前证据来自本地、单用户、临时数据库工作负载；SQLite 数据量约 25.95 MiB，搜索仍为毫秒级，且没有锁冲突或并发失败。",
        "- 当前最重阶段是文件系统扫描和 Python 文档解析/切分/ORM 持久化。仅更换数据库不能直接消除这些成本。",
        "- 尚未测量多用户并发写入、远程服务部署、连接池压力或锁竞争；没有足够证据承担迁移复杂度。",
        "",
        "## 重新评估条件",
        "",
        "出现以下任一证据时，重新评估索引或 PostgreSQL：",
        "",
        "- 代表性并发测试稳定复现 `database is locked`、写吞吐不足或事务等待；",
        "- 规模化文件搜索或 `knowledge_search` 超出以后明确制定的产品延迟目标；",
        "- 查询计划证明具体 SQL 可以从一个明确索引获益，并通过新增索引前后对照验证；",
        "- 产品从本地单用户应用转为多用户远程服务，出现独立数据库服务、备份、权限或高可用需求；",
        "- SQLite 文件大小、迁移时间或维护成本成为可重复的实际问题。",
        "",
        "## 当前不做的事",
        "",
        "- 不新增 Alembic migration。",
        "- 不修改 SQLAlchemy engine 或数据库 URL。",
        "- 不引入 PostgreSQL 驱动或服务依赖。",
        "- 不实现未经规模查询证据支持的 FTS、额外 B-tree 或向量数据库。",
        "",
        "## 证据文件 SHA-256",
        "",
    ]
    for name, digest in sorted(evidence_hashes.items()):
        lines.append(f"- `{name}`：`{digest}`")

    lines.extend(
        [
            "",
            "## 结论",
            "",
            "现有证据支持“继续 SQLite、暂不新增索引、暂不迁移 PostgreSQL”。这是当前规模和工作负载下的可撤销决定，不是永久承诺；后续必须由新的失败、查询计划或并发数据触发复审。",
            "",
            "## 复现决定报告",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe -m backend.app.scale_architecture_decision `",
            "  --scan-report backend\\evaluation\\scale_measurements_e35-02.json `",
            "  --document-report backend\\evaluation\\scale_measurements_e35-03.json `",
            "  --bottleneck-report backend\\evaluation\\scale_measurements_e35-04.md `",
            "  --output backend\\evaluation\\scale_architecture_decision_e35-05-rerun.md",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def write_architecture_decision(decision: str, output_path: Path) -> None:
    """只写入尚不存在的架构决定报告。"""

    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise DecisionOutputError(
            "architecture decision output must not already exist"
        )
    try:
        output_path.write_text(decision, encoding="utf-8", newline="\n")
    except OSError as error:
        raise DecisionOutputError(
            "architecture decision output could not be written"
        ) from error


def file_sha256(path: Path) -> str:
    """计算决定输入文件的 SHA-256，固定证据来源。"""

    try:
        return sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise ArchitectureDecisionError(
            f"unable to hash evidence file: {path}"
        ) from error


def _large_measurement(report: dict[str, Any]) -> dict[str, Any]:
    measurements = report.get("measurements")
    if not isinstance(measurements, list):
        raise ArchitectureDecisionError("measurement list is missing")
    matches = [
        item
        for item in measurements
        if isinstance(item, dict) and item.get("scale") == "large"
    ]
    if len(matches) != 1:
        raise ArchitectureDecisionError(
            "measurement report must contain exactly one large scale"
        )
    return matches[0]


def _median_ms(measurement: dict[str, Any], field: str) -> float:
    summary = measurement.get(field)
    if not isinstance(summary, dict):
        raise ArchitectureDecisionError(f"missing timing summary: {field}")
    value = summary.get("median_ms")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ArchitectureDecisionError(f"invalid timing median: {field}")
    return float(value)


def _median_int(measurement: dict[str, Any], field: str) -> int:
    summary = measurement.get(field)
    if not isinstance(summary, dict):
        raise ArchitectureDecisionError(f"missing numeric summary: {field}")
    return _positive_int(summary, "median")


def _positive_int(parent: dict[str, Any], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ArchitectureDecisionError(f"invalid positive integer: {key}")
    return value


def _format_bytes(value: int) -> str:
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.2f} MiB"
    return f"{value / 1024:.2f} KiB"


def build_parser() -> argparse.ArgumentParser:
    """构建 E35-05 决策报告命令。"""

    parser = argparse.ArgumentParser(
        description="根据 FileNest 第 35 课规模证据生成数据库架构决定。"
    )
    parser.add_argument("--scan-report", type=Path, required=True)
    parser.add_argument("--document-report", type=Path, required=True)
    parser.add_argument("--bottleneck-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """读取 E35 证据并写出 E35-05 架构决定。"""

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
        load_bottleneck_report(args.bottleneck_report)
        evidence_hashes = {
            args.scan_report.name: file_sha256(args.scan_report),
            args.document_report.name: file_sha256(args.document_report),
            args.bottleneck_report.name: file_sha256(args.bottleneck_report),
        }
        decision = build_architecture_decision(
            scan_report,
            document_report,
            evidence_hashes=evidence_hashes,
        )
        write_architecture_decision(decision, args.output)
    except (ArchitectureDecisionError, BottleneckSummaryError) as error:
        parser.error(str(error))

    print(f"architecture decision written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
