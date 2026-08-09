from pathlib import Path

import pytest

from backend.app.scale_architecture_decision import (
    DecisionOutputError,
    build_architecture_decision,
    file_sha256,
    load_bottleneck_report,
    write_architecture_decision,
)
from backend.app.scale_bottleneck_summary import load_measurement_report


PROJECT_ROOT = Path(__file__).parents[1]
SCAN_REPORT = PROJECT_ROOT / "backend" / "evaluation" / (
    "scale_measurements_e35-02.json"
)
DOCUMENT_REPORT = PROJECT_ROOT / "backend" / "evaluation" / (
    "scale_measurements_e35-03.json"
)
BOTTLENECK_REPORT = PROJECT_ROOT / "backend" / "evaluation" / (
    "scale_measurements_e35-04.md"
)


def test_decision_keeps_sqlite_without_unmeasured_index_changes() -> None:
    scan_report = load_measurement_report(
        SCAN_REPORT,
        expected_task="E35-02",
    )
    document_report = load_measurement_report(
        DOCUMENT_REPORT,
        expected_task="E35-03",
    )
    load_bottleneck_report(BOTTLENECK_REPORT)
    decision = build_architecture_decision(
        scan_report,
        document_report,
        evidence_hashes={
            SCAN_REPORT.name: file_sha256(SCAN_REPORT),
            DOCUMENT_REPORT.name: file_sha256(DOCUMENT_REPORT),
            BOTTLENECK_REPORT.name: file_sha256(BOTTLENECK_REPORT),
        },
    )

    large = scan_report["measurements"][-1]
    expected_search = large["search"]["median_ms"]
    assert "当前继续使用 SQLite" in decision
    assert "本课不新增数据库索引" in decision
    assert "本课不迁移 PostgreSQL" in decision
    assert f"固定文件搜索中位数：{expected_search:.3f} ms" in decision
    assert "规模化 `knowledge_search`/全文检索未在 E35-02 中测量" in decision
    assert "后续必须由新的失败、查询计划或并发数据触发复审" in decision


def test_architecture_decision_refuses_existing_output(tmp_path: Path) -> None:
    output_path = tmp_path / "decision.md"
    output_path.write_text("keep me", encoding="utf-8")

    with pytest.raises(
        DecisionOutputError,
        match="must not already exist",
    ):
        write_architecture_decision("new decision", output_path)

    assert output_path.read_text(encoding="utf-8") == "keep me"
