from pathlib import Path

import pytest

from backend.app.scale_bottleneck_summary import (
    SummaryOutputError,
    build_bottleneck_summary,
    load_measurement_report,
    write_bottleneck_summary,
)


PROJECT_ROOT = Path(__file__).parents[1]
SCAN_REPORT = PROJECT_ROOT / "backend" / "evaluation" / (
    "scale_measurements_e35-02.json"
)
DOCUMENT_REPORT = PROJECT_ROOT / "backend" / "evaluation" / (
    "scale_measurements_e35-03.json"
)


def test_summary_preserves_measurement_facts_and_identifies_bottlenecks() -> None:
    scan_report = load_measurement_report(
        SCAN_REPORT,
        expected_task="E35-02",
    )
    document_report = load_measurement_report(
        DOCUMENT_REPORT,
        expected_task="E35-03",
    )

    summary = build_bottleneck_summary(scan_report, document_report)

    assert "本次正式测量没有观察到产品行动失败" in summary
    assert "扫描与文件索引持久化随文件数近似线性增长" in summary
    assert "文档索引是当前最重的测量阶段" in summary
    assert "不提前决定索引、PostgreSQL 或继续 SQLite" in summary
    assert "10000/8000/16000" in summary
    assert "SQLite 大小" in summary


def test_bottleneck_summary_refuses_existing_output(tmp_path: Path) -> None:
    output_path = tmp_path / "summary.md"
    output_path.write_text("keep me", encoding="utf-8")

    with pytest.raises(
        SummaryOutputError,
        match="must not already exist",
    ):
        write_bottleneck_summary("new summary", output_path)

    assert output_path.read_text(encoding="utf-8") == "keep me"
