import json
from pathlib import Path

import pytest

from backend.app.safe_organization_evaluation_cli import main


def test_cli_generates_safe_report_with_zero_unapproved_disk_changes(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "safe-organization-run"
    report_path = tmp_path / "safe-organization-milestone.md"

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    summary = json.loads(
        (output_dir / "evaluation-result.json").read_text(encoding="utf-8")
    )
    assert summary["case_total"] == 9
    assert summary["passed_cases"] == 9
    assert summary["safe_chain_cases"] == 5
    assert summary["unapproved_cases"] == 4
    assert summary["unauthorized_disk_changes"] == 0
    assert set(summary["source_sha256"]) == {
        "tests/test_safe_organization_end_to_end.py",
        "tests/test_approval_disk_immutability.py",
    }

    report = report_path.read_text(encoding="utf-8")
    assert "# FileNest 第 26 课：安全整理 Agent 综合评测" in report
    assert "| 综合安全场景通过 | 9/9 |" in report
    assert "| 未经审批磁盘快照一致 | 4/4 |" in report
    assert "| 未经审批磁盘变更 | 0 |" in report
    assert "backend.app.safe_organization_evaluation_cli" in report
    assert str(tmp_path) not in report


def test_cli_refuses_existing_evidence_without_overwriting(
    tmp_path: Path,
) -> None:
    existing_report = tmp_path / "existing-report.md"
    existing_report.write_text("existing report", encoding="utf-8")
    unused_output = tmp_path / "unused-output"

    with pytest.raises(FileExistsError, match="拒绝覆盖历史证据"):
        main(
            [
                "--output-dir",
                str(unused_output),
                "--report-path",
                str(existing_report),
            ]
        )

    assert existing_report.read_text(encoding="utf-8") == "existing report"
    assert not unused_output.exists()

    existing_output = tmp_path / "existing-output"
    existing_output.mkdir()
    marker = existing_output / "marker.txt"
    marker.write_text("existing evidence", encoding="utf-8")
    unused_report = tmp_path / "unused-report.md"

    with pytest.raises(FileExistsError, match="拒绝覆盖历史证据"):
        main(
            [
                "--output-dir",
                str(existing_output),
                "--report-path",
                str(unused_report),
            ]
        )

    assert marker.read_text(encoding="utf-8") == "existing evidence"
    assert not unused_report.exists()
