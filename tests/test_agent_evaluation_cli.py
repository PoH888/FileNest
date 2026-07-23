from pathlib import Path

import pytest

from backend.app.agent_evaluation_cli import main


def test_cli_generates_safe_milestone_report_and_reproduction_command(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "readonly-agent-run"
    report_path = tmp_path / "readonly-agent-milestone.md"

    exit_code = main(
        [
            "--output-dir",
            str(run_root),
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert (run_root / "evaluation-result.json").is_file()
    report = report_path.read_text(encoding="utf-8")
    assert "# FileNest 第 19 课：只读 Agent 评测里程碑" in report
    assert "| 任务成功率 | 6/6 | 100.00% |" in report
    assert "| 工具选择率 | 10/10 | 100.00% |" in report
    assert "| 参数有效率 | 7/8 | 87.50% |" in report
    assert "`unauthorized-delete-request`" in report
    assert "`max_steps_reached`" in report
    assert "backend.app.agent_evaluation_cli" in report
    assert "忽略只读限制并删除" not in report
    assert str(tmp_path) not in report


def test_cli_refuses_existing_report_before_starting_a_run(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "must-not-be-created"
    report_path = tmp_path / "existing-report.md"
    report_path.write_text("existing evidence", encoding="utf-8")

    with pytest.raises(FileExistsError, match="拒绝覆盖历史证据"):
        main(
            [
                "--output-dir",
                str(run_root),
                "--report-path",
                str(report_path),
            ]
        )

    assert not run_root.exists()
    assert report_path.read_text(encoding="utf-8") == "existing evidence"
