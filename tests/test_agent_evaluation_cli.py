import json
from pathlib import Path
import xml.etree.ElementTree as ElementTree

import pytest
from pydantic import ValidationError

import backend.app.agent_evaluation_cli as cli_module
from backend.app.agent_evaluation_cli import main
from backend.app.model_client import ModelMessage, ModelResponse


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
    assert (run_root / "evaluation-summary.md").is_file()
    assert (run_root / "junit.xml").is_file()
    assert (run_root / "run-metadata.json").is_file()
    assert (tmp_path / "evaluation-history.jsonl").is_file()
    assert len(
        (tmp_path / "evaluation-history.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 1
    report = report_path.read_text(encoding="utf-8")
    assert "# FileNest 第 19 课：只读 Agent 评测里程碑" in report
    assert "| 任务成功率 | 6/6 | 100.00% |" in report
    assert "| 工具选择率 | 10/10 | 100.00% |" in report
    assert "| 参数有效率 | 7/8 | 87.50% |" in report
    assert "`unauthorized-delete-request`" in report
    assert "`max_steps_reached`" in report
    assert "## Forbidden Tools" in report
    assert "- 检查结果：通过" in report
    assert "## Risk Constraint Evaluation" in report
    assert "- 失败用例：无" in report
    assert "- Prompt version：`readonly_agent_prompt_v1`" in report
    assert "- Model version：`scripted_fake`" in report
    assert "- Evaluation dataset version：`readonly_agent_v1`" in report
    assert "- Git commit：`" in report
    assert "- Timestamp：`2026-" in report
    assert "backend.app.agent_evaluation_cli" in report
    assert "忽略只读限制并删除" not in report
    assert str(tmp_path) not in report
    assert (run_root / "evaluation-summary.md").read_text(encoding="utf-8") == report

    metadata = json.loads(
        (run_root / "run-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["model_source"] == "scripted_fake"
    assert metadata["model_type"] == "deterministic_scripted_fake"
    assert metadata["model_provider"] is None
    assert len(metadata["dataset_sha256"]) == 64
    assert len(metadata["git_commit"]) >= 7
    assert metadata["failure_case_ids"] == []
    assert "忽略只读限制并删除" not in json.dumps(metadata, ensure_ascii=False)

    junit = ElementTree.parse(run_root / "junit.xml").getroot()
    assert junit.attrib["tests"] == "6"
    assert junit.attrib["failures"] == "0"


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


def test_cli_runs_explicit_real_model_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FILENEST_MODEL_PROVIDER", "test-provider")
    monkeypatch.setenv("FILENEST_MODEL_NAME", "test-model")
    monkeypatch.setenv("FILENEST_MODEL_API_KEY", "secret-for-test")

    class StubRealModelClient:
        def complete(self, *, messages: tuple, tools: tuple) -> ModelResponse:
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content="评测响应。",
                ),
                finish_reason="stop",
            )

    monkeypatch.setattr(
        cli_module,
        "OpenAICompatibleModelClient",
        lambda settings: StubRealModelClient(),
    )
    report_path = tmp_path / "real-model-report.md"

    assert (
        main(
            [
                "--model-source",
                "real_model",
                "--output-dir",
                str(tmp_path / "real-model-run"),
                "--report-path",
                str(report_path),
            ]
        )
        == 0
    )

    report = report_path.read_text(encoding="utf-8")
    assert "- 模型来源：`real_model`" in report
    assert "包含真实模型调用结果" in report
    metadata = json.loads(
        (tmp_path / "real-model-run" / "run-metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["model_provider"] == "test-provider"
    assert metadata["model_version"] == "test-model"
    assert metadata["model_type"] == "configured_real_model"
    assert "secret-for-test" not in json.dumps(metadata, ensure_ascii=False)


def test_cli_real_model_mode_requires_configuration_before_starting_a_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in (
        "FILENEST_MODEL_PROVIDER",
        "FILENEST_MODEL_NAME",
        "FILENEST_MODEL_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)

    with pytest.raises(ValidationError):
        main(
            [
                "--model-source",
                "real_model",
                "--output-dir",
                str(tmp_path / "must-not-be-created"),
                "--report-path",
                str(tmp_path / "real-model-report.md"),
            ]
        )

    assert not (tmp_path / "must-not-be-created").exists()


def test_cli_refuses_run_when_git_commit_cannot_be_saved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_git(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("git")

    monkeypatch.setattr(cli_module.subprocess, "run", missing_git)

    with pytest.raises(
        cli_module.EvaluationMetadataError,
        match="无法读取评测所需的 Git commit",
    ):
        main(
            [
                "--output-dir",
                str(tmp_path / "must-not-be-created"),
                "--report-path",
                str(tmp_path / "missing-git-report.md"),
            ]
        )

    assert not (tmp_path / "must-not-be-created").exists()
