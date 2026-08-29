import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import xml.etree.ElementTree as ElementTree

import pytest

from backend.app.agent_contract_dataset import load_agent_contract_dataset
from backend.app.agent_contract_runner import (
    AgentContractRunDirectoryError,
    AgentContractEvaluationSummary,
    run_agent_contract_evaluation,
)
from backend.app.agent_evaluation import EvaluationVersionInfo
from backend.app.model_client import ModelMessage, ModelResponse


DATASET_PATH = (
    Path(__file__).parents[1]
    / "backend"
    / "evaluation"
    / "agent_contract_v1.json"
)


def _version_info(model_version: str = "scripted_fake") -> EvaluationVersionInfo:
    return EvaluationVersionInfo(
        prompt_version="agent-contract-test-prompt-v1",
        model_version=model_version,
        git_commit="a" * 40,
        evaluation_dataset_version="agent-contract-v1",
        timestamp=datetime(2026, 9, 3, tzinfo=timezone.utc),
    )


def test_contract_runner_executes_all_twenty_cases_and_writes_isolated_evidence(
    tmp_path: Path,
) -> None:
    dataset = load_agent_contract_dataset(DATASET_PATH)
    run_root = tmp_path / "agent-contract-run"

    summary = run_agent_contract_evaluation(
        dataset,
        run_root,
        dataset_sha256=sha256(DATASET_PATH.read_bytes()).hexdigest(),
        version_info=_version_info(),
    )

    assert len(summary.cases) == 20
    assert summary.metrics.end_to_end_successes == 20
    assert summary.metrics.end_to_end_success_rate == 1
    assert summary.metrics.tool_selection_accuracy == 1
    assert summary.metrics.argument_accuracy == 1
    assert summary.metrics.proposal_validity_rate == 1
    assert summary.metrics.approval_interception_rate == 1
    assert summary.metrics.unauthorized_disk_changes == 0
    assert summary.metrics.unauthorized_disk_changes_gate_passed is True
    assert summary.metrics.citation_coverage == 1
    assert summary.metrics.no_evidence_refusal_rate == 1

    for name in (
        "evaluation-result.json",
        "evaluation-summary.md",
        "junit.xml",
        "run-metadata.json",
    ):
        assert (run_root / name).is_file()
    assert (run_root / "evaluation.db").is_file()
    assert (run_root / "workflow-checkpoints.sqlite").is_file()
    assert (run_root / "workspace" / "private" / ".env").is_file()

    saved_summary = AgentContractEvaluationSummary.model_validate_json(
        (run_root / "evaluation-result.json").read_text(encoding="utf-8")
    )
    assert saved_summary == summary
    report = (run_root / "evaluation-summary.md").read_text(encoding="utf-8")
    assert "数据集 SHA-256" in report
    assert "Git commit" in report
    assert "evaluation-summary.md" in report
    assert "查找文件名包含" not in report
    assert str(tmp_path) not in report

    metadata = json.loads(
        (run_root / "run-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["case_count"] == 20
    assert metadata["failure_case_ids"] == []
    assert metadata["model_type"] == "deterministic_scripted_fake"
    assert "查找文件名包含" not in json.dumps(metadata, ensure_ascii=False)

    junit = ElementTree.parse(run_root / "junit.xml").getroot()
    assert junit.attrib["tests"] == "20"
    assert junit.attrib["failures"] == "0"


def test_contract_runner_uses_five_to_eight_case_subset_for_real_model(
    tmp_path: Path,
) -> None:
    dataset = load_agent_contract_dataset(DATASET_PATH)

    class StubRealModelClient:
        def complete(self, *, messages: tuple, tools: tuple) -> ModelResponse:
            return ModelResponse(
                message=ModelMessage(role="assistant", content="stub"),
                finish_reason="stop",
                model_provider="test-provider",
                model_name="test-model",
            )

    case_ids = tuple(case.case_id for case in dataset.cases[:5])
    summary = run_agent_contract_evaluation(
        dataset,
        tmp_path / "real-contract-run",
        dataset_sha256=sha256(DATASET_PATH.read_bytes()).hexdigest(),
        version_info=_version_info("test-model"),
        model_source="real_model",
        model_client_factory=lambda: StubRealModelClient(),
        model_provider="test-provider",
        case_ids=case_ids,
    )

    assert summary.model_source == "real_model"
    assert summary.model_provider == "test-provider"
    assert summary.selected_case_ids == case_ids
    assert len(summary.cases) == 5
    assert summary.metrics.end_to_end_success_rate == 0
    assert summary.metrics.usage_case_count == 0


def test_contract_runner_refuses_reusing_a_run_directory(tmp_path: Path) -> None:
    dataset = load_agent_contract_dataset(DATASET_PATH)
    run_root = tmp_path / "existing-run"
    run_root.mkdir()

    with pytest.raises(AgentContractRunDirectoryError):
        run_agent_contract_evaluation(
            dataset,
            run_root,
            dataset_sha256=sha256(DATASET_PATH.read_bytes()).hexdigest(),
            version_info=_version_info(),
        )
