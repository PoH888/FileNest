import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.agent_contract_dataset import (
    AgentContractDatasetError,
    AgentContractFixtureFile,
    load_agent_contract_dataset,
)


DATASET_PATH = (
    Path(__file__).parents[1]
    / "backend"
    / "evaluation"
    / "agent_contract_v1.json"
)


def test_agent_contract_dataset_has_base_coverage_and_directory_cases() -> None:
    dataset = load_agent_contract_dataset(DATASET_PATH)

    assert dataset.dataset_version == "agent-contract-v1"
    assert dataset.fixture.seed == 42
    assert dataset.fixture.root_policy == "fresh_temporary_workspace"
    assert len(dataset.cases) == 29
    assert Counter(case.category for case in dataset.cases) == {
        "tool_selection": 7,
        "argument_validity": 5,
        "proposal_validity": 4,
        "security_boundary": 9,
        "rag_citation": 4,
    }


def test_agent_contract_cases_define_inputs_expectations_and_safety() -> None:
    dataset = load_agent_contract_dataset(DATASET_PATH)
    all_paths = {file.relative_path for file in dataset.fixture.files}
    all_tags = {tag for case in dataset.cases for tag in case.tags}

    assert {
        "normal",
        "failure",
        "repeat_execution",
        "prompt_injection_document",
    } <= all_tags
    assert set(dataset.fixture.prompt_injection_paths) <= all_paths
    assert set(dataset.fixture.sensitive_paths) <= all_paths

    for case in dataset.cases:
        assert case.input.request_text
        assert case.fixture == dataset.fixture.fixture_id
        assert case.expected.allowed_results
        assert case.security_assertions.unauthorized_disk_changes == 0
        assert case.security_assertions.approval_required_before_write is True
        assert set(case.expected.expected_source_paths) <= all_paths
        for tool_call in case.expected.tool_calls:
            assert tool_call.name
            assert isinstance(tool_call.arguments, dict)


def test_agent_contract_dataset_rejects_unknown_fixture_reference(
    tmp_path: Path,
) -> None:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    payload["cases"][0]["fixture"] = "missing-fixture"
    invalid_path = tmp_path / "invalid-agent-contract.json"
    invalid_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(AgentContractDatasetError, match="格式无效"):
        load_agent_contract_dataset(invalid_path)


def test_agent_contract_fixture_rejects_windows_or_parent_paths() -> None:
    with pytest.raises(ValidationError):
        AgentContractFixtureFile(
            relative_path="..\\outside.txt",
            content="fixture",
        )

    with pytest.raises(ValidationError):
        AgentContractFixtureFile(
            relative_path="C:/outside.txt",
            content="fixture",
        )
