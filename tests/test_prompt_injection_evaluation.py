import json
from pathlib import Path

import pytest

from backend.app.prompt_injection_evaluation import (
    PromptInjectionDatasetError,
    load_prompt_injection_dataset,
)


DATASET_PATH = (
    Path(__file__).parents[1]
    / "backend"
    / "evaluation"
    / "prompt_injection_v1.json"
)


def test_prompt_injection_dataset_has_three_representative_cases() -> None:
    dataset = load_prompt_injection_dataset(DATASET_PATH)

    assert dataset.schema_version == "1.0"
    assert [case.category for case in dataset.cases] == [
        "instruction_override",
        "write_tool_escalation",
        "workspace_escape",
    ]
    assert [case.expected_boundary for case in dataset.cases] == [
        "document_is_untrusted_data",
        "write_capability_not_granted",
        "workspace_scope_not_expanded",
    ]
    assert len({case.source.relative_path for case in dataset.cases}) == 3


def test_prompt_injection_dataset_rejects_duplicate_cases(
    tmp_path: Path,
) -> None:
    dataset_data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    dataset_data["cases"][1] = dataset_data["cases"][0]
    invalid_dataset = tmp_path / "duplicate-prompt-injection-cases.json"
    invalid_dataset.write_text(
        json.dumps(dataset_data, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(
        PromptInjectionDatasetError,
        match="提示注入测试集格式无效",
    ):
        load_prompt_injection_dataset(invalid_dataset)
