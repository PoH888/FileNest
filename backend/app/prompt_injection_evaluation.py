"""文档提示注入评测数据的严格、不可执行契约。"""

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from backend.app.agent_evaluation import EvaluationFileSpec


PromptInjectionCategory = Literal[
    "instruction_override",
    "write_tool_escalation",
    "workspace_escape",
]
PromptInjectionExpectedBoundary = Literal[
    "document_is_untrusted_data",
    "write_capability_not_granted",
    "workspace_scope_not_expanded",
]

_EXPECTED_BOUNDARY_BY_CATEGORY: dict[
    PromptInjectionCategory,
    PromptInjectionExpectedBoundary,
] = {
    "instruction_override": "document_is_untrusted_data",
    "write_tool_escalation": "write_capability_not_granted",
    "workspace_escape": "workspace_scope_not_expanded",
}


class PromptInjectionDatasetError(ValueError):
    """提示注入测试集无法读取或不符合固定契约。"""


class PromptInjectionCase(BaseModel):
    """一条仅作为数据保存的文档提示注入场景。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(
        min_length=3,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    category: PromptInjectionCategory
    description: str = Field(min_length=1, max_length=200)
    source: EvaluationFileSpec
    question: str = Field(min_length=1, max_length=2_000)
    expected_boundary: PromptInjectionExpectedBoundary

    @field_validator("description", "question")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("prompt injection text must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_case_boundary(self) -> "PromptInjectionCase":
        """固定类别与预期边界，避免测试数据自行改写安全目标。"""

        if not self.source.content.strip():
            raise ValueError("prompt injection source content must not be blank")
        if self.expected_boundary != _EXPECTED_BOUNDARY_BY_CATEGORY[self.category]:
            raise ValueError("expected boundary must match the attack category")
        return self


class PromptInjectionDataset(BaseModel):
    """版本化的三场景文档提示注入测试集。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    name: str = Field(min_length=1, max_length=100)
    cases: tuple[PromptInjectionCase, ...] = Field(min_length=3, max_length=3)

    @field_validator("name")
    @classmethod
    def reject_blank_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("dataset name must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_representative_cases(self) -> "PromptInjectionDataset":
        """每类只保留一个代表场景，阻止测试集无边界扩张。"""

        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("prompt injection case ids must be unique")

        categories = [case.category for case in self.cases]
        if set(categories) != set(_EXPECTED_BOUNDARY_BY_CATEGORY):
            raise ValueError("prompt injection categories must be complete and unique")

        source_paths = [case.source.relative_path for case in self.cases]
        if len(set(source_paths)) != len(source_paths):
            raise ValueError("prompt injection source paths must be unique")
        return self


def load_prompt_injection_dataset(path: Path) -> PromptInjectionDataset:
    """从 UTF-8 JSON 读取测试数据，不执行其中的任何文档内容。"""

    try:
        raw_data = path.read_text(encoding="utf-8")
    except OSError as error:
        raise PromptInjectionDatasetError("无法读取提示注入测试集") from error

    try:
        return PromptInjectionDataset.model_validate_json(raw_data)
    except ValidationError as error:
        raise PromptInjectionDatasetError("提示注入测试集格式无效") from error
