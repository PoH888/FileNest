"""T4 固定 Agent 合同评测数据集的严格契约。"""

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)


AgentContractCategory = Literal[
    "tool_selection",
    "argument_validity",
    "proposal_validity",
    "security_boundary",
    "rag_citation",
]


class AgentContractDatasetError(ValueError):
    """固定 Agent 合同评测数据无法读取或不符合契约。"""


class AgentContractFixtureFile(BaseModel):
    """固定临时工作区中的一个 POSIX 相对路径文本文件。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(
        min_length=1,
        max_length=500,
        pattern=r"^[^\\]+$",
    )
    content: str = Field(min_length=1)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        """只接受规范的 POSIX 相对路径，禁止测试数据越出临时目录。"""

        if value != value.strip() or "\\" in value:
            raise ValueError("fixture paths must use normalized POSIX form")

        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or PureWindowsPath(value).drive
            or any(part in {"", ".", ".."} for part in path.parts)
            or str(path) != value
        ):
            raise ValueError("fixture paths must stay inside the workspace")
        return value


class AgentContractFixture(BaseModel):
    """可重复物化到全新临时目录的固定评测 fixture。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fixture_id: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    seed: int = Field(ge=0)
    workspace_id: int = Field(ge=1)
    workspace_name: str = Field(min_length=1, max_length=100)
    foreign_workspace_id: int = Field(ge=1)
    files: tuple[AgentContractFixtureFile, ...] = Field(min_length=1)
    prompt_injection_paths: tuple[str, ...] = ()
    sensitive_paths: tuple[str, ...] = ()
    root_policy: Literal["fresh_temporary_workspace"]

    @field_validator("workspace_name")
    @classmethod
    def reject_blank_workspace_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("fixture workspace name must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_fixture_references(self) -> "AgentContractFixture":
        file_paths = [file.relative_path for file in self.files]
        if len(set(file_paths)) != len(file_paths):
            raise ValueError("fixture file paths must be unique")

        path_set = set(file_paths)
        for reference_group in (
            self.prompt_injection_paths,
            self.sensitive_paths,
        ):
            if any(path not in path_set for path in reference_group):
                raise ValueError("fixture references an unknown file path")
            if len(set(reference_group)) != len(reference_group):
                raise ValueError("fixture file references must be unique")
        if self.workspace_id == self.foreign_workspace_id:
            raise ValueError("fixture workspace ids must be different")
        return self


class AgentContractInput(BaseModel):
    """一个固定合同用例的用户输入。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: int = Field(ge=1)
    request_text: str = Field(min_length=1, max_length=2_000)

    @field_validator("request_text")
    @classmethod
    def reject_blank_request(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("contract request must not be blank")
        return normalized


class AgentContractToolCall(BaseModel):
    """供应商无关的预期工具调用和参数。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class AgentContractExpectedOutcome(BaseModel):
    """用例允许的结果、工具轨迹和引用期望。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_calls: tuple[AgentContractToolCall, ...] = ()
    allowed_results: tuple[str, ...] = Field(min_length=1)
    expected_source_paths: tuple[str, ...] = ()
    expected_proposal_operation: str | None = Field(default=None, max_length=100)

    @field_validator("allowed_results")
    @classmethod
    def validate_allowed_results(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not result.strip() for result in value):
            raise ValueError("allowed results must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("allowed results must be unique")
        return value

    @field_validator("expected_proposal_operation")
    @classmethod
    def normalize_proposal_operation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("proposal operation must not be blank")
        return normalized


class AgentContractSecurityAssertions(BaseModel):
    """每个合同用例必须携带的安全结果断言。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unauthorized_disk_changes: Literal[0] = 0
    approval_required_before_write: bool = True
    forbidden_tool_names: tuple[str, ...] = ("approve", "execute", "undo")
    workspace_scope: Literal[
        "current_only",
        "no_scope_expansion",
        "not_applicable",
    ] = "current_only"
    sensitive_path_action: Literal["deny", "not_applicable"] = "not_applicable"
    repeat_execution: Literal[
        "not_applicable",
        "idempotent",
        "no_duplicate_effect",
    ] = "not_applicable"

    @field_validator("forbidden_tool_names")
    @classmethod
    def validate_forbidden_tool_names(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("forbidden tool names must be unique and non-empty")
        if any(
            not name or not name.replace("_", "a").isalnum()
            for name in value
        ):
            raise ValueError("forbidden tool names must be identifiers")
        return value


class AgentContractCase(BaseModel):
    """一条带输入、fixture 引用、期望和安全断言的合同用例。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    category: AgentContractCategory
    tags: tuple[str, ...] = Field(min_length=1)
    description: str = Field(min_length=1, max_length=300)
    input: AgentContractInput
    fixture: str = Field(min_length=3, max_length=100)
    expected: AgentContractExpectedOutcome
    security_assertions: AgentContractSecurityAssertions
    max_steps: int = Field(default=8, ge=1, le=20)
    cancel_before_run: bool = False

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not tag.strip() for tag in value):
            raise ValueError("case tags must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("case tags must be unique")
        return value

    @field_validator("description")
    @classmethod
    def reject_blank_description(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("case description must not be blank")
        return normalized


class AgentContractDataset(BaseModel):
    """固定 Agent 合同数据集，保留五类至少四条的基础覆盖。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_version: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    fixture: AgentContractFixture
    cases: tuple[AgentContractCase, ...] = Field(min_length=20, max_length=100)

    @model_validator(mode="after")
    def validate_dataset_contract(self) -> "AgentContractDataset":
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("contract case ids must be unique")

        category_counts = {
            category: sum(case.category == category for case in self.cases)
            for category in (
                "tool_selection",
                "argument_validity",
                "proposal_validity",
                "security_boundary",
                "rag_citation",
            )
        }
        if any(count < 4 for count in category_counts.values()):
            raise ValueError("each contract category must contain at least four cases")

        for case in self.cases:
            if case.fixture != self.fixture.fixture_id:
                raise ValueError("case references an unknown fixture")
            if any(
                path not in {
                    file.relative_path for file in self.fixture.files
                }
                for path in case.expected.expected_source_paths
            ):
                raise ValueError("case references an unknown source path")
        return self


def load_agent_contract_dataset(path: Path) -> AgentContractDataset:
    """从 UTF-8 JSON 文件读取并严格校验 T4 合同数据集。"""

    try:
        raw_data = path.read_text(encoding="utf-8")
    except OSError as error:
        raise AgentContractDatasetError("无法读取 Agent 合同评测数据") from error

    try:
        return AgentContractDataset.model_validate_json(raw_data)
    except ValidationError as error:
        raise AgentContractDatasetError("Agent 合同评测数据格式无效") from error
