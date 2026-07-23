"""只读 Agent 评测数据与固定工作区的最小契约。"""

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


EvaluationCaseCategory = Literal[
    "normal",
    "ambiguous",
    "no_result",
    "invalid_arguments",
    "unauthorized",
    "max_steps",
]
EvaluationRunStatus = Literal[
    "completed",
    "max_steps_reached",
    "timed_out",
    "cancelled",
    "failed",
]


class EvaluationDatasetError(ValueError):
    """评测数据无法读取或不符合固定契约。"""


class EvaluationWorkspaceMaterializationError(ValueError):
    """固定评测工作区无法在不覆盖现有数据的前提下创建。"""


class EvaluationFileSpec(BaseModel):
    """固定评测工作区中的一个 UTF-8 文本文件。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1, max_length=500)
    content: str = ""

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        """只接受规范的 POSIX 相对路径，避免测试数据越出临时目录。"""

        if value != value.strip() or "\\" in value:
            raise ValueError(
                "relative_path must be normalized without backslashes"
            )

        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or PureWindowsPath(value).drive
            or any(part in {"", ".", ".."} for part in path.parts)
            or str(path) != value
        ):
            raise ValueError("relative_path must stay inside the workspace")
        return value


class EvaluationWorkspaceSpec(BaseModel):
    """可重复物化到临时目录的固定评测工作区。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=100)
    files: tuple[EvaluationFileSpec, ...] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("workspace name must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_file_paths(self) -> "EvaluationWorkspaceSpec":
        """拒绝重复路径以及文件与目录互相占位的清单。"""

        paths = {file.relative_path for file in self.files}
        if len(paths) != len(self.files):
            raise ValueError("workspace file paths must be unique")

        for value in paths:
            parents = PurePosixPath(value).parents
            if any(str(parent) in paths for parent in parents if str(parent) != "."):
                raise ValueError("workspace file paths must not conflict")
        return self


class EvaluationToolExpectation(BaseModel):
    """一个预期工具调用及其安全结果断言。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    result_ok: bool
    error_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    data_subset: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_result_expectation(self) -> "EvaluationToolExpectation":
        if self.result_ok and self.error_code is not None:
            raise ValueError("successful tool result must not expect an error")
        if not self.result_ok and self.error_code is None:
            raise ValueError("failed tool result must expect an error code")
        if not self.result_ok and self.data_subset:
            raise ValueError("failed tool result must not expect data")
        return self


class EvaluationModelToolCallSpec(BaseModel):
    """固定模型响应中的一个供应商无关工具调用。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    arguments: dict[str, JsonValue]


class EvaluationModelResponseSpec(BaseModel):
    """供可复现评测使用的一次固定模型响应。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    finish_reason: Literal["stop", "tool_calls"]
    content: str | None = None
    tool_calls: tuple[EvaluationModelToolCallSpec, ...] = ()

    @model_validator(mode="after")
    def validate_response_state(self) -> "EvaluationModelResponseSpec":
        has_content = self.content is not None and bool(self.content.strip())
        if self.finish_reason == "stop" and (
            not has_content or self.tool_calls
        ):
            raise ValueError("stop response must contain only final text")
        if self.finish_reason == "tool_calls" and not self.tool_calls:
            raise ValueError("tool_calls response must request a tool")
        return self


class EvaluationCase(BaseModel):
    """一个供应商无关、可评分的只读 Agent 评测用例。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(
        min_length=3,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    category: EvaluationCaseCategory
    description: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=2_000)
    max_steps: int = Field(default=8, ge=1, le=20)
    expected_run_status: EvaluationRunStatus
    expected_tool_names: tuple[
        str,
        ...,
    ] = ()
    expected_tool_results: tuple[EvaluationToolExpectation, ...] = ()
    scripted_responses: tuple[EvaluationModelResponseSpec, ...] = Field(
        min_length=1
    )

    @field_validator("description", "prompt")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("evaluation text must not be blank")
        return normalized

    @field_validator("expected_tool_names")
    @classmethod
    def validate_expected_tool_names(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        for name in value:
            if (
                not name
                or not name.replace("_", "a").isalnum()
                or not name[0].islower()
            ):
                raise ValueError("expected tool names must use snake_case")
        return value


class EvaluationDataset(BaseModel):
    """一个带版本、固定工作区和评测用例的完整数据集。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    workspace: EvaluationWorkspaceSpec
    cases: tuple[EvaluationCase, ...] = ()

    @model_validator(mode="after")
    def validate_case_ids(self) -> "EvaluationDataset":
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("evaluation case ids must be unique")
        return self


def load_evaluation_dataset(path: Path) -> EvaluationDataset:
    """从 UTF-8 JSON 文件读取并严格校验评测数据。"""

    try:
        raw_data = path.read_text(encoding="utf-8")
    except OSError as error:
        raise EvaluationDatasetError("无法读取评测数据") from error

    try:
        return EvaluationDataset.model_validate_json(raw_data)
    except ValidationError as error:
        raise EvaluationDatasetError("评测数据格式无效") from error


def materialize_evaluation_workspace(
    workspace: EvaluationWorkspaceSpec,
    target_root: Path,
) -> Path:
    """在一个全新目录中物化固定文件，绝不覆盖调用方已有内容。"""

    if target_root.exists() or target_root.is_symlink():
        raise EvaluationWorkspaceMaterializationError(
            "评测工作区目标必须是尚不存在的目录"
        )

    target_root.mkdir(parents=True)
    resolved_root = target_root.resolve(strict=True)
    for file in workspace.files:
        destination = resolved_root.joinpath(
            *PurePosixPath(file.relative_path).parts
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(file.content, encoding="utf-8", newline="\n")

    return resolved_root
