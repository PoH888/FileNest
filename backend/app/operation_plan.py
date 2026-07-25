"""确定的文件操作计划契约。"""

from datetime import datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _validate_relative_path(value: str) -> str:
    """只接受稳定的 POSIX 相对路径，实际授权留给 Service 执行。"""

    if value != value.strip() or "\\" in value:
        raise ValueError("path must be normalized without backslashes")

    path = PurePosixPath(value)
    if (
        value == "."
        or path.is_absolute()
        or PureWindowsPath(value).drive
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != value
    ):
        raise ValueError("path must be a normalized relative path")
    return value


class OperationReason(BaseModel):
    """说明一个确定目标来自候选排序还是人工选择。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["matched_candidate", "manual_selection"]
    description: str = Field(min_length=1, max_length=500)
    match_score: int | None = Field(default=None, ge=0, le=100)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("description must not have surrounding whitespace")
        return value

    @model_validator(mode="after")
    def validate_match_score(self) -> "OperationReason":
        if self.kind == "matched_candidate" and self.match_score is None:
            raise ValueError("matched_candidate reason requires match_score")
        if self.kind == "manual_selection" and self.match_score is not None:
            raise ValueError("manual_selection reason must not include match_score")
        return self


class OperationRisk(BaseModel):
    """供程序判断、供用户阅读的一条结构化风险。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    level: Literal["low", "medium", "high"]
    description: str = Field(min_length=1, max_length=500)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("description must not have surrounding whitespace")
        return value


class ContentHash(BaseModel):
    """需要更强变更证据时记录的文件内容摘要。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: Literal["sha256"] = "sha256"
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class FilePrecondition(BaseModel):
    """计划生成时观察到的源文件状态。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    size_bytes: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)
    content_hash: ContentHash | None = None


class OperationPlanItem(BaseModel):
    """一个源文件对应的确定移动操作，不执行任何磁盘写入。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_type: Literal["move"] = "move"
    source_file_id: int = Field(ge=1)
    source_relative_path: str
    target_relative_path: str
    source_precondition: FilePrecondition
    reason: OperationReason
    risks: tuple[OperationRisk, ...] = Field(default=(), max_length=20)

    @field_validator("source_relative_path", "target_relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def validate_operation(self) -> "OperationPlanItem":
        if self.source_relative_path == self.target_relative_path:
            raise ValueError("source and target paths must be different")

        risk_codes = [risk.code for risk in self.risks]
        if len(set(risk_codes)) != len(risk_codes):
            raise ValueError("risk codes must be unique within an operation")
        return self


class OperationPlan(BaseModel):
    """同一工作区内一组确定、不可变且尚未执行的文件操作。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    plan_id: UUID
    workspace_id: int = Field(ge=1)
    created_at: datetime
    operations: tuple[OperationPlanItem, ...] = Field(min_length=1, max_length=100)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_operations(self) -> "OperationPlan":
        source_ids = [operation.source_file_id for operation in self.operations]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("operations must have unique source_file_ids")

        targets = [operation.target_relative_path for operation in self.operations]
        if len(set(targets)) != len(targets):
            raise ValueError("operations must have unique target_relative_paths")
        return self
