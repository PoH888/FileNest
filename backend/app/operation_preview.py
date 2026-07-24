"""文件整理预览的只读契约与候选排序适配。"""

from collections.abc import Sequence
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.matcher import get_best_matches


def _validate_relative_path(value: str) -> str:
    """只接受稳定的 POSIX 相对路径，实际授权仍由 Service 执行。"""

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


class OperationPreviewRequest(BaseModel):
    """一次只读整理预览所需的文件和候选目录。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: int = Field(ge=1)
    source_file_ids: tuple[int, ...] = Field(min_length=1, max_length=100)
    target_directories: tuple[str, ...] = Field(min_length=1, max_length=200)

    @field_validator("source_file_ids")
    @classmethod
    def validate_source_file_ids(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(file_id < 1 for file_id in value):
            raise ValueError("source_file_ids must contain positive integers")
        if len(set(value)) != len(value):
            raise ValueError("source_file_ids must be unique")
        return value

    @field_validator("target_directories")
    @classmethod
    def validate_target_directories(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        validated = tuple(_validate_relative_path(path) for path in value)
        if len(set(validated)) != len(validated):
            raise ValueError("target_directories must be unique")
        return validated


class OperationPreviewCandidate(BaseModel):
    """matcher 返回的一个候选目录及其排序分数。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_directory: str
    score: int = Field(ge=0, le=100)

    @field_validator("relative_directory")
    @classmethod
    def validate_relative_directory(cls, value: str) -> str:
        return _validate_relative_path(value)


class OperationPreviewItem(BaseModel):
    """单个源文件的候选排序；空候选表示没有有意义的匹配。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_file_id: int = Field(ge=1)
    source_relative_path: str
    candidates: tuple[OperationPreviewCandidate, ...] = ()

    @field_validator("source_relative_path")
    @classmethod
    def validate_source_relative_path(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def validate_candidates(self) -> "OperationPreviewItem":
        directories = [candidate.relative_directory for candidate in self.candidates]
        if len(set(directories)) != len(directories):
            raise ValueError("candidate directories must be unique")

        scores = [candidate.score for candidate in self.candidates]
        if scores != sorted(scores, reverse=True):
            raise ValueError("candidates must be sorted by descending score")
        return self


class OperationPreviewResponse(BaseModel):
    """不包含执行决定、数据库计划或磁盘写入的整理预览。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: int = Field(ge=1)
    items: tuple[OperationPreviewItem, ...]
    read_only: Literal[True] = True

    @field_validator("items")
    @classmethod
    def validate_items(
        cls,
        value: tuple[OperationPreviewItem, ...],
    ) -> tuple[OperationPreviewItem, ...]:
        file_ids = [item.source_file_id for item in value]
        if len(set(file_ids)) != len(file_ids):
            raise ValueError("preview items must have unique source_file_ids")
        return value


def rank_preview_candidates(
    source_file_name: str,
    target_directories: Sequence[str],
) -> tuple[OperationPreviewCandidate, ...]:
    """把 V1 matcher 的 Path 结果转换为只读预览候选。"""

    if (
        not source_file_name
        or source_file_name != source_file_name.strip()
        or "/" in source_file_name
        or "\\" in source_file_name
    ):
        raise ValueError("source_file_name must be a normalized file name")

    validated_directories = tuple(
        _validate_relative_path(directory) for directory in target_directories
    )
    if not validated_directories:
        raise ValueError("target_directories must not be empty")
    if len(set(validated_directories)) != len(validated_directories):
        raise ValueError("target_directories must be unique")

    matches = get_best_matches(
        source_file_name,
        [Path(directory) for directory in validated_directories],
    )
    return tuple(
        OperationPreviewCandidate(
            relative_directory=directory.as_posix(),
            score=score,
        )
        for directory, score in matches
    )
