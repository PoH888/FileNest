"""第 31 课固定关键词检索基线的数据契约。"""

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class KeywordBaselineError(ValueError):
    """关键词检索基线数据无法读取或不符合固定契约。"""


def _validate_relative_path(value: str) -> str:
    """只接受可重复比较的 POSIX 相对路径。"""

    if value != value.strip() or "\\" in value:
        raise ValueError(
            "relative_path must be normalized without backslashes"
        )

    path = PurePosixPath(value)
    if (
        value == "."
        or path.is_absolute()
        or PureWindowsPath(value).drive
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != value
    ):
        raise ValueError("relative_path must stay inside the baseline corpus")
    return value


class KeywordBaselineDocument(BaseModel):
    """固定基线语料中的一个 Markdown/TXT 文档。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        value = _validate_relative_path(value)
        if PurePosixPath(value).suffix.casefold() not in {
            ".md",
            ".markdown",
            ".txt",
        }:
            raise ValueError(
                "baseline documents must use Markdown or TXT extensions"
            )
        return value


class KeywordBaselineQuestion(BaseModel):
    """一个固定问题及其人工确定的相关文档集合。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str = Field(
        min_length=3,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    question: str = Field(min_length=1, max_length=500)
    related_document_paths: tuple[str, ...] = ()

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("baseline question must not be blank")
        return normalized

    @field_validator("related_document_paths")
    @classmethod
    def validate_related_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_validate_relative_path(path) for path in value)
        if len(set(normalized)) != len(normalized):
            raise ValueError("related document paths must be unique")
        return normalized


class KeywordBaselineDataset(BaseModel):
    """带版本、固定文档集合和固定问题集合的基线数据集。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    name: str = Field(min_length=1, max_length=100)
    documents: tuple[KeywordBaselineDocument, ...] = Field(min_length=1)
    questions: tuple[KeywordBaselineQuestion, ...] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("baseline dataset name must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_fixed_references(self) -> "KeywordBaselineDataset":
        document_paths = [document.relative_path for document in self.documents]
        if len(set(document_paths)) != len(document_paths):
            raise ValueError("baseline document paths must be unique")

        question_ids = [question.question_id for question in self.questions]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("baseline question ids must be unique")

        known_paths = set(document_paths)
        for question in self.questions:
            unknown_paths = set(question.related_document_paths) - known_paths
            if unknown_paths:
                raise ValueError(
                    "question related document path does not exist: "
                    + ", ".join(sorted(unknown_paths))
                )
        return self


def load_keyword_baseline(path: Path) -> KeywordBaselineDataset:
    """从 UTF-8 JSON 文件读取并严格校验固定基线数据。"""

    try:
        raw_data = path.read_text(encoding="utf-8")
    except OSError as error:
        raise KeywordBaselineError("无法读取关键词检索基线数据") from error

    try:
        return KeywordBaselineDataset.model_validate_json(raw_data)
    except ValidationError as error:
        raise KeywordBaselineError("关键词检索基线数据格式无效") from error
