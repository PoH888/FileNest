"""可追踪文档与文本片段的数据契约。"""

from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def _validate_source_relative_path(value: str) -> str:
    """保持来源路径稳定可比较，实际文件授权仍由 PathPolicy 负责。"""

    if value != value.strip() or "\\" in value:
        raise ValueError("source_relative_path must be normalized without backslashes")

    path = PurePosixPath(value)
    if (
        value == "."
        or path.is_absolute()
        or PureWindowsPath(value).drive
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != value
    ):
        raise ValueError("source_relative_path must be a normalized relative path")
    return value


class Document(BaseModel):
    """一个 Markdown/TXT 文件规范化后的不可变文档。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    document_id: UUID
    workspace_id: int = Field(ge=1)
    file_entry_id: int = Field(ge=1)
    source_relative_path: str
    source_format: Literal["markdown", "text"]
    normalized_text: str
    source_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_updated_at: AwareDatetime

    @field_validator("source_relative_path")
    @classmethod
    def validate_source_relative_path(cls, value: str) -> str:
        return _validate_source_relative_path(value)


class Chunk(BaseModel):
    """一段文本及其在来源文件中的可追踪位置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    chunk_id: UUID
    document_id: UUID
    file_entry_id: int = Field(ge=1)
    source_relative_path: str
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @field_validator("source_relative_path")
    @classmethod
    def validate_source_relative_path(cls, value: str) -> str:
        return _validate_source_relative_path(value)

    @model_validator(mode="after")
    def validate_source_position(self) -> "Chunk":
        # 半开区间便于后续直接使用 document[start_offset:end_offset] 复核出处。
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        if self.end_line < self.start_line:
            raise ValueError("end_line must not be earlier than start_line")
        return self
