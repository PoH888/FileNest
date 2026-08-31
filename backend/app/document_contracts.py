"""可追踪文档与文本片段的数据契约。"""

import hashlib
import json
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


def validate_source_relative_path(value: str) -> str:
    """公开复用来源相对路径校验，避免各出口各自放宽规则。"""

    return _validate_source_relative_path(value)


class DocumentPage(BaseModel):
    """一页来源文本在规范化文档中的可追踪区间。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page_number: int = Field(ge=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_page_position(self) -> "DocumentPage":
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must not be earlier than start_offset")
        return self


class DocumentPosition(BaseModel):
    """DOCX 段落或表格单元格在规范化文档中的可追踪位置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    element_type: Literal["paragraph", "table_cell"]
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    section_index: int | None = Field(default=None, ge=0)
    heading_level: int | None = Field(default=None, ge=1)
    paragraph_index: int | None = Field(default=None, ge=0)
    table_index: int | None = Field(default=None, ge=0)
    row_index: int | None = Field(default=None, ge=0)
    cell_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_position(self) -> "DocumentPosition":
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must not be earlier than start_offset")

        if self.element_type == "paragraph":
            if self.paragraph_index is None or any(
                value is not None
                for value in (self.table_index, self.row_index, self.cell_index)
            ):
                raise ValueError("paragraph position metadata is inconsistent")
        elif (
            self.paragraph_index is not None
            or self.table_index is None
            or self.row_index is None
            or self.cell_index is None
        ):
            raise ValueError("table cell position metadata is inconsistent")
        return self


class Document(BaseModel):
    """一个来源文件规范化后的不可变文档。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    document_id: UUID
    workspace_id: int = Field(ge=1)
    file_entry_id: int = Field(ge=1)
    source_relative_path: str
    source_format: Literal["markdown", "text", "pdf", "docx"]
    normalized_text: str
    pages: tuple[DocumentPage, ...] = Field(default_factory=tuple)
    source_positions: tuple[DocumentPosition, ...] = Field(default_factory=tuple)
    source_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_updated_at: AwareDatetime

    @field_validator("source_relative_path")
    @classmethod
    def validate_source_relative_path(cls, value: str) -> str:
        return _validate_source_relative_path(value)

    @model_validator(mode="after")
    def validate_page_metadata(self) -> "Document":
        if self.source_format == "pdf":
            if self.source_positions:
                raise ValueError("source_positions are not used for PDF documents")
        elif self.source_format == "docx":
            if self.pages:
                raise ValueError("pages are not used for DOCX documents")
        elif self.pages or self.source_positions:
            raise ValueError(
                "page and source position metadata require PDF or DOCX documents"
            )
        if self.source_format != "pdf":
            return self

        if not self.pages:
            raise ValueError("PDF documents must preserve at least one page")

        previous_page_number = 0
        previous_end_offset = 0
        for page in self.pages:
            if page.page_number != previous_page_number + 1:
                raise ValueError("PDF page numbers must be consecutive from 1")
            if page.start_offset < previous_end_offset:
                raise ValueError("PDF page offsets must not overlap")
            if page.end_offset > len(self.normalized_text):
                raise ValueError("PDF page offsets must fit normalized_text")
            previous_page_number = page.page_number
            previous_end_offset = page.end_offset

        return self

    @model_validator(mode="after")
    def validate_source_positions(self) -> "Document":
        previous_end_offset = 0
        for position in self.source_positions:
            if position.start_offset < previous_end_offset:
                raise ValueError("source position offsets must not overlap")
            if position.end_offset > len(self.normalized_text):
                raise ValueError("source position offsets must fit normalized_text")
            previous_end_offset = position.end_offset
        return self


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
    source_positions: tuple[DocumentPosition, ...] = Field(default_factory=tuple)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

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
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("page_start and page_end must be provided together")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end must not be earlier than page_start")
        for position in self.source_positions:
            if (
                self.start_offset >= position.end_offset
                or position.start_offset >= self.end_offset
            ):
                raise ValueError("source position must overlap chunk range")
        return self


class RetrievedChunk(BaseModel):
    """一次检索中可回到来源文档的片段快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: int = Field(ge=1)
    document_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9][a-z0-9_-]{0,127}$",
    )
    chunk_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9][a-z0-9_-]{0,127}$",
    )
    citation_id: str = Field(
        min_length=6,
        max_length=134,
        pattern=r"^cite_[a-z0-9][a-z0-9_-]{0,127}$",
    )
    file_id: int = Field(ge=1)
    source_relative_path: str
    text: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    source_version: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_updated_at: AwareDatetime | None = None
    indexed_at: AwareDatetime | None = None
    score: int = Field(ge=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    source_positions: tuple[DocumentPosition, ...] = Field(default_factory=tuple)

    @field_validator("source_relative_path")
    @classmethod
    def validate_source_relative_path(cls, value: str) -> str:
        return _validate_source_relative_path(value)

    @model_validator(mode="after")
    def validate_provenance(self) -> "RetrievedChunk":
        if self.end_offset <= self.start_offset:
            raise ValueError("retrieved chunk offsets must be increasing")
        if self.end_line < self.start_line:
            raise ValueError("retrieved chunk lines must be increasing")
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("retrieved PDF page range must be complete")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("retrieved PDF page range must be ordered")
        for position in self.source_positions:
            if (
                self.start_offset >= position.end_offset
                or position.start_offset >= self.end_offset
            ):
                raise ValueError("retrieved DOCX position must overlap chunk")
        return self


class RetrievalContext(BaseModel):
    """Knowledge API、工具和 Agent 共用的检索结果快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    workspace_id: int = Field(ge=1)
    query: str = Field(min_length=1, max_length=200)
    total: int = Field(ge=0)
    top_k: int = Field(ge=1, le=10)
    has_more: bool
    retrieved_at: AwareDatetime
    chunks: tuple[RetrievedChunk, ...]
    snapshot_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_snapshot(self) -> "RetrievalContext":
        if len(self.chunks) > self.top_k:
            raise ValueError("retrieval context exceeds top_k")
        if self.total < len(self.chunks):
            raise ValueError("retrieval total must include selected chunks")

        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        citation_ids = [chunk.citation_id for chunk in self.chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("retrieved chunk ids must be unique")
        if len(set(citation_ids)) != len(citation_ids):
            raise ValueError("retrieved citation ids must be unique")
        if any(
            chunk.workspace_id != self.workspace_id for chunk in self.chunks
        ):
            raise ValueError("retrieved chunk escaped the workspace")

        canonical = json.dumps(
            self.model_dump(mode="json", exclude={"snapshot_hash"}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        computed_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if self.snapshot_hash is not None and self.snapshot_hash != computed_hash:
            raise ValueError("snapshot_hash must match retrieval context")
        object.__setattr__(self, "snapshot_hash", computed_hash)
        return self

    @property
    def has_complete_source_versions(self) -> bool:
        """只有所有片段都有 source version 时才可作当前事实证据。"""

        return bool(self.chunks) and all(
            chunk.source_version is not None for chunk in self.chunks
        )
