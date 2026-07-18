"""FileNest API 对外数据契约。"""

from enum import StrEnum

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class FileSortField(StrEnum):
    """文件列表允许客户端选择的排序字段。"""

    RELATIVE_PATH = "relative_path"
    NAME = "name"
    SIZE_BYTES = "size_bytes"
    MODIFIED_AT = "modified_at"


class SortOrder(StrEnum):
    """文件列表的排序方向。"""

    ASC = "asc"
    DESC = "desc"


class FileQueryParams(BaseModel):
    """文件列表、搜索、过滤、排序和分页参数。"""

    model_config = ConfigDict(extra="forbid")

    keyword: str | None = Field(default=None, max_length=200)
    extension: str | None = None
    modified_from: AwareDatetime | None = None
    modified_to: AwareDatetime | None = None
    sort_by: FileSortField = FileSortField.RELATIVE_PATH
    sort_order: SortOrder = SortOrder.ASC
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=100)

    @field_validator("keyword")
    @classmethod
    def normalize_keyword(cls, value: str | None) -> str | None:
        """去除无意义空白，并拒绝无法执行搜索的空关键词。"""

        if value is None:
            return None

        normalized = value.strip()
        if not normalized:
            raise ValueError("keyword must not be blank")
        return normalized

    @field_validator("extension")
    @classmethod
    def normalize_extension(cls, value: str | None) -> str | None:
        """将扩展名统一为小写且带点的数据库查询形式。"""

        if value is None:
            return None

        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("extension must not be blank")
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        if len(normalized) > 50:
            raise ValueError("extension must not exceed 50 characters")
        return normalized

    @model_validator(mode="after")
    def validate_modified_range(self) -> "FileQueryParams":
        """防止把相反的时间范围交给后续查询层。"""

        if (
            self.modified_from is not None
            and self.modified_to is not None
            and self.modified_from > self.modified_to
        ):
            raise ValueError("modified_from must not be later than modified_to")
        return self


class FileListItemResponse(BaseModel):
    """文件列表中一条可安全公开的索引摘要。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    relative_path: str
    name: str
    extension: str
    size_bytes: int = Field(ge=0)
    modified_at: AwareDatetime


class FileDetailResponse(FileListItemResponse):
    """指定文件索引的详情；不公开工作区绝对路径。"""

    workspace_id: int


class FileListResponse(BaseModel):
    """带分页元数据的文件列表响应。"""

    items: list[FileListItemResponse]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
