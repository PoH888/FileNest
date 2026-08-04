"""第 31 课最小关键词全文检索基线。"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .keyword_baseline import (
    KeywordBaselineDataset,
    KeywordBaselineDocument,
)


class KeywordSearchError(ValueError):
    """关键词检索请求不符合当前基线契约。"""


class KeywordSearchSource(BaseModel):
    """一个关键词命中的可复核文档位置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_relative_path: str = Field(min_length=1)
    matched_text: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_source_position(self) -> "KeywordSearchSource":
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        if self.end_line < self.start_line:
            raise ValueError("end_line must not be earlier than start_line")
        return self


class KeywordSearchResult(BaseModel):
    """一个按命中次数排序、带文档出处的检索结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document: KeywordBaselineDocument
    score: int = Field(ge=1)
    sources: tuple[KeywordSearchSource, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_score(self) -> "KeywordSearchResult":
        if self.score != len(self.sources):
            raise ValueError("score must equal the number of sources")
        return self


def _normalize_keyword(keyword: str) -> str:
    if not isinstance(keyword, str) or not keyword.strip():
        raise KeywordSearchError("keyword must not be blank")
    return keyword.strip().casefold()


def _find_match_ranges(
    content: str,
    normalized_keyword: str,
) -> tuple[tuple[int, int], ...]:
    normalized_content = content.casefold()
    ranges: list[tuple[int, int]] = []
    search_start = 0
    while True:
        match_start = normalized_content.find(
            normalized_keyword,
            search_start,
        )
        if match_start < 0:
            return tuple(ranges)
        match_end = match_start + len(normalized_keyword)
        ranges.append((match_start, match_end))
        search_start = match_end


def _build_source(
    document: KeywordBaselineDocument,
    start_offset: int,
    end_offset: int,
) -> KeywordSearchSource:
    return KeywordSearchSource(
        source_relative_path=document.relative_path,
        matched_text=document.content[start_offset:end_offset],
        start_offset=start_offset,
        end_offset=end_offset,
        start_line=document.content.count("\n", 0, start_offset) + 1,
        end_line=(
            document.content.count(
                "\n",
                0,
                max(start_offset, end_offset - 1),
            )
            + 1
        ),
    )


def search_keyword_documents(
    dataset: KeywordBaselineDataset,
    keyword: str,
) -> tuple[KeywordBaselineDocument, ...]:
    """按文档内容做大小写不敏感的连续字符串匹配。

    当前基线只返回固定文档顺序中的命中文档，不做分词、评分或排序。
    """

    normalized_keyword = _normalize_keyword(keyword)
    return tuple(
        document
        for document in dataset.documents
        if normalized_keyword in document.content.casefold()
    )


def search_keyword_results(
    dataset: KeywordBaselineDataset,
    keyword: str,
) -> tuple[KeywordSearchResult, ...]:
    """按非重叠命中次数降序返回结果，并保留固定顺序作为平分规则。"""

    normalized_keyword = _normalize_keyword(keyword)
    ranked_results: list[tuple[int, int, KeywordSearchResult]] = []

    for document_order, document in enumerate(dataset.documents):
        match_ranges = _find_match_ranges(
            document.content,
            normalized_keyword,
        )
        if not match_ranges:
            continue

        sources = tuple(
            _build_source(document, start_offset, end_offset)
            for start_offset, end_offset in match_ranges
        )
        result = KeywordSearchResult(
            document=document,
            score=len(sources),
            sources=sources,
        )
        ranked_results.append((-result.score, document_order, result))

    ranked_results.sort(key=lambda item: (item[0], item[1]))
    return tuple(result for _, _, result in ranked_results)
