"""Agent 回答的稳定引用解析与 RetrievalContext 绑定边界。"""

from collections.abc import Iterable, Mapping, Sequence
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .document_contracts import RetrievedChunk, RetrievalContext


CITATION_PARSER_VERSION = "citation-parser-v1"
_VALID_CITATION = re.compile(
    r"^cite_[a-z0-9][a-z0-9_-]{0,127}$",
)
_CITATION_TOKEN = re.compile(
    r"\[\[(?P<citation_id>cite_[a-z0-9][a-z0-9_-]{0,127})\]\]",
)
_ANY_DOUBLE_BRACKET = re.compile(r"\[\[(?P<value>[^\]]+)\]\]")


class CitationClaim(BaseModel):
    """回答中由一个稳定引用标识绑定的最小声明片段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    citation_ids: tuple[str, ...] = Field(min_length=1)


CitationBindingStatus = Literal["bound", "unbound", "invalid"]


class CitationBinding(BaseModel):
    """一次回答与检索快照之间的可审计绑定结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: int = Field(ge=1)
    parser_version: str = Field(min_length=1)
    status: CitationBindingStatus
    navigation_only: bool
    citation_ids: tuple[str, ...]
    claims: tuple[CitationClaim, ...]
    invalid_citation_ids: tuple[str, ...] = ()
    invalid_reasons: tuple[str, ...] = ()
    snapshot_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


def extract_citation_ids(answer: str) -> tuple[str, ...]:
    """提取格式正确且保持首次出现顺序的引用标识。"""

    return _unique(
        match.group("citation_id")
        for match in _CITATION_TOKEN.finditer(answer)
    )


def parse_citation_claims(answer: str) -> tuple[CitationClaim, ...]:
    """将引用前的回答文本保留为声明，避免模型自行传入 source 内容。"""

    claims: list[CitationClaim] = []
    cursor = 0
    for index, match in enumerate(_CITATION_TOKEN.finditer(answer), start=1):
        text = answer[cursor : match.start()].strip(" \t\r\n，。；;:")
        if not text:
            text = answer[: match.start()].strip(" \t\r\n，。；;:")
        if text:
            claims.append(
                CitationClaim(
                    claim_id=f"claim_{index}",
                    text=text,
                    citation_ids=(match.group("citation_id"),),
                )
            )
        cursor = match.end()
    return tuple(claims)


def bind_citations(
    answer: str,
    retrieval_context: RetrievalContext
    | Sequence[RetrievalContext]
    | None,
    *,
    workspace_id: int | None = None,
    current_source_versions: Mapping[str, str | None] | None = None,
    require_citations: bool = False,
    navigation_only: bool = False,
) -> CitationBinding:
    """把回答中的引用限制在当前检索快照，并对来源新鲜度做 fail-closed 校验。"""

    contexts = _as_contexts(retrieval_context)
    expected_workspace_id = (
        workspace_id
        if workspace_id is not None
        else contexts[0].workspace_id
        if contexts
        else 1
    )
    citation_ids = extract_citation_ids(answer)
    malformed_ids = tuple(
        value
        for value in _ANY_DOUBLE_BRACKET.findall(answer)
        if not _VALID_CITATION.fullmatch(value)
    )
    known_chunks: dict[str, RetrievedChunk] = {}
    reasons: list[str] = []

    for context in contexts:
        if context.workspace_id != expected_workspace_id:
            reasons.append("cross_workspace_context")
        for chunk in context.chunks:
            if chunk.workspace_id != expected_workspace_id:
                reasons.append("cross_workspace_source")
            if chunk.citation_id in known_chunks:
                reasons.append("duplicate_citation_id")
            known_chunks[chunk.citation_id] = chunk

    invalid_ids = list(malformed_ids)
    for citation_id in citation_ids:
        bound_chunk = known_chunks.get(citation_id)
        if bound_chunk is None:
            invalid_ids.append(citation_id)
            reasons.append("unknown_citation_id")
            continue
        if not _has_source_position(bound_chunk):
            invalid_ids.append(citation_id)
            reasons.append("citation_without_source_position")
        if bound_chunk.source_version is None:
            invalid_ids.append(citation_id)
            reasons.append("missing_source_version")
        if current_source_versions is not None:
            current_version = _current_version(
                bound_chunk,
                current_source_versions,
            )
            if current_version is not None and (
                bound_chunk.source_version != current_version
            ):
                invalid_ids.append(citation_id)
                reasons.append("source_version_mismatch")
            elif _has_version_key(bound_chunk, current_source_versions) and (
                current_version is None
            ):
                invalid_ids.append(citation_id)
                reasons.append("source_deleted")

    if malformed_ids:
        reasons.append("malformed_citation_token")
    if not contexts and citation_ids:
        reasons.append("missing_retrieval_context")
    if not citation_ids and require_citations and not navigation_only:
        reasons.append("citations_required")

    unique_invalid_ids = _unique(invalid_ids)
    unique_reasons = _unique(reasons)
    if unique_reasons:
        status: CitationBindingStatus = "invalid"
    elif citation_ids:
        status = "bound"
    else:
        status = "unbound"

    return CitationBinding(
        workspace_id=expected_workspace_id,
        parser_version=CITATION_PARSER_VERSION,
        status=status,
        navigation_only=navigation_only,
        citation_ids=citation_ids,
        claims=parse_citation_claims(answer),
        invalid_citation_ids=unique_invalid_ids,
        invalid_reasons=unique_reasons,
        snapshot_hash=(
            contexts[0].snapshot_hash
            if len(contexts) == 1
            else None
        ),
    )


def _as_contexts(
    retrieval_context: RetrievalContext | Sequence[RetrievalContext] | None,
) -> tuple[RetrievalContext, ...]:
    if retrieval_context is None:
        return ()
    if isinstance(retrieval_context, RetrievalContext):
        return (retrieval_context,)
    return tuple(retrieval_context)


def _has_source_position(chunk: RetrievedChunk) -> bool:
    return (
        chunk.end_offset > chunk.start_offset
        and chunk.end_line >= chunk.start_line
    ) or chunk.page_start is not None or bool(chunk.source_positions)


def _has_version_key(
    chunk: RetrievedChunk,
    current_source_versions: Mapping[str, str | None],
) -> bool:
    return any(
        key in current_source_versions
        for key in (chunk.document_id, chunk.chunk_id, chunk.citation_id)
    )


def _current_version(
    chunk: RetrievedChunk,
    current_source_versions: Mapping[str, str | None],
) -> str | None:
    for key in (chunk.document_id, chunk.chunk_id, chunk.citation_id):
        if key in current_source_versions:
            return current_source_versions[key]
    return None


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)
