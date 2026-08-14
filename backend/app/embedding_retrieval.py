"""基于 SQLite 片段向量的向量检索与最小混合检索。"""

import math
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .embedding_client import (
    EmbeddingClient,
    EmbeddingVector,
    validate_embedding_texts,
    validate_embedding_vector,
)
from .models import ChunkEmbeddingRecord, ChunkRecord, FileEntry


DEFAULT_VECTOR_WEIGHT = 0.7
DEFAULT_KEYWORD_WEIGHT = 0.3


class EmbeddingRetrievalError(ValueError):
    """向量检索请求或候选向量不符合当前契约。"""


class VectorDimensionMismatchError(EmbeddingRetrievalError):
    """查询向量与持久化片段向量维度不一致。"""


class ChunkRetrievalResult(BaseModel):
    """一个带原始片段出处和检索分数的结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_relative_path: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    score: float
    vector_score: float
    keyword_score: float = Field(ge=0, le=1)


@dataclass(frozen=True)
class _Candidate:
    chunk: ChunkRecord
    vector_score: float
    keyword_count: int = 0


def search_vector_chunks(
    session: Session,
    *,
    workspace_id: int,
    query: str,
    embedding_client: EmbeddingClient,
    embedding_model: str,
    top_k: int = 5,
) -> tuple[ChunkRetrievalResult, ...]:
    """在指定工作区内按余弦相似度返回最相近的片段。"""

    _validate_workspace_id(workspace_id)
    _validate_embedding_model(embedding_model)
    top_k = _validate_top_k(top_k)

    query_vector = _embed_query(embedding_client, query)
    candidates = _load_candidates(
        session,
        workspace_id=workspace_id,
        embedding_model=embedding_model,
        query_vector=query_vector,
    )
    candidates.sort(key=_vector_sort_key)

    return tuple(
        _build_result(
            candidate,
            score=candidate.vector_score,
            keyword_score=0.0,
        )
        for candidate in candidates[:top_k]
    )


def search_hybrid_chunks(
    session: Session,
    *,
    workspace_id: int,
    query: str,
    embedding_client: EmbeddingClient,
    embedding_model: str,
    top_k: int = 5,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
) -> tuple[ChunkRetrievalResult, ...]:
    """结合向量相似度与现有字面关键词规则返回片段。"""

    _validate_workspace_id(workspace_id)
    _validate_embedding_model(embedding_model)
    top_k = _validate_top_k(top_k)
    vector_weight, keyword_weight = _validate_weights(
        vector_weight,
        keyword_weight,
    )

    query_vector = _embed_query(embedding_client, query)
    normalized_keyword = _normalize_keyword(query)
    candidates = _load_candidates(
        session,
        workspace_id=workspace_id,
        embedding_model=embedding_model,
        query_vector=query_vector,
    )
    candidates = [
        _Candidate(
            chunk=candidate.chunk,
            vector_score=candidate.vector_score,
            keyword_count=_count_keyword_occurrences(
                candidate.chunk.text,
                normalized_keyword,
            ),
        )
        for candidate in candidates
    ]

    max_keyword_count = max(
        (candidate.keyword_count for candidate in candidates),
        default=0,
    )
    ranked: list[tuple[float, float, _Candidate]] = []
    total_weight = vector_weight + keyword_weight
    for candidate in candidates:
        keyword_score = (
            candidate.keyword_count / max_keyword_count
            if max_keyword_count
            else 0.0
        )
        vector_score = (candidate.vector_score + 1.0) / 2.0
        hybrid_score = (
            vector_weight * vector_score + keyword_weight * keyword_score
        ) / total_weight
        ranked.append((hybrid_score, keyword_score, candidate))

    ranked.sort(key=_hybrid_sort_key)
    return tuple(
        _build_result(
            candidate,
            score=hybrid_score,
            keyword_score=keyword_score,
        )
        for hybrid_score, keyword_score, candidate in ranked[:top_k]
    )


def _load_candidates(
    session: Session,
    *,
    workspace_id: int,
    embedding_model: str,
    query_vector: EmbeddingVector,
) -> list[_Candidate]:
    statement = (
        select(ChunkRecord, ChunkEmbeddingRecord)
        .join(
            ChunkEmbeddingRecord,
            ChunkEmbeddingRecord.chunk_id == ChunkRecord.chunk_id,
        )
        .join(FileEntry, FileEntry.id == ChunkRecord.file_entry_id)
        .where(
            FileEntry.workspace_id == workspace_id,
            ChunkEmbeddingRecord.embedding_model == embedding_model,
        )
        .order_by(
            ChunkRecord.source_relative_path.asc(),
            ChunkRecord.chunk_index.asc(),
            ChunkRecord.chunk_id.asc(),
        )
    )

    candidates: list[_Candidate] = []
    for chunk, embedding in session.execute(statement):
        candidates.append(
            _Candidate(
                chunk=chunk,
                vector_score=_cosine_similarity(
                    query_vector,
                    embedding.vector,
                ),
            )
        )
    return candidates


def _embed_query(
    embedding_client: EmbeddingClient,
    query: str,
) -> EmbeddingVector:
    query_texts = validate_embedding_texts((query,))
    try:
        vectors = embedding_client.embed(texts=query_texts)
        if len(vectors) != 1:
            raise EmbeddingRetrievalError(
                "Embedding 客户端必须为一个查询返回一个向量"
            )
        return validate_embedding_vector(vectors[0])
    except (IndexError, TypeError) as error:
        raise EmbeddingRetrievalError(
            "Embedding 客户端返回的查询向量格式无效"
        ) from error


def _cosine_similarity(
    left: EmbeddingVector,
    right: EmbeddingVector,
) -> float:
    if len(left) != len(right):
        raise VectorDimensionMismatchError(
            "查询向量与片段向量维度不一致"
        )

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise EmbeddingRetrievalError("向量范数必须大于零")

    similarity = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    ) / (left_norm * right_norm)
    if not math.isfinite(similarity):
        raise EmbeddingRetrievalError("向量相似度必须是有限数值")
    return max(-1.0, min(1.0, similarity))


def _build_result(
    candidate: _Candidate,
    *,
    score: float,
    keyword_score: float,
) -> ChunkRetrievalResult:
    chunk = candidate.chunk
    return ChunkRetrievalResult(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source_relative_path=chunk.source_relative_path,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        score=score,
        vector_score=candidate.vector_score,
        keyword_score=keyword_score,
    )


def _normalize_keyword(query: str) -> str:
    return query.strip().casefold()


def _count_keyword_occurrences(text: str, keyword: str) -> int:
    if not keyword:
        return 0
    return text.casefold().count(keyword)


def _vector_sort_key(candidate: _Candidate) -> tuple[float, str, int, str]:
    chunk = candidate.chunk
    return (
        -candidate.vector_score,
        chunk.source_relative_path,
        chunk.chunk_index,
        chunk.chunk_id,
    )


def _hybrid_sort_key(
    item: tuple[float, float, _Candidate],
) -> tuple[float, float, float, str, int, str]:
    hybrid_score, keyword_score, candidate = item
    chunk = candidate.chunk
    return (
        -hybrid_score,
        -candidate.vector_score,
        -keyword_score,
        chunk.source_relative_path,
        chunk.chunk_index,
        chunk.chunk_id,
    )


def _validate_workspace_id(workspace_id: int) -> None:
    if (
        not isinstance(workspace_id, int)
        or isinstance(workspace_id, bool)
        or workspace_id < 1
    ):
        raise EmbeddingRetrievalError(
            "workspace_id must be a positive integer"
        )


def _validate_embedding_model(embedding_model: str) -> None:
    if (
        not isinstance(embedding_model, str)
        or not embedding_model
        or embedding_model != embedding_model.strip()
    ):
        raise EmbeddingRetrievalError(
            "embedding_model must be non-empty without surrounding whitespace"
        )


def _validate_top_k(top_k: int) -> int:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise EmbeddingRetrievalError("top_k must be a positive integer")
    return top_k


def _validate_weights(
    vector_weight: float,
    keyword_weight: float,
) -> tuple[float, float]:
    weights = (vector_weight, keyword_weight)
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight < 0
        for weight in weights
    ):
        raise EmbeddingRetrievalError(
            "检索权重必须是非负有限数值"
        )
    if vector_weight + keyword_weight <= 0:
        raise EmbeddingRetrievalError("至少一个检索权重必须大于零")
    return float(vector_weight), float(keyword_weight)
