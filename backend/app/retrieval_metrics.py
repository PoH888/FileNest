"""Recall@K 与 MRR 的最小检索评测计算。"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievalMetricError(ValueError):
    """检索评测输入不符合当前契约。"""


class RetrievalMetricCase(BaseModel):
    """一个带相关文档集合和排序结果的离线评测案例。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(
        min_length=3,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    relevant_document_paths: tuple[str, ...] = Field(min_length=1)
    ranked_document_paths: tuple[str, ...] = ()

    @field_validator("relevant_document_paths", "ranked_document_paths")
    @classmethod
    def validate_document_paths(
        cls,
        paths: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not isinstance(path, str) or not path.strip() for path in paths):
            raise ValueError("document paths must be non-empty strings")
        if len(set(paths)) != len(paths):
            raise ValueError("document paths must be unique")
        return paths


class RecallAtKMetric(BaseModel):
    """一组评测案例在指定 K 下的平均 Recall。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    k: int = Field(gt=0)
    value: float = Field(ge=0, le=1)


class RetrievalMetricsSummary(BaseModel):
    """一组评测案例的 Recall@K 与 MRR 汇总。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_count: int = Field(gt=0)
    recall_at_k: tuple[RecallAtKMetric, ...] = Field(min_length=1)
    mrr: float = Field(ge=0, le=1)


def recall_at_k(case: RetrievalMetricCase, k: int) -> float:
    """计算单个案例的 Recall@K。"""

    _validate_case(case)
    k = _validate_positive_k(k)
    relevant_paths = set(case.relevant_document_paths)
    retrieved_paths = set(case.ranked_document_paths[:k])
    return len(relevant_paths & retrieved_paths) / len(relevant_paths)


def reciprocal_rank(case: RetrievalMetricCase) -> float:
    """返回第一个相关文档的倒数排名，未命中时返回 0。"""

    _validate_case(case)
    relevant_paths = set(case.relevant_document_paths)
    for rank, path in enumerate(case.ranked_document_paths, start=1):
        if path in relevant_paths:
            return 1.0 / rank
    return 0.0


def aggregate_retrieval_metrics(
    cases: Sequence[RetrievalMetricCase],
    *,
    ks: Sequence[int] = (1, 3),
) -> RetrievalMetricsSummary:
    """对固定案例集合计算平均 Recall@K 与 MRR。"""

    normalized_cases = _validate_cases(cases)
    normalized_ks = _validate_ks(ks)
    recall_metrics = tuple(
        RecallAtKMetric(
            k=k,
            value=sum(recall_at_k(case, k) for case in normalized_cases)
            / len(normalized_cases),
        )
        for k in normalized_ks
    )
    mean_reciprocal_rank = sum(
        reciprocal_rank(case) for case in normalized_cases
    ) / len(normalized_cases)
    return RetrievalMetricsSummary(
        case_count=len(normalized_cases),
        recall_at_k=recall_metrics,
        mrr=mean_reciprocal_rank,
    )


def _validate_case(case: RetrievalMetricCase) -> None:
    if not isinstance(case, RetrievalMetricCase):
        raise RetrievalMetricError(
            "cases must contain RetrievalMetricCase instances"
        )


def _validate_cases(
    cases: Sequence[RetrievalMetricCase],
) -> tuple[RetrievalMetricCase, ...]:
    if isinstance(cases, (str, bytes)) or not cases:
        raise RetrievalMetricError("cases must not be empty")
    normalized_cases = tuple(cases)
    for case in normalized_cases:
        _validate_case(case)
    case_ids = tuple(case.case_id for case in normalized_cases)
    if len(set(case_ids)) != len(case_ids):
        raise RetrievalMetricError("case_id values must be unique")
    return normalized_cases


def _validate_ks(ks: Sequence[int]) -> tuple[int, ...]:
    if isinstance(ks, (str, bytes)) or not ks:
        raise RetrievalMetricError("ks must not be empty")
    normalized_ks = tuple(_validate_positive_k(k) for k in ks)
    if len(set(normalized_ks)) != len(normalized_ks):
        raise RetrievalMetricError("ks values must be unique")
    return normalized_ks


def _validate_positive_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise RetrievalMetricError("k must be a positive integer")
    return k
