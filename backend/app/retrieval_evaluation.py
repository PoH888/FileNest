"""可复现的 Keyword、Vector 与可选 Hybrid 检索评测入口。"""

import argparse
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .embedding_client import (
    EmbeddingClient,
    EmbeddingVector,
    EmbeddingClientError,
    validate_embedding_vector,
)
from .keyword_baseline import (
    KeywordBaselineDataset,
    KeywordBaselineDocument,
    load_keyword_baseline,
)
from .keyword_retrieval import search_keyword_results
from .openai_embedding_client import EmbeddingSettings, OpenAIEmbeddingClient
from .retrieval_metrics import (
    RetrievalLatencySummary,
    RetrievalMetricCase,
    RetrievalMetricsSummary,
    aggregate_retrieval_metrics,
    measure_latency_ms,
    summarize_latency_ms,
)


class RetrievalEvaluationError(ValueError):
    """检索评测输入或执行结果不符合统一契约。"""


class RetrievalEvaluationCase(BaseModel):
    """一个同时支持 Keyword 与 Vector 的固定评测案例。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(
        min_length=3,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    question_id: str = Field(min_length=3, max_length=64)
    keyword_query: str = Field(min_length=1, max_length=500)
    vector_query: str = Field(min_length=1, max_length=500)
    relevant_document_paths: tuple[str, ...] = Field(min_length=1)
    top_k: int = Field(gt=0, default=5)

    @field_validator("keyword_query", "vector_query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("retrieval query must not be blank")
        return normalized

    @field_validator("relevant_document_paths")
    @classmethod
    def validate_relevant_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not isinstance(path, str) or not path.strip() for path in value):
            raise ValueError("relevant document paths must be non-empty strings")
        if len(set(value)) != len(value):
            raise ValueError("relevant document paths must be unique")
        return value


class RetrievalEvaluationCaseResult(BaseModel):
    """一个案例在各检索方法下的排序与 latency 观察。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    question_id: str
    keyword_query: str
    vector_query: str
    relevant_document_paths: tuple[str, ...]
    rankings: dict[str, tuple[str, ...]]
    latency_ms: dict[str, float]


class RetrievalEvaluationReport(BaseModel):
    """可保存、可复核的检索方法对比报告。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    dataset: str = Field(min_length=1)
    embedding_source: str = Field(min_length=1)
    embedding_model_version: str = Field(min_length=1)
    quality_evidence: bool
    cases: tuple[RetrievalEvaluationCaseResult, ...] = Field(min_length=1)
    metrics: dict[str, RetrievalMetricsSummary]
    latency: dict[str, RetrievalLatencySummary]


RankingFunction = Callable[[str, int], Sequence[str]]


def load_retrieval_cases(path: Path) -> tuple[RetrievalEvaluationCase, ...]:
    """从 JSON 报告读取固定案例，不读取报告中的历史排序结果。"""

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
        raw_cases = raw_data["cases"]
        return tuple(
            RetrievalEvaluationCase.model_validate(
                {
                    "case_id": case["case_id"],
                    "question_id": case["question_id"],
                    "keyword_query": case["keyword_query"],
                    "vector_query": case["vector_query"],
                    "relevant_document_paths": case[
                        "relevant_document_paths"
                    ],
                    "top_k": raw_data["evaluation_scope"]["ranking_depth"],
                }
            )
            for case in raw_cases
        )
    except (KeyError, TypeError, ValidationError, json.JSONDecodeError, OSError) as error:
        raise RetrievalEvaluationError("检索评测案例数据格式无效") from error


def build_keyword_ranker(dataset: KeywordBaselineDataset) -> RankingFunction:
    """将现有 Keyword 检索实现适配为评测排序函数。"""

    def rank(query: str, top_k: int) -> tuple[str, ...]:
        return tuple(
            result.document.relative_path
            for result in search_keyword_results(dataset, query)[:top_k]
        )

    return rank


class InMemoryVectorIndex:
    """使用真实或注入的 Embedding 客户端构建一次性向量实验索引。"""

    def __init__(
        self,
        documents: Sequence[KeywordBaselineDocument],
        embedding_client: EmbeddingClient,
        *,
        model_version: str | None = None,
    ) -> None:
        if not documents:
            raise RetrievalEvaluationError("vector index documents must not be empty")

        self._documents = tuple(documents)
        self._embedding_client = embedding_client
        try:
            vectors = embedding_client.embed(
                texts=tuple(document.content for document in self._documents)
            )
        except EmbeddingClientError as error:
            raise RetrievalEvaluationError("无法为评测文档生成 Embedding") from error

        if len(vectors) != len(self._documents):
            raise RetrievalEvaluationError("文档 Embedding 数量与文档数量不一致")
        self._vectors = tuple(validate_embedding_vector(vector) for vector in vectors)
        self._model_version = model_version or _read_model_version(embedding_client)

    @property
    def model_version(self) -> str:
        """返回建立索引时使用的模型版本。"""

        return self._model_version

    def rank(self, query: str, top_k: int) -> tuple[str, ...]:
        """向量化查询并按余弦相似度返回固定顺序的文档路径。"""

        if top_k < 1:
            raise RetrievalEvaluationError("top_k must be a positive integer")
        try:
            query_vectors = self._embedding_client.embed(texts=(query,))
            if len(query_vectors) != 1:
                raise RetrievalEvaluationError("查询 Embedding 数量必须为一")
            query_vector = validate_embedding_vector(query_vectors[0])
        except EmbeddingClientError as error:
            raise RetrievalEvaluationError("无法为评测查询生成 Embedding") from error

        ranked = [
            (
                -_cosine_similarity(query_vector, vector),
                document.relative_path,
                document,
            )
            for document, vector in zip(
                self._documents,
                self._vectors,
                strict=True,
            )
        ]
        ranked.sort(key=lambda item: (item[0], item[1]))
        return tuple(document.relative_path for _, _, document in ranked[:top_k])


def run_retrieval_evaluation(
    cases: Sequence[RetrievalEvaluationCase],
    *,
    dataset: str,
    embedding_source: str,
    embedding_model_version: str,
    keyword_ranker: RankingFunction,
    vector_ranker: RankingFunction,
    hybrid_ranker: RankingFunction | None = None,
    quality_evidence: bool = False,
    ks: Sequence[int] = (1, 3),
) -> RetrievalEvaluationReport:
    """在同一案例集上测量 Keyword、Vector 和可选 Hybrid。"""

    normalized_cases = _validate_cases(cases)
    if not isinstance(dataset, str) or not dataset.strip():
        raise RetrievalEvaluationError("dataset must not be blank")
    if not isinstance(embedding_source, str) or not embedding_source.strip():
        raise RetrievalEvaluationError("embedding_source must not be blank")
    if (
        not isinstance(embedding_model_version, str)
        or not embedding_model_version.strip()
    ):
        raise RetrievalEvaluationError(
            "embedding_model_version must not be blank"
        )

    rankers: dict[str, RankingFunction] = {
        "keyword": keyword_ranker,
        "vector": vector_ranker,
    }
    if hybrid_ranker is not None:
        rankers["hybrid"] = hybrid_ranker

    metric_cases: dict[str, list[RetrievalMetricCase]] = {
        method: [] for method in rankers
    }
    latency_samples: dict[str, list[float]] = {method: [] for method in rankers}
    result_cases: list[RetrievalEvaluationCaseResult] = []

    for case in normalized_cases:
        rankings: dict[str, tuple[str, ...]] = {}
        case_latencies: dict[str, float] = {}
        for method, ranker in rankers.items():
            query = case.keyword_query if method == "keyword" else case.vector_query
            try:
                ranking_result, latency_ms = measure_latency_ms(
                    lambda ranker=ranker, query=query, top_k=case.top_k: ranker(
                        query,
                        top_k,
                    )
                )
                ranking = _normalize_ranking(ranking_result)
            except RetrievalEvaluationError:
                raise
            except Exception as error:
                raise RetrievalEvaluationError(
                    f"{method} 检索评测执行失败"
                ) from error

            rankings[method] = ranking
            case_latencies[method] = latency_ms
            latency_samples[method].append(latency_ms)
            metric_cases[method].append(
                RetrievalMetricCase(
                    case_id=case.case_id,
                    relevant_document_paths=case.relevant_document_paths,
                    ranked_document_paths=ranking,
                )
            )

        result_cases.append(
            RetrievalEvaluationCaseResult(
                case_id=case.case_id,
                question_id=case.question_id,
                keyword_query=case.keyword_query,
                vector_query=case.vector_query,
                relevant_document_paths=case.relevant_document_paths,
                rankings=rankings,
                latency_ms=case_latencies,
            )
        )

    return RetrievalEvaluationReport(
        schema_version="1.0",
        dataset=dataset,
        embedding_source=embedding_source,
        embedding_model_version=embedding_model_version,
        quality_evidence=quality_evidence,
        cases=tuple(result_cases),
        metrics={
            method: aggregate_retrieval_metrics(method_cases, ks=ks)
            for method, method_cases in metric_cases.items()
        },
        latency={
            method: summarize_latency_ms(samples)
            for method, samples in latency_samples.items()
        },
    )


def _validate_cases(
    cases: Sequence[RetrievalEvaluationCase],
) -> tuple[RetrievalEvaluationCase, ...]:
    if isinstance(cases, (str, bytes)) or not cases:
        raise RetrievalEvaluationError("cases must not be empty")
    normalized = tuple(cases)
    if any(not isinstance(case, RetrievalEvaluationCase) for case in normalized):
        raise RetrievalEvaluationError(
            "cases must contain RetrievalEvaluationCase instances"
        )
    case_ids = tuple(case.case_id for case in normalized)
    if len(set(case_ids)) != len(case_ids):
        raise RetrievalEvaluationError("case_id values must be unique")
    return normalized


def _normalize_ranking(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RetrievalEvaluationError("retriever must return a path sequence")
    ranking = tuple(value)
    if any(not isinstance(path, str) or not path.strip() for path in ranking):
        raise RetrievalEvaluationError(
            "retriever paths must be non-empty strings"
        )
    if len(set(ranking)) != len(ranking):
        raise RetrievalEvaluationError("retriever paths must be unique")
    return ranking


def _read_model_version(embedding_client: EmbeddingClient) -> str:
    model_version = getattr(embedding_client, "model_version", None)
    if not isinstance(model_version, str) or not model_version.strip():
        raise RetrievalEvaluationError(
            "Embedding 客户端必须暴露非空 model_version"
        )
    return model_version


def _cosine_similarity(
    left: EmbeddingVector,
    right: EmbeddingVector,
) -> float:
    if len(left) != len(right):
        raise RetrievalEvaluationError("查询与文档 Embedding 维度不一致")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise RetrievalEvaluationError("Embedding 向量范数必须大于零")

    similarity = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    ) / (left_norm * right_norm)
    if not math.isfinite(similarity):
        raise RetrievalEvaluationError("Embedding 相似度必须是有限数值")
    return max(-1.0, min(1.0, similarity))


def main() -> None:
    """使用真实 Embedding 执行评测，并将结果打印或保存为 JSON。"""

    parser = argparse.ArgumentParser(description="运行 FileNest 真实 Retrieval Evaluation")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("backend/evaluation/keyword_retrieval_baseline_v1.json"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("backend/evaluation/retrieval_comparison_v1.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = load_keyword_baseline(args.baseline)
    cases = load_retrieval_cases(args.cases)
    embedding_client = OpenAIEmbeddingClient(EmbeddingSettings())
    vector_index = InMemoryVectorIndex(dataset.documents, embedding_client)
    report = run_retrieval_evaluation(
        cases,
        dataset="keyword_retrieval_baseline_v1",
        embedding_source="openai_api",
        embedding_model_version=vector_index.model_version,
        keyword_ranker=build_keyword_ranker(dataset),
        vector_ranker=vector_index.rank,
        quality_evidence=True,
    )
    serialized = report.model_dump_json(indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
