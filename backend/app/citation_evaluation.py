"""Citation 评测数据契约。"""

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .citation_runtime import (
    CITATION_PARSER_VERSION,
    CitationClaim,
    bind_citations,
)
from .document_contracts import RetrievedChunk, RetrievalContext


class CitationDatasetError(ValueError):
    """Citation 评测数据无法读取或不符合固定契约。"""


class CitationSource(BaseModel):
    """评测用的一份出处文本。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)

    @field_validator("title", "content")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("citation source text must not be blank")
        return value


class CitationRef(BaseModel):
    """回答中的一个 citation 及其人工标注的相关性。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    source_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    relevant_to_question: bool


class CitationFact(BaseModel):
    """回答中的一个事实及其出处支持标注。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    claim: str = Field(min_length=1, max_length=1_000)
    citation_ids: tuple[str, ...] = ()
    supported_by_source: bool

    @field_validator("claim")
    @classmethod
    def reject_blank_claim(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("citation fact claim must not be blank")
        return value

    @model_validator(mode="after")
    def validate_unique_citations(self) -> "CitationFact":
        if len(set(self.citation_ids)) != len(self.citation_ids):
            raise ValueError("citation ids within a fact must be unique")
        return self


class CitationCase(BaseModel):
    """一个带明确答案、出处和事实级标注的 Citation 用例。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    question: str = Field(min_length=1, max_length=2_000)
    answer: str = Field(min_length=1, max_length=4_000)
    sources: tuple[CitationSource, ...] = Field(min_length=1)
    citations: tuple[CitationRef, ...] = Field(min_length=1)
    facts: tuple[CitationFact, ...] = Field(min_length=1)

    @field_validator("question", "answer")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("citation case text must not be blank")
        return value

    @model_validator(mode="after")
    def validate_references(self) -> "CitationCase":
        source_ids = [source.source_id for source in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("citation source ids must be unique")

        citation_ids = [citation.citation_id for citation in self.citations]
        if len(set(citation_ids)) != len(citation_ids):
            raise ValueError("citation ids must be unique")

        fact_ids = [fact.fact_id for fact in self.facts]
        if len(set(fact_ids)) != len(fact_ids):
            raise ValueError("citation fact ids must be unique")

        source_id_set = set(source_ids)
        unknown_sources = {
            citation.source_id
            for citation in self.citations
            if citation.source_id not in source_id_set
        }
        if unknown_sources:
            raise ValueError("citation references an unknown source")

        citation_id_set = set(citation_ids)
        for fact in self.facts:
            if fact.claim not in self.answer:
                raise ValueError("citation fact claim must appear in answer")
            if any(
                citation_id not in citation_id_set
                for citation_id in fact.citation_ids
            ):
                raise ValueError("citation fact references an unknown citation")
        return self


class CitationDataset(BaseModel):
    """一个可重复加载的 Citation 评测数据集。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    cases: tuple[CitationCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case_ids(self) -> "CitationDataset":
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("citation case ids must be unique")
        return self


CitationDimension = Literal["source_relevance", "faithfulness"]


class CitationDimensionResult(BaseModel):
    """一个 Citation 维度的事实级计数结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: CitationDimension
    case_id: str
    total: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    failed_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_counts(self) -> "CitationDimensionResult":
        if self.passed + self.failed != self.total:
            raise ValueError("citation dimension counts must add up to total")
        if len(self.failed_ids) != self.failed:
            raise ValueError("failed id count must match failed count")
        return self


def evaluate_source_relevance(case: CitationCase) -> CitationDimensionResult:
    """统计每个 citation 是否被标注为与问题相关。"""

    failed_ids = tuple(
        citation.citation_id
        for citation in case.citations
        if not citation.relevant_to_question
    )
    return CitationDimensionResult(
        dimension="source_relevance",
        case_id=case.case_id,
        total=len(case.citations),
        passed=len(case.citations) - len(failed_ids),
        failed=len(failed_ids),
        failed_ids=failed_ids,
    )


def evaluate_faithfulness(case: CitationCase) -> CitationDimensionResult:
    """统计回答中的每个事实是否被标注为 source 支持。"""

    failed_ids = tuple(
        fact.fact_id for fact in case.facts if not fact.supported_by_source
    )
    return CitationDimensionResult(
        dimension="faithfulness",
        case_id=case.case_id,
        total=len(case.facts),
        passed=len(case.facts) - len(failed_ids),
        failed=len(failed_ids),
        failed_ids=failed_ids,
    )


class CitationFactResult(BaseModel):
    """一个事实的 Citation Correctness 判定明细。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str
    citation_ids: tuple[str, ...]
    has_citation: bool
    has_relevant_citation: bool
    faithful: bool
    correct: bool

    @model_validator(mode="after")
    def validate_fact_result(self) -> "CitationFactResult":
        if self.has_citation != bool(self.citation_ids):
            raise ValueError("has_citation must match citation ids")
        if not self.has_citation and self.has_relevant_citation:
            raise ValueError("an uncited fact cannot have a relevant citation")
        expected_correct = (
            self.has_citation
            and self.has_relevant_citation
            and self.faithful
        )
        if self.correct != expected_correct:
            raise ValueError("correctness must combine citation and support checks")
        return self


class CitationCorrectnessResult(BaseModel):
    """一个用例的事实级 Citation Correctness 结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    total_facts: int = Field(ge=0)
    correct_facts: int = Field(ge=0)
    incorrect_facts: int = Field(ge=0)
    incorrect_fact_ids: tuple[str, ...] = ()
    fact_results: tuple[CitationFactResult, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> "CitationCorrectnessResult":
        if self.correct_facts + self.incorrect_facts != self.total_facts:
            raise ValueError("citation correctness counts must add up to total")
        if len(self.fact_results) != self.total_facts:
            raise ValueError("fact result count must match total facts")
        if len(self.incorrect_fact_ids) != self.incorrect_facts:
            raise ValueError("incorrect fact id count must match incorrect facts")
        return self


CitationObservationStatus = Literal["valid", "invalid_observation"]


class CitationObservation(BaseModel):
    """一次真实 Agent Run 的可审计 Citation 评测观察。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: int = Field(ge=1)
    workspace_id: int = Field(ge=1)
    workspace_fixture: str = Field(min_length=1)
    request_text: str = Field(min_length=1)
    final_answer: str = Field(min_length=1)
    source_snapshot_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    parser_version: str = Field(min_length=1)
    prompt_version: str | None = None
    status: CitationObservationStatus
    invalid_reasons: tuple[str, ...] = ()
    sources: tuple[RetrievedChunk, ...] = ()
    claims: tuple[CitationClaim, ...] = ()
    source_relevance: CitationDimensionResult
    faithfulness: CitationDimensionResult
    correctness: CitationCorrectnessResult

    @model_validator(mode="after")
    def validate_observation_state(self) -> "CitationObservation":
        if self.status == "valid" and self.invalid_reasons:
            raise ValueError("valid observation must not contain invalid reasons")
        if self.status == "valid" and self.source_snapshot_hash is None:
            raise ValueError("valid observation requires a source snapshot hash")
        return self


def evaluate_agent_citation_observation(
    *,
    run_id: int,
    workspace_id: int,
    workspace_fixture: str,
    request_text: str,
    final_answer: str,
    retrieval_context: RetrievalContext
    | Sequence[RetrievalContext]
    | None,
    prompt_version: str | None,
    current_source_versions: Mapping[str, str | None] | None = None,
    navigation_only: bool = False,
) -> CitationObservation:
    """把真实 Agent 回答和检索快照适配到既有 Citation 评测规则核。"""

    contexts = _as_contexts(retrieval_context)
    binding = bind_citations(
        final_answer,
        contexts,
        workspace_id=workspace_id,
        current_source_versions=current_source_versions,
        require_citations=not navigation_only,
        navigation_only=navigation_only,
    )
    reasons = list(binding.invalid_reasons)
    if not prompt_version or not prompt_version.strip():
        reasons.append("prompt_version_missing")
    snapshot_hash = _observation_snapshot_hash(contexts)
    if snapshot_hash is None:
        reasons.append("source_snapshot_missing")

    chunks = _unique_chunks(contexts)
    chunks_by_citation = {
        chunk.citation_id: chunk for chunk in chunks
    }
    if binding.citation_ids and not binding.claims:
        reasons.append("citation_claim_text_missing")

    for citation_id in binding.citation_ids:
        chunk = chunks_by_citation.get(citation_id)
        if chunk is None:
            continue
        suffix = Path(chunk.source_relative_path).suffix.casefold()
        if suffix == ".pdf" and (
            chunk.page_start is None or chunk.page_end is None
        ):
            reasons.append("pdf_citation_without_page")
        if suffix == ".docx" and not chunk.source_positions:
            reasons.append("docx_citation_without_structure")

    citation_refs = tuple(
        CitationRef(
            citation_id=citation_id,
            source_id=citation_id,
            relevant_to_question=(
                citation_id in chunks_by_citation
                and _is_relevant(
                    contexts,
                    chunks_by_citation[citation_id],
                )
            ),
        )
        for citation_id in binding.citation_ids
        if citation_id in chunks_by_citation
    )
    facts = tuple(
        CitationFact(
            fact_id=claim.claim_id,
            claim=claim.text,
            citation_ids=tuple(
                citation_id
                for citation_id in claim.citation_ids
                if citation_id in chunks_by_citation
            ),
            supported_by_source=_claim_is_supported(
                claim,
                chunks_by_citation,
            ),
        )
        for claim in binding.claims
        if any(
            citation_id in chunks_by_citation
            for citation_id in claim.citation_ids
        )
    )

    case_id = f"agent_run_{run_id}"
    if citation_refs and facts:
        case = CitationCase(
            case_id=case_id,
            question=request_text,
            answer=final_answer,
            sources=tuple(
                CitationSource(
                    source_id=citation_id,
                    title=chunks_by_citation[citation_id].source_relative_path,
                    content=chunks_by_citation[citation_id].text,
                )
                for citation_id in binding.citation_ids
                if citation_id in chunks_by_citation
            ),
            citations=citation_refs,
            facts=facts,
        )
        source_relevance = evaluate_source_relevance(case)
        faithfulness = evaluate_faithfulness(case)
        correctness = evaluate_citation_correctness(case)
    else:
        source_relevance = _empty_dimension(
            "source_relevance",
            case_id,
        )
        faithfulness = _empty_dimension("faithfulness", case_id)
        correctness = CitationCorrectnessResult(
            case_id=case_id,
            total_facts=0,
            correct_facts=0,
            incorrect_facts=0,
            fact_results=(),
        )

    unique_reasons = _unique_strings(reasons)
    return CitationObservation(
        run_id=run_id,
        workspace_id=workspace_id,
        workspace_fixture=workspace_fixture,
        request_text=request_text,
        final_answer=final_answer,
        source_snapshot_hash=snapshot_hash,
        parser_version=CITATION_PARSER_VERSION,
        prompt_version=prompt_version,
        status=("invalid_observation" if unique_reasons else "valid"),
        invalid_reasons=unique_reasons,
        sources=chunks,
        claims=binding.claims,
        source_relevance=source_relevance,
        faithfulness=faithfulness,
        correctness=correctness,
    )


def _as_contexts(
    retrieval_context: RetrievalContext
    | Sequence[RetrievalContext]
    | None,
) -> tuple[RetrievalContext, ...]:
    if retrieval_context is None:
        return ()
    if isinstance(retrieval_context, RetrievalContext):
        return (retrieval_context,)
    return tuple(retrieval_context)


def _observation_snapshot_hash(
    contexts: tuple[RetrievalContext, ...],
) -> str | None:
    hashes = [context.snapshot_hash for context in contexts]
    if not hashes or any(value is None for value in hashes):
        return None
    if len(hashes) == 1:
        return hashes[0]
    canonical = "|".join(sorted(value for value in hashes if value))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _unique_chunks(
    contexts: tuple[RetrievalContext, ...],
) -> tuple[RetrievedChunk, ...]:
    chunks: list[RetrievedChunk] = []
    seen: set[str] = set()
    for context in contexts:
        for chunk in context.chunks:
            if chunk.citation_id in seen:
                continue
            seen.add(chunk.citation_id)
            chunks.append(chunk)
    return tuple(chunks)


def _is_relevant(
    contexts: tuple[RetrievalContext, ...],
    chunk: RetrievedChunk,
) -> bool:
    return any(
        context.query.casefold() in chunk.text.casefold()
        for context in contexts
        if context.workspace_id == chunk.workspace_id
    )


def _claim_is_supported(
    claim: CitationClaim,
    chunks_by_citation: Mapping[str, RetrievedChunk],
) -> bool:
    normalized_claim = " ".join(claim.text.casefold().split())
    return any(
        normalized_claim in " ".join(
            chunks_by_citation[citation_id].text.casefold().split()
        )
        for citation_id in claim.citation_ids
        if citation_id in chunks_by_citation
    )


def _empty_dimension(
    dimension: Literal["source_relevance", "faithfulness"],
    case_id: str,
) -> CitationDimensionResult:
    return CitationDimensionResult(
        dimension=dimension,
        case_id=case_id,
        total=0,
        passed=0,
        failed=0,
        failed_ids=(),
    )


def _unique_strings(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def evaluate_citation_correctness(
    case: CitationCase,
) -> CitationCorrectnessResult:
    """检查回答中的每个事实是否有相关且忠实的 citation 支持。"""

    citations_by_id = {
        citation.citation_id: citation for citation in case.citations
    }
    fact_results = tuple(
        CitationFactResult(
            fact_id=fact.fact_id,
            citation_ids=fact.citation_ids,
            has_citation=bool(fact.citation_ids),
            has_relevant_citation=any(
                citations_by_id[citation_id].relevant_to_question
                for citation_id in fact.citation_ids
            ),
            faithful=fact.supported_by_source,
            correct=(
                bool(fact.citation_ids)
                and any(
                    citations_by_id[citation_id].relevant_to_question
                    for citation_id in fact.citation_ids
                )
                and fact.supported_by_source
            ),
        )
        for fact in case.facts
    )
    incorrect_fact_ids = tuple(
        result.fact_id for result in fact_results if not result.correct
    )
    return CitationCorrectnessResult(
        case_id=case.case_id,
        total_facts=len(fact_results),
        correct_facts=len(fact_results) - len(incorrect_fact_ids),
        incorrect_facts=len(incorrect_fact_ids),
        incorrect_fact_ids=incorrect_fact_ids,
        fact_results=fact_results,
    )


def load_citation_evaluation_dataset(path: Path) -> CitationDataset:
    """从 UTF-8 JSON 文件读取并严格校验 Citation 数据集。"""

    try:
        raw_data = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CitationDatasetError("无法读取 Citation 评测数据") from error

    try:
        return CitationDataset.model_validate_json(raw_data)
    except ValidationError as error:
        raise CitationDatasetError("Citation 评测数据格式无效") from error
