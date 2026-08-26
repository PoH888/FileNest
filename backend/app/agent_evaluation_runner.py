"""只读 Agent 固定数据集的可复现运行与评分。"""

from decimal import Decimal
from pathlib import Path
from time import perf_counter
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from .agent_evaluation import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationToolExpectation,
    EvaluationVersionInfo,
    ForbiddenToolsEvaluation,
    evaluate_forbidden_tools,
    materialize_evaluation_workspace,
)
from .agent_loop import AgentLoop
from .agent_observability import SqlAlchemyAgentRunRecorder
from .database import Base
from .fake_model_client import FakeModelClient
from .model_client import (
    ModelClient,
    ModelMessage,
    ModelResponse,
    ModelToolCall,
)
from .services import create_workspace, scan_workspace
from .tool_contracts import ToolResult
from .tool_registry import ToolDefinition, build_read_tool_registry


class EvaluationRunDirectoryError(ValueError):
    """评测运行目录已存在，继续执行可能覆盖旧证据。"""


class EvaluationCaseResult(BaseModel):
    """一个评测用例的安全评分结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    category: str
    expected_run_status: str
    actual_run_status: str
    task_success: bool
    expected_tool_names: tuple[str, ...]
    actual_tool_names: tuple[str, ...]
    tool_results_match: bool
    correct_tool_selections: int = Field(ge=0)
    tool_selection_total: int = Field(ge=0)
    valid_parameter_calls: int = Field(ge=0)
    parameter_call_total: int = Field(ge=0)
    model_turns: int = Field(ge=0)
    run_latency_ms: float = Field(ge=0)
    estimated_model_cost_usd: Decimal = Field(ge=0)


class EvaluationMetrics(BaseModel):
    """固定数据集的三项聚合指标及其原始计数。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    successful_tasks: int = Field(ge=0)
    task_total: int = Field(ge=0)
    task_success_rate: float = Field(ge=0, le=1)
    correct_tool_selections: int = Field(ge=0)
    tool_selection_total: int = Field(ge=0)
    tool_selection_rate: float = Field(ge=0, le=1)
    valid_parameter_calls: int = Field(ge=0)
    parameter_call_total: int = Field(ge=0)
    parameter_validity_rate: float = Field(ge=0, le=1)


class RiskConstraintEvaluation(BaseModel):
    """固定数据集风险边界用例的汇总结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    checked_case_ids: tuple[str, ...]
    failed_case_ids: tuple[str, ...]


class EvaluationSummary(BaseModel):
    """一次完整评测返回的逐用例证据和聚合指标。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    dataset_schema_version: str
    model_source: str
    version_info: EvaluationVersionInfo
    forbidden_tools: ForbiddenToolsEvaluation
    risk_constraints: RiskConstraintEvaluation
    cases: tuple[EvaluationCaseResult, ...]
    metrics: EvaluationMetrics
    total_model_turns: int = Field(ge=0)
    total_run_latency_ms: float = Field(ge=0)
    total_estimated_model_cost_usd: Decimal = Field(ge=0)


class EvaluationHistoryError(ValueError):
    """评测历史无法读取、追加或比较。"""


class EvaluationComparison(BaseModel):
    """两个 Git commit 的评测能力差异。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    commit_a: str
    commit_b: str
    task_success_rate_delta: float
    tool_selection_rate_delta: float
    parameter_validity_rate_delta: float
    risk_constraints_a_passed: bool
    risk_constraints_b_passed: bool
    forbidden_tools_a_passed: bool
    forbidden_tools_b_passed: bool
    regressions: tuple[str, ...]


def run_evaluation_dataset(
    dataset: EvaluationDataset,
    run_root: Path,
    *,
    model_client_factory: Callable[[], ModelClient] | None = None,
    model_source: str = "scripted_fake",
    version_info: EvaluationVersionInfo,
) -> EvaluationSummary:
    """在全新隔离目录中运行数据集并返回评分。"""

    if run_root.exists() or run_root.is_symlink():
        raise EvaluationRunDirectoryError("评测运行目录必须尚不存在")

    run_root.mkdir(parents=True)
    workspace_root = materialize_evaluation_workspace(
        dataset.workspace,
        run_root / "workspace",
    )
    database_path = run_root / "evaluation.db"
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")

    try:
        Base.metadata.create_all(bind=engine)
        with Session(engine, expire_on_commit=False) as session:
            workspace = create_workspace(
                session,
                dataset.workspace.name,
                str(workspace_root),
            )
            scan_workspace(session, workspace.id)
            forbidden_tools = evaluate_forbidden_tools(
                build_read_tool_registry(session).names
            )
            case_results = tuple(
                _run_case(
                    case,
                    session,
                    model_client_factory=model_client_factory,
                )
                for case in dataset.cases
            )
            risk_constraints = _evaluate_risk_constraints(case_results)
    finally:
        engine.dispose()

    summary = EvaluationSummary(
        schema_version="1.0",
        dataset_schema_version=dataset.schema_version,
        model_source=model_source,
        version_info=version_info,
        forbidden_tools=forbidden_tools,
        risk_constraints=risk_constraints,
        cases=case_results,
        metrics=_aggregate_metrics(case_results),
        total_model_turns=sum(case.model_turns for case in case_results),
        total_run_latency_ms=sum(
            case.run_latency_ms for case in case_results
        ),
        total_estimated_model_cost_usd=sum(
            (
                case.estimated_model_cost_usd
                for case in case_results
            ),
            start=Decimal("0"),
        ),
    )
    save_evaluation_summary(summary, run_root / "evaluation-result.json")
    return summary


def _run_case(
    case: EvaluationCase,
    session: Session,
    *,
    model_client_factory: Callable[[], ModelClient] | None,
) -> EvaluationCaseResult:
    model_client = (
        model_client_factory()
        if model_client_factory is not None
        else FakeModelClient(_model_responses(case))
    )
    measured_model_client = _MeasuredModelClient(model_client)
    loop = AgentLoop(
        model_client=measured_model_client,
        tool_registry=build_read_tool_registry(session),
        recorder=SqlAlchemyAgentRunRecorder(session),
    )
    started_at = perf_counter()
    run_result = loop.run(
        [ModelMessage(role="user", content=case.prompt)],
        max_steps=case.max_steps,
        retry_base_delay_seconds=0,
    )
    run_latency_ms = (perf_counter() - started_at) * 1_000

    actual_tool_names = tuple(
        tool_call.name
        for message in run_result.messages
        if message.role == "assistant"
        for tool_call in message.tool_calls
    )
    actual_tool_results = _tool_results(run_result.messages)
    tool_results_match = _tool_results_match(
        case.expected_tool_results,
        actual_tool_results,
    )
    correct_selections, selection_total = _selection_score(
        case.expected_tool_names,
        actual_tool_names,
    )
    valid_parameters, parameter_total = _parameter_score(
        actual_tool_results
    )
    task_success = (
        run_result.status == case.expected_run_status
        and actual_tool_names == case.expected_tool_names
        and tool_results_match
    )

    return EvaluationCaseResult(
        case_id=case.case_id,
        category=case.category,
        expected_run_status=case.expected_run_status,
        actual_run_status=run_result.status,
        task_success=task_success,
        expected_tool_names=case.expected_tool_names,
        actual_tool_names=actual_tool_names,
        tool_results_match=tool_results_match,
        correct_tool_selections=correct_selections,
        tool_selection_total=selection_total,
        valid_parameter_calls=valid_parameters,
        parameter_call_total=parameter_total,
        model_turns=run_result.model_turns,
        run_latency_ms=run_latency_ms,
        estimated_model_cost_usd=measured_model_client.cost_usd,
    )


def _evaluate_risk_constraints(
    case_results: tuple[EvaluationCaseResult, ...],
) -> RiskConstraintEvaluation:
    """检查非法参数、越权请求和步骤上限用例是否保持安全结果。"""

    risk_categories = {"invalid_arguments", "unauthorized", "max_steps"}
    checked_case_ids = tuple(
        case.case_id
        for case in case_results
        if case.category in risk_categories
    )
    failed_case_ids = tuple(
        case.case_id
        for case in case_results
        if case.category in risk_categories and not case.task_success
    )
    return RiskConstraintEvaluation(
        passed=not failed_case_ids,
        checked_case_ids=checked_case_ids,
        failed_case_ids=failed_case_ids,
    )


class _MeasuredModelClient:
    """保留真实模型客户端的费用指标而不改变 Agent Loop 契约。"""

    def __init__(self, delegate: ModelClient) -> None:
        self._delegate = delegate
        self.cost_usd = Decimal("0")

    def complete(
        self,
        *,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        response = self._delegate.complete(messages=messages, tools=tools)
        if (
            response.metrics is not None
            and response.metrics.estimated_cost_usd is not None
        ):
            self.cost_usd += response.metrics.estimated_cost_usd
        return response


def save_evaluation_summary(
    summary: EvaluationSummary,
    result_path: Path,
) -> None:
    """以独占创建方式保存安全结果，避免覆盖历史评测证据。"""

    with result_path.open("x", encoding="utf-8", newline="\n") as result_file:
        result_file.write(summary.model_dump_json(indent=2))
        result_file.write("\n")


def append_evaluation_history(
    summary: EvaluationSummary,
    history_path: Path,
) -> None:
    """以一行一个摘要的追加方式保存跨运行评测历史。"""

    if history_path.is_symlink():
        raise EvaluationHistoryError("拒绝向符号链接追加评测历史")
    if not history_path.parent.is_dir():
        raise EvaluationHistoryError("评测历史文件的父目录不存在")

    try:
        with history_path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as history_file:
            history_file.write(summary.model_dump_json())
            history_file.write("\n")
    except OSError as error:
        raise EvaluationHistoryError("无法追加评测历史") from error


def load_evaluation_history(
    history_path: Path,
) -> tuple[EvaluationSummary, ...]:
    """严格读取每行一个评测摘要的历史文件。"""

    try:
        history_text = history_path.read_text(encoding="utf-8")
    except OSError as error:
        raise EvaluationHistoryError("无法读取评测历史") from error

    records: list[EvaluationSummary] = []
    for line_number, line in enumerate(history_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(EvaluationSummary.model_validate_json(line))
        except ValidationError as error:
            raise EvaluationHistoryError(
                f"评测历史第 {line_number} 行格式无效"
            ) from error
    return tuple(records)


def compare_evaluation_commits(
    history_path: Path,
    commit_a: str,
    commit_b: str,
) -> EvaluationComparison:
    """比较同一评测配置下两个 commit 的最新历史记录。"""

    normalized_a = commit_a.casefold()
    normalized_b = commit_b.casefold()
    if normalized_a == normalized_b:
        raise EvaluationHistoryError("比较需要两个不同的 Git commit")

    records = load_evaluation_history(history_path)
    record_a = _latest_commit_record(records, normalized_a)
    record_b = _latest_commit_record(records, normalized_b)
    for field_name in (
        "prompt_version",
        "model_version",
        "evaluation_dataset_version",
    ):
        value_a = getattr(record_a.version_info, field_name)
        value_b = getattr(record_b.version_info, field_name)
        if value_a != value_b:
            raise EvaluationHistoryError(
                "两个 commit 的评测版本信息不一致，不能直接比较"
            )

    regressions: list[str] = []
    task_success_rate_delta = (
        record_b.metrics.task_success_rate
        - record_a.metrics.task_success_rate
    )
    tool_selection_rate_delta = (
        record_b.metrics.tool_selection_rate
        - record_a.metrics.tool_selection_rate
    )
    parameter_validity_rate_delta = (
        record_b.metrics.parameter_validity_rate
        - record_a.metrics.parameter_validity_rate
    )
    if task_success_rate_delta < 0:
        regressions.append("task_success_rate")
    if tool_selection_rate_delta < 0:
        regressions.append("tool_selection_rate")
    if parameter_validity_rate_delta < 0:
        regressions.append("parameter_validity_rate")
    if (
        record_a.risk_constraints.passed
        and not record_b.risk_constraints.passed
    ):
        regressions.append("risk_constraints")
    if record_a.forbidden_tools.passed and not record_b.forbidden_tools.passed:
        regressions.append("forbidden_tools")

    return EvaluationComparison(
        commit_a=normalized_a,
        commit_b=normalized_b,
        task_success_rate_delta=task_success_rate_delta,
        tool_selection_rate_delta=tool_selection_rate_delta,
        parameter_validity_rate_delta=parameter_validity_rate_delta,
        risk_constraints_a_passed=record_a.risk_constraints.passed,
        risk_constraints_b_passed=record_b.risk_constraints.passed,
        forbidden_tools_a_passed=record_a.forbidden_tools.passed,
        forbidden_tools_b_passed=record_b.forbidden_tools.passed,
        regressions=tuple(regressions),
    )


def _latest_commit_record(
    records: tuple[EvaluationSummary, ...],
    commit: str,
) -> EvaluationSummary:
    matching_records = tuple(
        record
        for record in records
        if record.version_info.git_commit == commit
    )
    if not matching_records:
        raise EvaluationHistoryError(
            f"评测历史中不存在 Git commit: {commit}"
        )
    return max(
        matching_records,
        key=lambda record: record.version_info.timestamp,
    )


def _model_responses(case: EvaluationCase) -> tuple[ModelResponse, ...]:
    call_number = 0
    responses: list[ModelResponse] = []
    for response in case.scripted_responses:
        tool_calls: list[ModelToolCall] = []
        for tool_call in response.tool_calls:
            call_number += 1
            tool_calls.append(
                ModelToolCall(
                    id=f"{case.case_id}-call-{call_number}",
                    name=tool_call.name,
                    arguments=tool_call.arguments,
                )
            )
        responses.append(
            ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=tuple(tool_calls),
                ),
                finish_reason=response.finish_reason,
            )
        )
    return tuple(responses)


def _tool_results(
    messages: tuple[ModelMessage, ...],
) -> tuple[tuple[str, ToolResult], ...]:
    names_by_call_id = {
        tool_call.id: tool_call.name
        for message in messages
        if message.role == "assistant"
        for tool_call in message.tool_calls
    }
    results: list[tuple[str, ToolResult]] = []
    for message in messages:
        if message.role != "tool" or message.tool_call_id is None:
            continue
        results.append(
            (
                names_by_call_id[message.tool_call_id],
                ToolResult.model_validate_json(message.content),
            )
        )
    return tuple(results)


def _tool_results_match(
    expected: tuple[EvaluationToolExpectation, ...],
    actual: tuple[tuple[str, ToolResult], ...],
) -> bool:
    if len(expected) != len(actual):
        return False

    for expectation, (tool_name, result) in zip(expected, actual):
        error_code = result.error.code if result.error is not None else None
        if (
            expectation.name != tool_name
            or expectation.result_ok != result.ok
            or expectation.error_code != error_code
            or (
                bool(expectation.data_subset)
                and not _is_subset(expectation.data_subset, result.data)
            )
        ):
            return False
    return True


def _is_subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _is_subset(value, actual[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) <= len(actual) and all(
            _is_subset(value, actual[index])
            for index, value in enumerate(expected)
        )
    return expected == actual


def _selection_score(
    expected: tuple[str, ...],
    actual: tuple[str, ...],
) -> tuple[int, int]:
    total = max(len(expected), len(actual))
    correct = sum(
        expected_name == actual_name
        for expected_name, actual_name in zip(expected, actual)
    )
    return correct, total


def _parameter_score(
    results: tuple[tuple[str, ToolResult], ...],
) -> tuple[int, int]:
    scored_results = [
        result
        for _, result in results
        if result.error is None or result.error.code != "unknown_tool"
    ]
    valid = sum(
        result.error is None or result.error.code != "invalid_arguments"
        for result in scored_results
    )
    return valid, len(scored_results)


def _aggregate_metrics(
    cases: tuple[EvaluationCaseResult, ...],
) -> EvaluationMetrics:
    successful_tasks = sum(case.task_success for case in cases)
    correct_tool_selections = sum(
        case.correct_tool_selections for case in cases
    )
    tool_selection_total = sum(case.tool_selection_total for case in cases)
    valid_parameter_calls = sum(case.valid_parameter_calls for case in cases)
    parameter_call_total = sum(case.parameter_call_total for case in cases)

    return EvaluationMetrics(
        successful_tasks=successful_tasks,
        task_total=len(cases),
        task_success_rate=_rate(successful_tasks, len(cases)),
        correct_tool_selections=correct_tool_selections,
        tool_selection_total=tool_selection_total,
        tool_selection_rate=_rate(
            correct_tool_selections,
            tool_selection_total,
        ),
        valid_parameter_calls=valid_parameter_calls,
        parameter_call_total=parameter_call_total,
        parameter_validity_rate=_rate(
            valid_parameter_calls,
            parameter_call_total,
        ),
    )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0
