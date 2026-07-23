"""只读 Agent 固定数据集的可复现运行与评分。"""

from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from .agent_evaluation import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationToolExpectation,
    materialize_evaluation_workspace,
)
from .agent_loop import AgentLoop
from .agent_observability import SqlAlchemyAgentRunRecorder
from .database import Base
from .fake_model_client import FakeModelClient
from .model_client import ModelMessage, ModelResponse, ModelToolCall
from .services import create_workspace, scan_workspace
from .tool_contracts import ToolResult
from .tool_registry import build_read_tool_registry


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


class EvaluationSummary(BaseModel):
    """一次完整评测返回的逐用例证据和聚合指标。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    dataset_schema_version: str
    model_source: str
    cases: tuple[EvaluationCaseResult, ...]
    metrics: EvaluationMetrics
    total_model_turns: int = Field(ge=0)
    total_run_latency_ms: float = Field(ge=0)
    total_estimated_model_cost_usd: Decimal = Field(ge=0)


def run_evaluation_dataset(
    dataset: EvaluationDataset,
    run_root: Path,
) -> EvaluationSummary:
    """在全新隔离目录中运行数据集并返回确定性评分。"""

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
            case_results = tuple(
                _run_case(case, session)
                for case in dataset.cases
            )
    finally:
        engine.dispose()

    summary = EvaluationSummary(
        schema_version="1.0",
        dataset_schema_version=dataset.schema_version,
        model_source="scripted_fake",
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
) -> EvaluationCaseResult:
    model_client = FakeModelClient(_model_responses(case))
    loop = AgentLoop(
        model_client=model_client,
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
        estimated_model_cost_usd=Decimal("0"),
    )


def save_evaluation_summary(
    summary: EvaluationSummary,
    result_path: Path,
) -> None:
    """以独占创建方式保存安全结果，避免覆盖历史评测证据。"""

    with result_path.open("x", encoding="utf-8", newline="\n") as result_file:
        result_file.write(summary.model_dump_json(indent=2))
        result_file.write("\n")


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
