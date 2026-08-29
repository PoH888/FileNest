"""T4 Agent 合同数据集的真实程序边界运行器。"""

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from time import perf_counter
from typing import Literal
from uuid import NAMESPACE_URL, uuid5
import xml.etree.ElementTree as ElementTree

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from .agent_api import (
    _WorkspaceScopedToolRegistry,
    _build_initial_agent_messages,
    _source_references,
)
from .agent_contract_dataset import (
    AgentContractCase,
    AgentContractDataset,
    AgentContractToolCall,
    load_agent_contract_dataset,
)
from .agent_contract_metrics import (
    AgentContractMetrics,
    AgentContractRunObservation,
    ModelSource,
    calculate_agent_contract_metrics,
)
from .agent_evaluation import EvaluationVersionInfo
from .agent_loop import AgentLoop
from .document_chunker import chunk_document
from .database import Base
from .document_contracts import Document
from .fake_model_client import FakeModelClient
from .model_client import (
    ModelClient,
    ModelCallMetrics,
    ModelMessage,
    ModelResponse,
    ModelToolCall,
)
from .models import ChunkRecord, DocumentRecord, FileEntry, OperationPlanRecord
from .services import create_workspace, validate_operation_plan
from .tool_contracts import ToolResult
from .tool_registry import ToolDefinition
from .workflow_graph import open_checkpointed_workflow_graph
from langgraph.graph.state import CompiledStateGraph
from .models import Workspace


DEFAULT_DATASET_PATH = (
    Path(__file__).parents[1] / "evaluation" / "agent_contract_v1.json"
)
DEFAULT_PROMPT_VERSION = "agent_contract_prompt_v1"
DEFAULT_REPRODUCTION_COMMAND = (
    r".\.venv\Scripts\python.exe -m backend.app.agent_contract_runner "
    r"--dataset backend\evaluation\agent_contract_v1.json "
    r"--output-dir <new-run-directory> --model-source scripted_fake"
)


class AgentContractRunDirectoryError(ValueError):
    """评测运行目录必须独占创建，不能覆盖历史证据。"""


class AgentContractMetadataError(RuntimeError):
    """评测所需的版本元数据无法安全取得。"""


class AgentContractCaseResult(BaseModel):
    """不包含提示词和原始参数载荷的单用例结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    category: str
    expected_tool_names: tuple[str, ...]
    actual_tool_names: tuple[str, ...]
    actual_result: str
    expected_source_paths: tuple[str, ...]
    actual_source_paths: tuple[str, ...]
    proposal_valid: bool | None = None
    approval_intercepted: bool | None = None
    unauthorized_disk_changes: int = Field(ge=0)
    no_evidence_refusal: bool | None = None
    model_turns: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    end_to_end_success: bool
    failure_reasons: tuple[str, ...] = ()


class AgentContractEvaluationSummary(BaseModel):
    """一轮合同评测的安全结果、版本信息和聚合指标。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    dataset_version: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_source: ModelSource
    model_provider: str | None = None
    version_info: EvaluationVersionInfo
    selected_case_ids: tuple[str, ...]
    cases: tuple[AgentContractCaseResult, ...]
    metrics: AgentContractMetrics


class _MeasuredModelClient:
    """保留供应商返回的 usage，不为 Fake Model 伪造成本。"""

    def __init__(self, delegate: ModelClient) -> None:
        self._delegate = delegate
        self._metrics: list[ModelCallMetrics | None] = []

    def complete(
        self,
        *,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        response = self._delegate.complete(messages=messages, tools=tools)
        self._metrics.append(response.metrics)
        return response

    def usage(self) -> tuple[int | None, int | None, Decimal | None]:
        """只在真实响应提供相应字段时返回 token 和成本。"""

        if not self._metrics:
            return None, None, None

        token_usage = [
            metrics.token_usage
            if metrics is not None
            else None
            for metrics in self._metrics
        ]
        input_tokens = (
            sum(usage.input_tokens for usage in token_usage if usage is not None)
            if all(usage is not None for usage in token_usage)
            else None
        )
        output_tokens = (
            sum(usage.output_tokens for usage in token_usage if usage is not None)
            if all(usage is not None for usage in token_usage)
            else None
        )
        costs = [
            metrics.estimated_cost_usd
            if metrics is not None
            else None
            for metrics in self._metrics
        ]
        estimated_cost = (
            sum(
                (cost for cost in costs if cost is not None),
                start=Decimal("0"),
            )
            if all(cost is not None for cost in costs)
            else None
        )
        return input_tokens, output_tokens, estimated_cost


def run_agent_contract_evaluation(
    dataset: AgentContractDataset,
    run_root: Path,
    *,
    dataset_sha256: str,
    version_info: EvaluationVersionInfo,
    model_source: ModelSource = "scripted_fake",
    model_client_factory: Callable[[], ModelClient] | None = None,
    model_provider: str | None = None,
    case_ids: Sequence[str] | None = None,
) -> AgentContractEvaluationSummary:
    """在隔离的临时工作区中执行合同数据集并保存四类证据。"""

    _validate_dataset_hash(dataset_sha256)
    selected_cases = _select_cases(
        dataset,
        model_source=model_source,
        case_ids=case_ids,
    )
    if model_source == "real_model" and model_client_factory is None:
        raise ValueError("real_model 评测必须提供 model_client_factory")
    if run_root.exists() or run_root.is_symlink():
        raise AgentContractRunDirectoryError(
            "评测运行目录必须尚不存在"
        )

    run_root.mkdir(parents=True)
    workspace_root = _materialize_fixture(dataset, run_root / "workspace")
    (workspace_root / "reports" / "archive").mkdir(parents=True)
    (run_root / "quarantine").mkdir()
    database_path = run_root / "evaluation.db"
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")

    try:
        Base.metadata.create_all(bind=engine)
        with Session(engine, expire_on_commit=False) as session:
            workspace, file_ids = _seed_fixture(session, dataset, workspace_root)
            with open_checkpointed_workflow_graph(
                run_root / "workflow-checkpoints.sqlite",
                operation_plan_validator=lambda plan: validate_operation_plan(
                    session,
                    plan,
                ),
            ) as graph:
                observations: list[AgentContractRunObservation] = []
                case_results: list[AgentContractCaseResult] = []
                for case in selected_cases:
                    observation, case_result = _run_case(
                        case,
                        session=session,
                        graph=graph,
                        workspace_root=workspace_root,
                        quarantine_root=run_root / "quarantine",
                        workspace_id=workspace.id,
                        file_ids=file_ids,
                        model_source=model_source,
                        model_client_factory=model_client_factory,
                        prompt_version=version_info.prompt_version,
                    )
                    observations.append(observation)
                    case_results.append(case_result)
    finally:
        engine.dispose()

    metrics = calculate_agent_contract_metrics(
        dataset,
        observations,
        model_source=model_source,
    )
    summary = AgentContractEvaluationSummary(
        dataset_version=dataset.dataset_version,
        dataset_sha256=dataset_sha256,
        model_source=model_source,
        model_provider=model_provider,
        version_info=version_info,
        selected_case_ids=tuple(case.case_id for case in selected_cases),
        cases=tuple(case_results),
        metrics=metrics,
    )
    _write_exclusive_text(
        run_root / "evaluation-result.json",
        summary.model_dump_json(indent=2) + "\n",
    )
    _write_exclusive_text(
        run_root / "evaluation-summary.md",
        render_contract_report(summary),
    )
    _write_exclusive_text(
        run_root / "run-metadata.json",
        json.dumps(
            _run_metadata(summary),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    _write_junit_report(summary, run_root / "junit.xml")
    return summary


def _select_cases(
    dataset: AgentContractDataset,
    *,
    model_source: ModelSource,
    case_ids: Sequence[str] | None,
) -> tuple[AgentContractCase, ...]:
    by_id = {case.case_id: case for case in dataset.cases}
    if case_ids is None:
        selected_ids = (
            tuple(case.case_id for case in dataset.cases)
            if model_source == "scripted_fake"
            else tuple(case.case_id for case in dataset.cases[:6])
        )
    else:
        selected_ids = tuple(case_ids)

    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("评测用例选择必须非空且不能重复")
    if any(case_id not in by_id for case_id in selected_ids):
        raise ValueError("评测用例选择包含未知 case_id")
    if model_source == "real_model" and not 5 <= len(selected_ids) <= 8:
        raise ValueError("real_model 评测必须选择 5 到 8 条用例")
    return tuple(by_id[case_id] for case_id in selected_ids)


def _materialize_fixture(
    dataset: AgentContractDataset,
    target_root: Path,
) -> Path:
    if target_root.exists() or target_root.is_symlink():
        raise AgentContractRunDirectoryError(
            "评测工作区目标必须是尚不存在的目录"
        )
    target_root.mkdir(parents=True)
    for fixture_file in dataset.fixture.files:
        destination = target_root.joinpath(
            *PurePosixPath(fixture_file.relative_path).parts
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            fixture_file.content,
            encoding="utf-8",
            newline="\n",
        )
    return target_root.resolve(strict=True)


def _seed_fixture(
    session: Session,
    dataset: AgentContractDataset,
    workspace_root: Path,
) -> tuple[Workspace, dict[str, int]]:
    workspace = create_workspace(
        session,
        dataset.fixture.workspace_name,
        str(workspace_root),
    )
    file_ids: dict[str, int] = {}
    for fixture_file in dataset.fixture.files:
        path = workspace_root.joinpath(
            *PurePosixPath(fixture_file.relative_path).parts
        )
        metadata = path.stat()
        file_entry = FileEntry(
            workspace_id=workspace.id,
            relative_path=fixture_file.relative_path,
            name=path.name,
            extension=path.suffix.casefold(),
            size_bytes=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
        )
        session.add(file_entry)
        session.flush()
        file_ids[fixture_file.relative_path] = file_entry.id
    session.commit()

    for fixture_file in dataset.fixture.files:
        if fixture_file.relative_path in dataset.fixture.sensitive_paths:
            continue
        if Path(fixture_file.relative_path).suffix.casefold() not in {
            ".md",
            ".txt",
        }:
            continue
        path = workspace_root.joinpath(
            *PurePosixPath(fixture_file.relative_path).parts
        )
        document = Document(
            document_id=uuid5(
                NAMESPACE_URL,
                f"{dataset.fixture.fixture_id}:{fixture_file.relative_path}",
            ),
            workspace_id=workspace.id,
            file_entry_id=file_ids[fixture_file.relative_path],
            source_relative_path=fixture_file.relative_path,
            source_format=(
                "markdown"
                if path.suffix.casefold() == ".md"
                else "text"
            ),
            normalized_text=fixture_file.content,
            source_version=sha256(
                fixture_file.content.encode("utf-8")
            ).hexdigest(),
            source_updated_at=datetime.fromtimestamp(
                path.stat().st_mtime,
                tz=timezone.utc,
            ),
        )
        record = DocumentRecord.from_contract(document)
        session.add(record)
        session.add_all(
            ChunkRecord.from_contract(chunk)
            for chunk in chunk_document(document)
        )
    session.commit()
    return workspace, file_ids


def _run_case(
    case: AgentContractCase,
    *,
    session: Session,
    graph: CompiledStateGraph,
    workspace_root: Path,
    quarantine_root: Path,
    workspace_id: int,
    file_ids: dict[str, int],
    model_source: ModelSource,
    model_client_factory: Callable[[], ModelClient] | None,
    prompt_version: str,
) -> tuple[AgentContractRunObservation, AgentContractCaseResult]:
    before_snapshot = _workspace_snapshot(workspace_root)
    before_plan_ids = _plan_ids(session)
    if model_source == "scripted_fake":
        model_client: ModelClient = FakeModelClient(_fake_responses(case))
    else:
        assert model_client_factory is not None
        model_client = model_client_factory()
    measured_client = _MeasuredModelClient(model_client)
    registry = _WorkspaceScopedToolRegistry(
        session,
        workspace_id,
        graph,
        quarantine_root,
    )
    started_at = perf_counter()
    run_result = AgentLoop(
        model_client=measured_client,
        tool_registry=registry,
        prompt_version=prompt_version,
    ).run(
        _build_initial_agent_messages(
            case.input.workspace_id,
            case.input.request_text,
        ),
        retry_base_delay_seconds=0,
    )
    latency_ms = (perf_counter() - started_at) * 1_000
    after_snapshot = _workspace_snapshot(workspace_root)
    unauthorized_disk_changes = _count_snapshot_changes(
        before_snapshot,
        after_snapshot,
    )

    actual_calls = _actual_tool_calls(run_result.messages)
    tool_results = _tool_results(run_result.messages)
    actual_source_paths = tuple(
        source.relative_path
        for source in _source_references(run_result.messages, workspace_id)
    )
    valid_parameter_calls = sum(
        registry.validate(call.name, call.arguments).ok
        for call in actual_calls
    )
    parameter_call_total = len(actual_calls)
    new_plans = _new_plans(session, before_plan_ids)
    actual_result = _classify_result(
        case,
        run_result.status,
        tool_results,
        actual_source_paths,
        new_plans,
    )
    proposal_valid: bool | None = None
    approval_intercepted: bool | None = None
    if case.expected.expected_proposal_operation is not None:
        proposal_valid = (
            len(new_plans) == 1
            and new_plans[0].status == "WAITING_APPROVAL"
            and new_plans[0].operation_type
            == case.expected.expected_proposal_operation
        )
        approval_intercepted = proposal_valid and unauthorized_disk_changes == 0

    no_evidence_refusal: bool | None = None
    if "no_evidence_refusal" in case.expected.allowed_results:
        no_evidence_refusal = actual_result == "no_evidence_refusal"

    expected_calls = case.expected.tool_calls
    expected_paths = set(case.expected.expected_source_paths)
    actual_paths = set(actual_source_paths)
    end_to_end_success = (
        tuple(expected_calls) == tuple(actual_calls)
        and actual_result in case.expected.allowed_results
        and actual_paths == expected_paths
        and unauthorized_disk_changes
        == case.security_assertions.unauthorized_disk_changes
        and (
            case.expected.expected_proposal_operation is None
            or (proposal_valid is True and approval_intercepted is True)
        )
    )
    failure_reasons = _failure_reasons(
        case,
        actual_calls=actual_calls,
        actual_result=actual_result,
        actual_paths=actual_paths,
        unauthorized_disk_changes=unauthorized_disk_changes,
        proposal_valid=proposal_valid,
        no_evidence_refusal=no_evidence_refusal,
    )
    input_tokens, output_tokens, estimated_cost = measured_client.usage()
    observation = AgentContractRunObservation(
        case_id=case.case_id,
        actual_tool_calls=actual_calls,
        actual_result=actual_result,
        valid_parameter_calls=valid_parameter_calls,
        parameter_call_total=parameter_call_total,
        proposal_valid=proposal_valid,
        approval_intercepted=approval_intercepted,
        unauthorized_disk_changes=unauthorized_disk_changes,
        actual_source_paths=actual_source_paths,
        no_evidence_refusal=no_evidence_refusal,
        end_to_end_success=end_to_end_success,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimated_cost,
    )
    case_result = AgentContractCaseResult(
        case_id=case.case_id,
        category=case.category,
        expected_tool_names=tuple(call.name for call in expected_calls),
        actual_tool_names=tuple(call.name for call in actual_calls),
        actual_result=actual_result,
        expected_source_paths=case.expected.expected_source_paths,
        actual_source_paths=actual_source_paths,
        proposal_valid=proposal_valid,
        approval_intercepted=approval_intercepted,
        unauthorized_disk_changes=unauthorized_disk_changes,
        no_evidence_refusal=no_evidence_refusal,
        model_turns=run_result.model_turns,
        latency_ms=latency_ms,
        end_to_end_success=end_to_end_success,
        failure_reasons=failure_reasons,
    )
    return observation, case_result


def _fake_responses(case: AgentContractCase) -> tuple[ModelResponse, ...]:
    tool_calls = tuple(
        ModelToolCall(
            id=f"{case.case_id}-call-{index}",
            name=call.name,
            arguments=call.arguments,
        )
        for index, call in enumerate(case.expected.tool_calls, start=1)
    )
    if tool_calls:
        first_response = ModelResponse(
            message=ModelMessage(
                role="assistant",
                tool_calls=tool_calls,
            ),
            finish_reason="tool_calls",
        )
        final_content = (
            "没有足够证据，无法回答。"
            if "no_evidence_refusal" in case.expected.allowed_results
            and not case.expected.expected_source_paths
            else "已生成待审批计划，未执行文件操作。"
            if case.expected.expected_proposal_operation is not None
            else "已完成固定合同用例。"
        )
        return (
            first_response,
            ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=final_content,
                ),
                finish_reason="stop",
            ),
        )
    return (
        ModelResponse(
            message=ModelMessage(
                role="assistant",
                content="没有足够证据，无法回答。",
            ),
            finish_reason="stop",
        ),
    )


def _actual_tool_calls(
    messages: Sequence[ModelMessage],
) -> tuple[AgentContractToolCall, ...]:
    return tuple(
        AgentContractToolCall(
            name=tool_call.name,
            arguments=tool_call.arguments,
        )
        for message in messages
        if message.role == "assistant"
        for tool_call in message.tool_calls
    )


def _tool_results(
    messages: Sequence[ModelMessage],
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
                names_by_call_id.get(message.tool_call_id, "unknown_tool"),
                ToolResult.model_validate_json(message.content or "{}"),
            )
        )
    return tuple(results)


def _classify_result(
    case: AgentContractCase,
    run_status: str,
    tool_results: Sequence[tuple[str, ToolResult]],
    source_paths: Sequence[str],
    new_plans: Sequence[OperationPlanRecord],
) -> str:
    error_codes = tuple(
        result.error.code
        for _, result in tool_results
        if result.error is not None
    )
    if "unknown_tool" in error_codes:
        return "rejected_forbidden_tool"
    if "invalid_arguments" in error_codes:
        return "rejected_invalid_arguments"
    if case.security_assertions.sensitive_path_action == "deny" and error_codes:
        return "rejected_sensitive_path"
    if new_plans and case.expected.expected_proposal_operation is not None:
        return "proposal_waiting_approval"
    if not source_paths and "no_evidence_refusal" in case.expected.allowed_results:
        return "no_evidence_refusal"
    return "completed" if run_status == "completed" else run_status


def _failure_reasons(
    case: AgentContractCase,
    *,
    actual_calls: Sequence[AgentContractToolCall],
    actual_result: str,
    actual_paths: set[str],
    unauthorized_disk_changes: int,
    proposal_valid: bool | None,
    no_evidence_refusal: bool | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if tuple(actual_calls) != tuple(case.expected.tool_calls):
        reasons.append("tool_or_argument_mismatch")
    if actual_result not in case.expected.allowed_results:
        reasons.append("result_not_allowed")
    if actual_paths != set(case.expected.expected_source_paths):
        reasons.append("citation_paths_mismatch")
    if unauthorized_disk_changes != case.security_assertions.unauthorized_disk_changes:
        reasons.append("unauthorized_disk_changes")
    if case.expected.expected_proposal_operation is not None and not proposal_valid:
        reasons.append("proposal_not_waiting_approval")
    if no_evidence_refusal is False:
        reasons.append("no_evidence_refusal_missing")
    return tuple(reasons)


def _plan_ids(session: Session) -> set[str]:
    return {
        plan.plan_id
        for plan in session.scalars(select(OperationPlanRecord)).all()
    }


def _new_plans(
    session: Session,
    before_plan_ids: set[str],
) -> tuple[OperationPlanRecord, ...]:
    return tuple(
        plan
        for plan in session.scalars(
            select(OperationPlanRecord).order_by(OperationPlanRecord.created_at)
        ).all()
        if plan.plan_id not in before_plan_ids
    )


def _workspace_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def _count_snapshot_changes(
    before: dict[str, str],
    after: dict[str, str],
) -> int:
    return sum(
        before.get(path) != after.get(path)
        for path in set(before) | set(after)
    )


def render_contract_report(summary: AgentContractEvaluationSummary) -> str:
    """生成不含提示词和原始工具参数的可读报告。"""

    metrics = summary.metrics
    lines = [
        "# FileNest Agent 合同评测报告",
        "",
        "## 评测边界",
        "",
        f"- 数据集版本：`{summary.dataset_version}`",
        f"- 数据集 SHA-256：`{summary.dataset_sha256}`",
        f"- 模型来源：`{summary.model_source}`",
        (
            "- 评测结果表示当前程序边界上的固定合同执行结果；"
            "scripted_fake 不代表真实模型质量。"
            if summary.model_source == "scripted_fake"
            else "- 评测包含真实模型调用；结果受模型、服务状态和网络影响。"
        ),
        "- 报告不保存提示词、原始工具参数、工具返回载荷或绝对路径。",
        "",
        "## 版本与复现",
        "",
        f"- Prompt version：`{summary.version_info.prompt_version}`",
        f"- Model version：`{summary.version_info.model_version}`",
        f"- Git commit：`{summary.version_info.git_commit}`",
        (
            "- Timestamp："
            f"`{summary.version_info.timestamp.isoformat()}`"
        ),
        f"- 执行命令：`{_reproduction_command(summary.model_source)}`",
        "",
        "## 指标",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        (
            "| Tool selection accuracy | "
            f"{_format_rate(metrics.tool_selection_accuracy)} |"
        ),
        (
            "| Argument accuracy | "
            f"{_format_rate(metrics.argument_accuracy)} |"
        ),
        (
            "| Argument validity rate | "
            f"{_format_rate(metrics.argument_validity_rate)} |"
        ),
        (
            "| Proposal validity rate | "
            f"{_format_rate(metrics.proposal_validity_rate)} |"
        ),
        (
            "| Approval interception rate | "
            f"{_format_rate(metrics.approval_interception_rate)} |"
        ),
        (
            "| Citation precision / coverage | "
            f"{_format_rate(metrics.citation_precision)} / "
            f"{_format_rate(metrics.citation_coverage)} |"
        ),
        (
            "| No-evidence refusal rate | "
            f"{_format_rate(metrics.no_evidence_refusal_rate)} |"
        ),
        (
            "| End-to-end success rate | "
            f"{metrics.end_to_end_success_rate:.2%} |"
        ),
        (
            "| Unauthorized disk changes | "
            f"{metrics.unauthorized_disk_changes} "
            f"({'PASS' if metrics.unauthorized_disk_changes_gate_passed else 'FAIL'}) |"
        ),
        f"| Latency P50 / P95 | {metrics.latency_p50_ms:.3f} / {metrics.latency_p95_ms:.3f} ms |",
        "",
        "## 用例结果",
        "",
        "| Case | Category | Result | E2E | Failure details |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in summary.cases:
        lines.append(
            f"| `{case.case_id}` | `{case.category}` | `{case.actual_result}` | "
            f"{'PASS' if case.end_to_end_success else 'FAIL'} | "
            f"{', '.join(case.failure_reasons) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Evidence files",
            "",
            "- `evaluation-result.json`",
            "- `evaluation-summary.md`",
            "- `junit.xml`",
            "- `run-metadata.json`",
            "",
        ]
    )
    return "\n".join(lines)


def _run_metadata(summary: AgentContractEvaluationSummary) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "dataset_sha256": summary.dataset_sha256,
        "dataset_version": summary.dataset_version,
        "git_commit": summary.version_info.git_commit,
        "execution_command": _reproduction_command(summary.model_source),
        "model_source": summary.model_source,
        "model_type": (
            "deterministic_scripted_fake"
            if summary.model_source == "scripted_fake"
            else "configured_real_model"
        ),
        "model_provider": summary.model_provider,
        "model_version": summary.version_info.model_version,
        "prompt_version": summary.version_info.prompt_version,
        "timestamp": summary.version_info.timestamp.isoformat(),
        "case_count": len(summary.cases),
        "selected_case_ids": list(summary.selected_case_ids),
        "failure_case_ids": [
            case.case_id for case in summary.cases if not case.end_to_end_success
        ],
    }


def _write_junit_report(
    summary: AgentContractEvaluationSummary,
    path: Path,
) -> None:
    failed_cases = [
        case for case in summary.cases if not case.end_to_end_success
    ]
    suite = ElementTree.Element(
        "testsuite",
        {
            "name": "FileNest Agent Contract Evaluation",
            "tests": str(len(summary.cases)),
            "failures": str(len(failed_cases)),
        },
    )
    for case in summary.cases:
        testcase = ElementTree.SubElement(
            suite,
            "testcase",
            {
                "classname": case.category,
                "name": case.case_id,
                "time": f"{case.latency_ms / 1000:.6f}",
            },
        )
        if not case.end_to_end_success:
            failure = ElementTree.SubElement(
                testcase,
                "failure",
                {"message": "contract case failed"},
            )
            failure.text = "; ".join(case.failure_reasons)
    ElementTree.indent(suite, space="  ")
    with path.open("x", encoding="utf-8", newline="\n") as output_file:
        ElementTree.ElementTree(suite).write(
            output_file,
            encoding="unicode",
            xml_declaration=True,
        )


def _write_exclusive_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output_file:
        output_file.write(content)


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _validate_dataset_hash(value: str) -> None:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("dataset_sha256 must be a 64-character hexadecimal digest")


def _reproduction_command(model_source: ModelSource) -> str:
    if model_source == "real_model":
        return DEFAULT_REPRODUCTION_COMMAND.replace(
            "scripted_fake",
            "real_model",
        )
    return DEFAULT_REPRODUCTION_COMMAND


def _current_git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        raise AgentContractMetadataError(
            "无法读取评测所需的 Git commit"
        ) from None
    commit = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", commit):
        raise AgentContractMetadataError(
            "无法读取评测所需的 Git commit"
        )
    return commit.casefold()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 FileNest Agent 合同固定评测。"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-source",
        choices=("scripted_fake", "real_model"),
        default="scripted_fake",
    )
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--case-id", action="append", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_bytes = args.dataset.read_bytes()
    dataset = load_agent_contract_dataset(args.dataset)
    model_client_factory: Callable[[], ModelClient] | None = None
    model_provider = None
    model_version = "scripted_fake"
    if args.model_source == "real_model":
        from .model_settings import ModelSettings
        from .openai_compatible_model_client import OpenAICompatibleModelClient

        settings = ModelSettings()
        real_model_client = OpenAICompatibleModelClient(settings)
        model_provider = settings.provider
        model_version = settings.name

        def create_model_client() -> ModelClient:
            return real_model_client

        model_client_factory = create_model_client

    version_info = EvaluationVersionInfo(
        prompt_version=args.prompt_version,
        model_version=model_version,
        git_commit=_current_git_commit(Path(__file__).resolve().parents[2]),
        evaluation_dataset_version=dataset.dataset_version,
        timestamp=datetime.now(timezone.utc),
    )
    summary = run_agent_contract_evaluation(
        dataset,
        args.output_dir,
        dataset_sha256=sha256(dataset_bytes).hexdigest(),
        version_info=version_info,
        model_source=args.model_source,
        model_client_factory=model_client_factory,
        model_provider=model_provider,
        case_ids=args.case_id,
    )
    print(
        "评测结果："
        f"{args.output_dir / 'evaluation-result.json'}"
    )
    print(f"通过用例：{summary.metrics.end_to_end_successes}/{summary.metrics.end_to_end_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
