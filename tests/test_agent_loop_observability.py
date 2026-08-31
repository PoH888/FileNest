from hashlib import sha256
from decimal import Decimal
from pathlib import Path
from threading import Event

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from backend.app.agent_loop import AgentLoop
from backend.app.agent_observability import (
    AgentObservabilityError,
    SqlAlchemyAgentRunRecorder,
)
from backend.app.database import Base
from backend.app.fake_model_client import FakeModelClient
from backend.app.model_client import (
    ModelClientRequestError,
    ModelCallMetrics,
    ModelMessage,
    ModelResponse,
    ModelTokenUsage,
    ModelToolCall,
)
from backend.app.models import (
    AgentMessage,
    AgentMetric,
    AgentModelRun,
    AgentRun,
    AgentStep,
    AgentToolCall,
)
from backend.app.tool_contracts import Tool, ToolResult
from backend.app.tool_registry import ToolRegistry


class SearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: int
    keyword: str


def _registry(handler) -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                name="search_files",
                description="搜索授权工作区内的文件索引",
                arguments_model=SearchArguments,
                handler=handler,
            )
        ]
    )


def _tool_response(
    *tool_calls: ModelToolCall,
    content: str | None = None,
    metrics: ModelCallMetrics | None = None,
    model_provider: str | None = None,
    model_name: str | None = None,
) -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        ),
        finish_reason="tool_calls",
        model_provider=model_provider,
        model_name=model_name,
        metrics=metrics,
    )


def _final_response(
    content: str = "查询完成。",
    *,
    metrics: ModelCallMetrics | None = None,
    model_provider: str | None = None,
    model_name: str | None = None,
) -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(role="assistant", content=content),
        finish_reason="stop",
        model_provider=model_provider,
        model_name=model_name,
        metrics=metrics,
    )


def test_agent_loop_persists_resume_context_without_tool_payload_rows(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'loop-observability.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)
    sensitive_prompt = "private-user-prompt-must-not-be-stored"
    sensitive_assistant_text = "private-assistant-text-must-not-be-stored"
    sensitive_keyword = "private-keyword-must-not-be-stored"
    sensitive_result = "private/result/path-must-not-be-stored.txt"
    raw_call_ids = ["call_private_1", "call_private_2"]
    handler_calls: list[str] = []

    def handle(arguments: BaseModel) -> ToolResult:
        parsed = SearchArguments.model_validate(arguments)
        handler_calls.append(parsed.keyword)
        return ToolResult.success(
            {"relative_path": sensitive_result, "count": 1}
        )

    model_client = FakeModelClient(
        [
            _tool_response(
                ModelToolCall(
                    id=raw_call_ids[0],
                    name="search_files",
                    arguments={
                        "workspace_id": 1,
                        "keyword": sensitive_keyword,
                    },
                ),
                ModelToolCall(
                    id=raw_call_ids[1],
                    name="search_files",
                    arguments={
                        "workspace_id": 1,
                        "keyword": sensitive_keyword,
                    },
                ),
                content=sensitive_assistant_text,
            ),
            _final_response(),
        ]
    )

    try:
        with Session(engine, expire_on_commit=False) as session:
            loop = AgentLoop(
                model_client=model_client,
                tool_registry=_registry(handle),
                recorder=SqlAlchemyAgentRunRecorder(session),
            )

            result = loop.run(
                [ModelMessage(role="user", content=sensitive_prompt)],
            )

            saved_run = session.scalar(select(AgentRun))
            saved_calls = list(
                session.scalars(
                    select(AgentToolCall).order_by(AgentToolCall.sequence_no)
                )
            )
            persisted_tool_rows = repr(
                session.execute(text("SELECT * FROM agent_tool_calls")).all()
            )

            assert result.status == "completed"
            assert saved_run is not None
            assert saved_run.status == "completed"
            assert saved_run.model_turns == 2
            assert [call.sequence_no for call in saved_calls] == [1, 2]
            assert [call.status for call in saved_calls] == [
                "succeeded",
                "succeeded",
            ]
            assert [call.model_call_id for call in saved_calls] == [
                sha256(call_id.encode("utf-8")).hexdigest()
                for call_id in raw_call_ids
            ]
            assert handler_calls == [sensitive_keyword, sensitive_keyword]
            assert saved_run.context_json is not None
            assert sensitive_prompt in saved_run.context_json
            assert sensitive_assistant_text in saved_run.context_json
            for sensitive_value in [
                sensitive_assistant_text,
                sensitive_keyword,
                sensitive_result,
                *raw_call_ids,
            ]:
                assert sensitive_value not in persisted_tool_rows
    finally:
        engine.dispose()


def test_agent_loop_accumulates_metrics_from_two_model_calls(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'aggregated-metrics.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)
    first_metrics = ModelCallMetrics(
        latency_ms=12.5,
        requested_max_output_tokens=512,
        token_usage=ModelTokenUsage(
            input_tokens=10,
            output_tokens=4,
            total_tokens=14,
        ),
        estimated_cost_usd=Decimal("0.001"),
    )
    second_metrics = ModelCallMetrics(
        latency_ms=25.0,
        requested_max_output_tokens=512,
        token_usage=ModelTokenUsage(
            input_tokens=30,
            output_tokens=6,
            total_tokens=36,
        ),
        estimated_cost_usd=Decimal("0.002"),
    )

    try:
        with Session(engine, expire_on_commit=False) as session:
            loop = AgentLoop(
                model_client=FakeModelClient(
                    [
                        _tool_response(
                            ModelToolCall(
                                id="call_metrics_1",
                                name="unknown_tool",
                                arguments={},
                            ),
                            model_provider="fake",
                            model_name="deterministic-model",
                            metrics=first_metrics,
                        ),
                        _final_response(
                            model_provider="fake",
                            model_name="deterministic-model",
                            metrics=second_metrics,
                        ),
                    ]
                ),
                tool_registry=ToolRegistry([]),
                recorder=SqlAlchemyAgentRunRecorder(session),
                prompt_version="agent-system-v1",
            )

            result = loop.run(
                [ModelMessage(role="user", content="统计指标")],
            )
            saved_run = session.scalar(select(AgentRun))

            assert result.status == "completed"
            assert saved_run is not None
            assert saved_run.model_provider == "fake"
            assert saved_run.model_name == "deterministic-model"
            assert saved_run.prompt_version == "agent-system-v1"
            assert saved_run.latency_ms == 37.5
            assert saved_run.input_tokens == 40
            assert saved_run.output_tokens == 10
            assert saved_run.estimated_cost_usd == Decimal("0.0030000000")
    finally:
        engine.dispose()


def test_agent_loop_keeps_usage_null_when_provider_omits_usage(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'missing-usage-metrics.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine, expire_on_commit=False) as session:
            loop = AgentLoop(
                model_client=FakeModelClient(
                    [
                        _final_response(
                            model_provider="fake",
                            model_name="deterministic-model",
                            metrics=ModelCallMetrics(
                                latency_ms=8.0,
                                requested_max_output_tokens=512,
                            ),
                        )
                    ]
                ),
                tool_registry=ToolRegistry([]),
                recorder=SqlAlchemyAgentRunRecorder(session),
                prompt_version="agent-system-v1",
            )

            result = loop.run(
                [ModelMessage(role="user", content="没有 usage")],
            )
            saved_run = session.scalar(select(AgentRun))

            assert result.status == "completed"
            assert saved_run is not None
            assert saved_run.model_provider == "fake"
            assert saved_run.model_name == "deterministic-model"
            assert saved_run.prompt_version == "agent-system-v1"
            assert saved_run.latency_ms == 8.0
            assert saved_run.input_tokens is None
            assert saved_run.output_tokens is None
            assert saved_run.estimated_cost_usd is None
    finally:
        engine.dispose()


def test_agent_loop_preserves_known_metrics_when_run_later_fails(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'failed-metrics.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)
    model_error = ModelClientRequestError(
        code="model_connection_error",
        message="provider unavailable",
        retryable=False,
    )
    first_response = _tool_response(
        ModelToolCall(
            id="call_failed_metrics_1",
            name="unknown_tool",
            arguments={},
        ),
        model_provider="fake",
        model_name="deterministic-model",
        metrics=ModelCallMetrics(
            latency_ms=17.5,
            requested_max_output_tokens=512,
            token_usage=ModelTokenUsage(
                input_tokens=7,
                output_tokens=3,
                total_tokens=10,
            ),
            estimated_cost_usd=Decimal("0.0005"),
        ),
    )

    class FailingAfterFirstResponse:
        def __init__(self) -> None:
            self._returned_first_response = False

        def complete(self, **_: object) -> ModelResponse:
            if not self._returned_first_response:
                self._returned_first_response = True
                return first_response
            raise model_error

    try:
        with Session(engine, expire_on_commit=False) as session:
            loop = AgentLoop(
                model_client=FailingAfterFirstResponse(),
                tool_registry=ToolRegistry([]),
                recorder=SqlAlchemyAgentRunRecorder(session),
                prompt_version="agent-system-v1",
            )

            result = loop.run(
                [ModelMessage(role="user", content="失败但保留指标")],
                max_model_retries=0,
            )
            saved_run = session.scalar(select(AgentRun))

            assert result.status == "failed"
            assert saved_run is not None
            assert saved_run.error_code == "model_connection_error"
            assert saved_run.model_provider == "fake"
            assert saved_run.model_name == "deterministic-model"
            assert saved_run.prompt_version == "agent-system-v1"
            assert saved_run.latency_ms == 17.5
            assert saved_run.input_tokens == 7
            assert saved_run.output_tokens == 3
            assert saved_run.estimated_cost_usd == Decimal("0.0005000000")
    finally:
        engine.dispose()


def test_agent_loop_records_unknown_tool_as_safe_rejection(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'unknown-tool-trace.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)
    invented_tool_name = "private_invented_tool_name"
    model_client = FakeModelClient(
        [
            _tool_response(
                ModelToolCall(
                    id="call_unknown_1",
                    name=invented_tool_name,
                    arguments={},
                )
            ),
            _final_response(),
        ]
    )

    try:
        with Session(engine, expire_on_commit=False) as session:
            loop = AgentLoop(
                model_client=model_client,
                tool_registry=ToolRegistry([]),
                recorder=SqlAlchemyAgentRunRecorder(session),
            )

            result = loop.run([ModelMessage(role="user", content="调用工具")])

            saved_call = session.scalar(select(AgentToolCall))
            assert result.status == "completed"
            assert saved_call is not None
            assert saved_call.tool_name == "unknown_tool"
            assert saved_call.status == "rejected"
            assert saved_call.error_code == "unknown_tool"
            assert invented_tool_name not in repr(saved_call.__dict__)
    finally:
        engine.dispose()


def test_agent_loop_persists_complete_lifecycle_graph_with_safe_summaries(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'complete-lifecycle-graph.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)
    sensitive_prompt = "private prompt must not be in lifecycle summaries"
    sensitive_keyword = "private keyword must not be in lifecycle summaries"

    def handle(arguments: BaseModel) -> ToolResult:
        parsed = SearchArguments.model_validate(arguments)
        assert parsed.keyword == sensitive_keyword
        return ToolResult.success({"items": [{"relative_path": "report.txt"}]})

    try:
        with Session(engine, expire_on_commit=False) as session:
            loop = AgentLoop(
                model_client=FakeModelClient(
                    [
                        _tool_response(
                            ModelToolCall(
                                id="call_lifecycle_1",
                                name="search_files",
                                arguments={
                                    "workspace_id": 1,
                                    "keyword": sensitive_keyword,
                                },
                            ),
                            content="private assistant text",
                            model_provider="fake",
                            model_name="deterministic-model",
                            metrics=ModelCallMetrics(
                                latency_ms=4.0,
                                requested_max_output_tokens=128,
                            ),
                        ),
                        _final_response(
                            content="private final answer",
                            model_provider="fake",
                            model_name="deterministic-model",
                            metrics=ModelCallMetrics(
                                latency_ms=5.0,
                                requested_max_output_tokens=128,
                            ),
                        ),
                    ]
                ),
                tool_registry=_registry(handle),
                recorder=SqlAlchemyAgentRunRecorder(session),
                prompt_version="agent-system-v1",
            )

            result = loop.run(
                [ModelMessage(role="user", content=sensitive_prompt)]
            )

            saved_run = session.scalar(select(AgentRun))
            assert result.status == "completed"
            assert saved_run is not None
            assert len(saved_run.sessions) == 1
            saved_session = saved_run.sessions[0]
            saved_steps = list(
                session.scalars(
                    select(AgentStep)
                    .where(AgentStep.agent_session_id == saved_session.id)
                    .order_by(AgentStep.step_index)
                )
            )
            saved_messages = list(
                session.scalars(
                    select(AgentMessage)
                    .join(AgentStep)
                    .where(AgentStep.agent_session_id == saved_session.id)
                    .order_by(AgentMessage.id)
                )
            )
            saved_model_runs = list(session.scalars(select(AgentModelRun)))
            saved_metrics = list(session.scalars(select(AgentMetric)))
            saved_tool_calls = list(session.scalars(select(AgentToolCall)))

            assert [step.step_index for step in saved_steps] == [0, 1]
            assert [step.status for step in saved_steps] == [
                "completed",
                "completed",
            ]
            assert [message.message_type for message in saved_messages] == [
                "user",
                "assistant",
                "tool_call",
                "tool_result",
                "tool_result",
                "assistant",
            ]
            assert len(saved_model_runs) == 2
            assert {metric.metric_name for metric in saved_metrics} >= {
                "model_turn",
                "latency_ms",
            }
            assert len(saved_tool_calls) == 1
            assert saved_tool_calls[0].agent_step_id == saved_steps[0].id
            lifecycle_payload = repr(
                [
                    step.input
                    for step in saved_steps
                ]
                + [step.output_summary for step in saved_steps]
                + [message.payload_json for message in saved_messages]
            )
            assert sensitive_prompt not in lifecycle_payload
            assert sensitive_keyword not in lifecycle_payload
            assert "private assistant text" not in lifecycle_payload
            assert "private final answer" not in lifecycle_payload
    finally:
        engine.dispose()


def test_agent_loop_records_tool_execution_failure(tmp_path: Path) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'tool-failure-trace.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    def fail(_: BaseModel) -> ToolResult:
        raise RuntimeError("private handler failure")

    model_client = FakeModelClient(
        [
            _tool_response(
                ModelToolCall(
                    id="call_failure_1",
                    name="search_files",
                    arguments={"workspace_id": 1, "keyword": "report"},
                )
            ),
            _final_response(),
        ]
    )

    try:
        with Session(engine, expire_on_commit=False) as session:
            loop = AgentLoop(
                model_client=model_client,
                tool_registry=_registry(fail),
                recorder=SqlAlchemyAgentRunRecorder(session),
            )

            result = loop.run([ModelMessage(role="user", content="查询")])

            saved_call = session.scalar(select(AgentToolCall))
            assert result.status == "completed"
            assert saved_call is not None
            assert saved_call.status == "failed"
            assert saved_call.error_code == "tool_execution_failed"
    finally:
        engine.dispose()


def test_agent_loop_records_cancellation_before_model_request(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'cancelled-trace.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)
    cancel_event = Event()
    cancel_event.set()
    model_client = FakeModelClient([_final_response()])

    try:
        with Session(engine, expire_on_commit=False) as session:
            loop = AgentLoop(
                model_client=model_client,
                tool_registry=ToolRegistry([]),
                recorder=SqlAlchemyAgentRunRecorder(session),
            )

            result = loop.run(
                [ModelMessage(role="user", content="不要执行")],
                cancel_event=cancel_event,
            )

            saved_run = session.scalar(select(AgentRun))
            assert result.status == "cancelled"
            assert saved_run is not None
            assert saved_run.status == "cancelled"
            assert saved_run.model_turns == 0
            assert session.scalar(select(AgentToolCall)) is None
            assert model_client.calls == ()
    finally:
        engine.dispose()


def test_agent_loop_records_exhausted_model_error(tmp_path: Path) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'model-error-trace.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)
    model_error = ModelClientRequestError(
        code="model_connection_error",
        message="private provider detail",
        retryable=True,
    )

    class FailingModelClient:
        def complete(self, **_: object) -> ModelResponse:
            raise model_error

    try:
        with Session(engine, expire_on_commit=False) as session:
            loop = AgentLoop(
                model_client=FailingModelClient(),
                tool_registry=ToolRegistry([]),
                recorder=SqlAlchemyAgentRunRecorder(session),
            )

            result = loop.run(
                [ModelMessage(role="user", content="查询")],
                max_model_retries=0,
            )

            saved_run = session.scalar(select(AgentRun))
            assert result.status == "failed"
            assert saved_run is not None
            assert saved_run.status == "failed"
            assert saved_run.error_code == "model_connection_error"
            assert "private provider detail" not in repr(saved_run.__dict__)
    finally:
        engine.dispose()


def test_agent_loop_does_not_execute_tool_when_requested_record_fails() -> None:
    handler_calls: list[bool] = []

    def handle(_: BaseModel) -> ToolResult:
        handler_calls.append(True)
        return ToolResult.success()

    class FailingRecorder:
        def start_run(self) -> int:
            return 1

        def start_tool_call(self, **_: object) -> int:
            raise AgentObservabilityError("Agent 可观察记录写入失败")

        def finish_tool_call(self, **_: object) -> None:
            raise AssertionError("tool must not finish without a start record")

        def finish_run(self, **_: object) -> None:
            raise AssertionError("failed recorder must remain visible")

    loop = AgentLoop(
        model_client=FakeModelClient(
            [
                _tool_response(
                    ModelToolCall(
                        id="call_fail_closed_1",
                        name="search_files",
                        arguments={"workspace_id": 1, "keyword": "report"},
                    )
                )
            ]
        ),
        tool_registry=_registry(handle),
        recorder=FailingRecorder(),
    )

    with pytest.raises(
        AgentObservabilityError,
        match="Agent 可观察记录写入失败",
    ):
        loop.run([ModelMessage(role="user", content="查询")])

    assert handler_calls == []


def test_agent_loop_does_not_call_model_tool_after_step_message_write_fails() -> None:
    handler_calls: list[bool] = []

    def handle(_: BaseModel) -> ToolResult:
        handler_calls.append(True)
        return ToolResult.success()

    class FailingLifecycleRecorder:
        def start_run(self) -> int:
            return 1

        def start_step(self, **_: object) -> int:
            return 1

        def next_message_sequence(self, **_: object) -> int:
            return 0

        def record_model_message(self, **_: object) -> None:
            raise AgentObservabilityError("Agent 可观察记录写入失败")

        def record_model_run(self, **_: object) -> int:
            raise AssertionError("assistant message must be recorded first")

    loop = AgentLoop(
        model_client=FakeModelClient(
            [
                _tool_response(
                    ModelToolCall(
                        id="call_fail_lifecycle_1",
                        name="search_files",
                        arguments={"workspace_id": 1, "keyword": "report"},
                    )
                )
            ]
        ),
        tool_registry=_registry(handle),
        recorder=FailingLifecycleRecorder(),
    )

    with pytest.raises(
        AgentObservabilityError,
        match="Agent 可观察记录写入失败",
    ):
        loop.run([ModelMessage(role="user", content="查询")])

    assert handler_calls == []
