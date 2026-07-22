from hashlib import sha256
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
    ModelMessage,
    ModelResponse,
    ModelToolCall,
)
from backend.app.models import AgentRun, AgentToolCall
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
) -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        ),
        finish_reason="tool_calls",
    )


def _final_response(content: str = "查询完成。") -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(role="assistant", content=content),
        finish_reason="stop",
    )


def test_agent_loop_persists_safe_trace_without_model_or_tool_payloads(
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
            persisted_rows = repr(
                session.execute(text("SELECT * FROM agent_runs")).all()
                + session.execute(
                    text("SELECT * FROM agent_tool_calls")
                ).all()
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
            for sensitive_value in [
                sensitive_prompt,
                sensitive_assistant_text,
                sensitive_keyword,
                sensitive_result,
                *raw_call_ids,
            ]:
                assert sensitive_value not in persisted_rows
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
