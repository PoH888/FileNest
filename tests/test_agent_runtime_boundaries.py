from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIConnectionError
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from backend.app.agent_loop import AgentLoop
from backend.app.agent_observability import SqlAlchemyAgentRunRecorder
from backend.app.database import Base
from backend.app.fake_model_client import FakeModelClient
from backend.app.model_client import (
    ModelMessage,
    ModelResponse,
    ModelToolCall,
)
from backend.app.model_settings import ModelSettings
from backend.app.models import AgentRun, AgentToolCall
from backend.app.openai_compatible_model_client import (
    OpenAICompatibleModelClient,
)
from backend.app.tool_contracts import Tool, ToolResult
from backend.app.tool_registry import ToolRegistry


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'agent-boundaries.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    with Session(engine, expire_on_commit=False) as test_session:
        yield test_session

    engine.dispose()


def _settings() -> ModelSettings:
    return ModelSettings(
        provider="openai",
        name="example-model",
        api_key="test-api-key",
    )


def _tool_response(
    *,
    call_id: str,
    arguments: dict[str, object],
) -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(
            role="assistant",
            tool_calls=(
                ModelToolCall(
                    id=call_id,
                    name="search_files",
                    arguments=arguments,
                ),
            ),
        ),
        finish_reason="tool_calls",
    )


def _final_response() -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(role="assistant", content="查询结束。"),
        finish_reason="stop",
    )


class SearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: int
    keyword: str


def _search_registry(handler) -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                name="search_files",
                description="搜索授权工作区中的文件索引",
                arguments_model=SearchArguments,
                handler=handler,
            )
        ]
    )


def test_network_error_is_retried_with_limit_and_safely_recorded(
    session: Session,
) -> None:
    private_sdk_detail = "private-sdk-network-detail"
    request = httpx.Request(
        "POST",
        "https://provider.example/v1/chat/completions",
    )

    class FailingCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_: Any) -> object:
            self.calls += 1
            raise APIConnectionError(
                message=private_sdk_detail,
                request=request,
            )

    completions = FailingCompletions()
    sdk_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    model_client = OpenAICompatibleModelClient(
        _settings(),
        sdk_client=sdk_client,
    )
    loop = AgentLoop(
        model_client=model_client,
        tool_registry=ToolRegistry([]),
        recorder=SqlAlchemyAgentRunRecorder(session),
    )

    result = loop.run(
        [ModelMessage(role="user", content="查询报告")],
        max_model_retries=2,
        retry_base_delay_seconds=0,
    )

    saved_run = session.scalar(select(AgentRun))
    persisted_rows = repr(
        session.execute(text("SELECT * FROM agent_runs")).all()
    )
    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "model_connection_error"
    assert result.error.attempts == 3
    assert completions.calls == 3
    assert saved_run is not None
    assert saved_run.status == "failed"
    assert saved_run.error_code == "model_connection_error"
    assert private_sdk_detail not in persisted_rows


def test_invalid_tool_arguments_are_rejected_without_calling_handler(
    session: Session,
) -> None:
    private_invalid_value = "private-invalid-argument-value"
    handler_calls: list[bool] = []

    def handle(_: BaseModel) -> ToolResult:
        handler_calls.append(True)
        return ToolResult.success()

    model_client = FakeModelClient(
        [
            _tool_response(
                call_id="call_invalid_arguments_1",
                arguments={
                    "workspace_id": 1,
                    "keyword": {"private": private_invalid_value},
                },
            ),
            _final_response(),
        ]
    )
    loop = AgentLoop(
        model_client=model_client,
        tool_registry=_search_registry(handle),
        recorder=SqlAlchemyAgentRunRecorder(session),
    )

    result = loop.run([ModelMessage(role="user", content="查询报告")])

    returned_tool_message = model_client.calls[1].messages[-1]
    tool_result = ToolResult.model_validate_json(returned_tool_message.content)
    saved_call = session.scalar(select(AgentToolCall))
    persisted_rows = repr(
        session.execute(text("SELECT * FROM agent_tool_calls")).all()
    )
    assert result.status == "completed"
    assert handler_calls == []
    assert tool_result.ok is False
    assert tool_result.error is not None
    assert tool_result.error.code == "invalid_arguments"
    assert saved_call is not None
    assert saved_call.status == "rejected"
    assert saved_call.error_code == "invalid_arguments"
    assert private_invalid_value not in persisted_rows


def test_looping_tool_calls_stop_at_step_budget_without_last_execution(
    session: Session,
) -> None:
    handler_calls: list[bool] = []

    def handle(_: BaseModel) -> ToolResult:
        handler_calls.append(True)
        return ToolResult.success({"count": 0})

    model_client = FakeModelClient(
        [
            _tool_response(
                call_id=f"call_loop_{index}",
                arguments={"workspace_id": 1, "keyword": "report"},
            )
            for index in range(1, 4)
        ]
    )
    loop = AgentLoop(
        model_client=model_client,
        tool_registry=_search_registry(handle),
        recorder=SqlAlchemyAgentRunRecorder(session),
    )

    result = loop.run(
        [ModelMessage(role="user", content="持续查询")],
        max_steps=3,
    )

    saved_run = session.scalar(select(AgentRun))
    saved_calls = list(
        session.scalars(
            select(AgentToolCall).order_by(AgentToolCall.sequence_no)
        )
    )
    assert result.status == "max_steps_reached"
    assert result.model_turns == 3
    assert len(model_client.calls) == 3
    assert handler_calls == [True, True]
    assert saved_run is not None
    assert saved_run.status == "max_steps_reached"
    assert saved_run.model_turns == 3
    assert [call.sequence_no for call in saved_calls] == [1, 2]
    assert [call.status for call in saved_calls] == [
        "succeeded",
        "succeeded",
    ]
