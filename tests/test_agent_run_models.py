from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import (
    AgentMetric,
    AgentMessage,
    AgentModelRun,
    AgentRun,
    AgentSession,
    AgentStep,
    AgentToolCall,
)


def _engine(tmp_path: Path, name: str):
    return create_engine(f"sqlite:///{(tmp_path / name).as_posix()}")


def test_agent_session_persists_metadata_and_timestamps(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, "agent-session.db")
    Base.metadata.create_all(bind=engine)

    try:
        schema = inspect(engine)
        assert [
            column["name"] for column in schema.get_columns("agent_sessions")
        ] == [
            "id",
            "workspace_id",
            "metadata_json",
            "created_at",
            "updated_at",
        ]

        with Session(engine) as session:
            agent_session = AgentSession(metadata_json='{"surface":"chat"}')
            session.add(agent_session)
            session.commit()
            session_id = agent_session.id

        with Session(engine) as session:
            saved_session = session.get(AgentSession, session_id)

            assert saved_session is not None
            assert saved_session.metadata_json == '{"surface":"chat"}'
            assert saved_session.created_at is not None
            assert saved_session.updated_at is not None
    finally:
        engine.dispose()


def test_agent_session_rejects_empty_metadata(tmp_path: Path) -> None:
    engine = _engine(tmp_path, "invalid-agent-session.db")
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            session.add(AgentSession(metadata_json=""))

            with pytest.raises(IntegrityError):
                session.commit()

            session.rollback()
    finally:
        engine.dispose()


def test_agent_step_persists_execution_fields(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, "agent-step.db")
    Base.metadata.create_all(bind=engine)
    started_at = datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc)
    completed_at = datetime(2026, 9, 2, 10, 0, 1, tzinfo=timezone.utc)

    try:
        schema = inspect(engine)
        assert [
            column["name"] for column in schema.get_columns("agent_steps")
        ] == [
            "id",
            "agent_session_id",
            "step_index",
            "step_type",
            "input",
            "output_summary",
            "status",
            "started_at",
            "completed_at",
        ]

        with Session(engine) as session:
            agent_session = AgentSession()
            session.add(agent_session)
            session.flush()
            agent_step = AgentStep(
                agent_session_id=agent_session.id,
                step_index=0,
                step_type="model_turn",
                input='{"request":"整理文件"}',
                output_summary="已生成整理建议",
                status="completed",
                started_at=started_at,
                completed_at=completed_at,
            )
            session.add(agent_step)
            session.commit()
            step_id = agent_step.id

        with Session(engine) as session:
            saved_step = session.get(AgentStep, step_id)

            assert saved_step is not None
            assert saved_step.step_index == 0
            assert saved_step.step_type == "model_turn"
            assert saved_step.input == '{"request":"整理文件"}'
            assert saved_step.output_summary == "已生成整理建议"
            assert saved_step.status == "completed"
            assert saved_step.started_at == started_at.replace(tzinfo=None)
            assert saved_step.completed_at == completed_at.replace(tzinfo=None)
    finally:
        engine.dispose()


def test_agent_step_rejects_duplicate_index_in_session(tmp_path: Path) -> None:
    engine = _engine(tmp_path, "duplicate-agent-step.db")
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            agent_session = AgentSession()
            session.add(agent_session)
            session.flush()
            session.add(
                AgentStep(
                    agent_session_id=agent_session.id,
                    step_index=0,
                    step_type="model_turn",
                    input="first",
                )
            )
            session.commit()

            session.add(
                AgentStep(
                    agent_session_id=agent_session.id,
                    step_index=0,
                    step_type="tool_call",
                    input="duplicate",
                )
            )

            with pytest.raises(IntegrityError):
                session.commit()

            session.rollback()
    finally:
        engine.dispose()


def test_agent_message_persists_four_intermediate_types(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, "agent-message.db")
    Base.metadata.create_all(bind=engine)

    try:
        schema = inspect(engine)
        assert [
            column["name"] for column in schema.get_columns("agent_messages")
        ] == [
            "id",
            "agent_step_id",
            "sequence_no",
            "message_type",
            "payload_json",
            "created_at",
        ]

        with Session(engine) as session:
            agent_session = AgentSession()
            session.add(agent_session)
            session.flush()
            agent_step = AgentStep(
                agent_session_id=agent_session.id,
                step_index=0,
                step_type="conversation",
                input="chat request",
            )
            session.add(agent_step)
            session.flush()
            session.add_all(
                [
                    AgentMessage(
                        agent_step_id=agent_step.id,
                        sequence_no=0,
                        message_type="user",
                        payload_json='{"role":"user","content":"整理文件"}',
                    ),
                    AgentMessage(
                        agent_step_id=agent_step.id,
                        sequence_no=1,
                        message_type="assistant",
                        payload_json='{"role":"assistant","content":"我来分析"}',
                    ),
                    AgentMessage(
                        agent_step_id=agent_step.id,
                        sequence_no=2,
                        message_type="tool_call",
                        payload_json=(
                            '{"id":"call_1","name":"search_files",'
                            '"arguments":{}}'
                        ),
                    ),
                    AgentMessage(
                        agent_step_id=agent_step.id,
                        sequence_no=3,
                        message_type="tool_result",
                        payload_json=(
                            '{"role":"tool","tool_call_id":"call_1",'
                            '"content":"找到 1 个文件"}'
                        ),
                    ),
                ]
            )
            session.commit()
            step_id = agent_step.id

        with Session(engine) as session:
            saved_messages = (
                session.query(AgentMessage)
                .filter(AgentMessage.agent_step_id == step_id)
                .order_by(AgentMessage.sequence_no)
                .all()
            )

            assert [message.message_type for message in saved_messages] == [
                "user",
                "assistant",
                "tool_call",
                "tool_result",
            ]
            assert saved_messages[2].payload_json == (
                '{"id":"call_1","name":"search_files","arguments":{}}'
            )
            assert saved_messages[3].created_at is not None
    finally:
        engine.dispose()


def test_agent_message_rejects_unknown_message_type(tmp_path: Path) -> None:
    engine = _engine(tmp_path, "invalid-agent-message.db")
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            agent_session = AgentSession()
            session.add(agent_session)
            session.flush()
            agent_step = AgentStep(
                agent_session_id=agent_session.id,
                step_index=0,
                step_type="conversation",
                input="chat request",
            )
            session.add(agent_step)
            session.flush()
            session.add(
                AgentMessage(
                    agent_step_id=agent_step.id,
                    sequence_no=0,
                    message_type="system",
                    payload_json="{}",
                )
            )

            with pytest.raises(IntegrityError):
                session.commit()

            session.rollback()
    finally:
        engine.dispose()


def test_agent_model_run_persists_model_prompt_tokens_and_latency(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, "agent-model-run.db")
    Base.metadata.create_all(bind=engine)

    try:
        schema = inspect(engine)
        assert [
            column["name"]
            for column in schema.get_columns("agent_model_runs")
        ] == [
            "id",
            "agent_step_id",
            "model",
            "prompt_version",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "latency_ms",
            "created_at",
        ]

        with Session(engine) as session:
            agent_session = AgentSession()
            session.add(agent_session)
            session.flush()
            agent_step = AgentStep(
                agent_session_id=agent_session.id,
                step_index=0,
                step_type="model_turn",
                input="chat request",
            )
            session.add(agent_step)
            session.flush()
            model_run = AgentModelRun(
                agent_step_id=agent_step.id,
                model="example-model",
                prompt_version="prompt-v1",
                input_tokens=1000,
                output_tokens=200,
                total_tokens=1200,
                latency_ms=250.5,
            )
            session.add(model_run)
            session.commit()
            model_run_id = model_run.id

        with Session(engine) as session:
            saved_model_run = session.get(AgentModelRun, model_run_id)

            assert saved_model_run is not None
            assert saved_model_run.model == "example-model"
            assert saved_model_run.prompt_version == "prompt-v1"
            assert saved_model_run.input_tokens == 1000
            assert saved_model_run.output_tokens == 200
            assert saved_model_run.total_tokens == 1200
            assert saved_model_run.latency_ms == 250.5
            assert saved_model_run.created_at is not None
    finally:
        engine.dispose()


def test_agent_model_run_rejects_negative_token_usage(tmp_path: Path) -> None:
    engine = _engine(tmp_path, "invalid-agent-model-run.db")
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            agent_session = AgentSession()
            session.add(agent_session)
            session.flush()
            agent_step = AgentStep(
                agent_session_id=agent_session.id,
                step_index=0,
                step_type="model_turn",
                input="chat request",
            )
            session.add(agent_step)
            session.flush()
            session.add(
                AgentModelRun(
                    agent_step_id=agent_step.id,
                    model="example-model",
                    input_tokens=-1,
                    output_tokens=2,
                    total_tokens=1,
                    latency_ms=1.0,
                )
            )

            with pytest.raises(IntegrityError):
                session.commit()

            session.rollback()
    finally:
        engine.dispose()


def test_agent_metric_persists_scoped_analysis_values(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, "agent-metrics.db")
    Base.metadata.create_all(bind=engine)

    try:
        schema = inspect(engine)
        assert [
            column["name"] for column in schema.get_columns("agent_metrics")
        ] == [
            "id",
            "agent_session_id",
            "agent_step_id",
            "agent_model_run_id",
            "metric_name",
            "value_json",
            "unit",
            "created_at",
        ]

        with Session(engine) as session:
            agent_session = AgentSession()
            session.add(agent_session)
            session.flush()
            agent_step = AgentStep(
                agent_session_id=agent_session.id,
                step_index=0,
                step_type="model_turn",
                input="chat request",
            )
            session.add(agent_step)
            session.flush()
            model_run = AgentModelRun(
                agent_step_id=agent_step.id,
                model="example-model",
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                latency_ms=25.0,
            )
            session.add(model_run)
            session.flush()
            model_run_id = model_run.id
            session.add_all(
                [
                    AgentMetric(
                        agent_session_id=agent_session.id,
                        agent_step_id=agent_step.id,
                        metric_name="step_latency",
                        value_json="25.0",
                        unit="ms",
                    ),
                    AgentMetric(
                        agent_session_id=agent_session.id,
                        agent_model_run_id=model_run.id,
                        metric_name="total_tokens",
                        value_json="120",
                        unit="tokens",
                    ),
                    AgentMetric(
                        agent_session_id=agent_session.id,
                        metric_name="trajectory_score",
                        value_json="0.9",
                    ),
                ]
            )
            session.commit()
            session_id = agent_session.id

        with Session(engine) as session:
            saved_metrics = (
                session.query(AgentMetric)
                .filter(AgentMetric.agent_session_id == session_id)
                .order_by(AgentMetric.id)
                .all()
            )

            assert [metric.metric_name for metric in saved_metrics] == [
                "step_latency",
                "total_tokens",
                "trajectory_score",
            ]
            assert saved_metrics[0].unit == "ms"
            assert saved_metrics[1].agent_model_run_id == model_run_id
            assert saved_metrics[2].value_json == "0.9"
            assert all(metric.created_at is not None for metric in saved_metrics)
    finally:
        engine.dispose()


def test_agent_metric_rejects_empty_metric_name(tmp_path: Path) -> None:
    engine = _engine(tmp_path, "invalid-agent-metrics.db")
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            agent_session = AgentSession()
            session.add(agent_session)
            session.flush()
            session.add(
                AgentMetric(
                    agent_session_id=agent_session.id,
                    metric_name="",
                    value_json="1",
                )
            )

            with pytest.raises(IntegrityError):
                session.commit()

            session.rollback()
    finally:
        engine.dispose()


def test_agent_run_and_tool_calls_persist_safe_lifecycle_metadata(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, "agent-run.db")
    Base.metadata.create_all(bind=engine)
    finished_at = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)

    try:
        with Session(engine) as session:
            agent_run = AgentRun(
                status="completed",
                finished_at=finished_at,
                model_turns=2,
            )
            session.add(agent_run)
            session.flush()
            session.add_all(
                [
                    AgentToolCall(
                        agent_run_id=agent_run.id,
                        sequence_no=1,
                        model_call_id="call_workspaces_1",
                        tool_name="list_workspaces",
                        status="succeeded",
                        finished_at=finished_at,
                    ),
                    AgentToolCall(
                        agent_run_id=agent_run.id,
                        sequence_no=2,
                        model_call_id="call_search_1",
                        tool_name="search_files",
                        status="rejected",
                        finished_at=finished_at,
                        error_code="invalid_arguments",
                    ),
                ]
            )
            session.commit()
            run_id = agent_run.id

        with Session(engine) as session:
            saved_run = session.get(AgentRun, run_id)
            saved_calls = (
                session.query(AgentToolCall)
                .filter(AgentToolCall.agent_run_id == run_id)
                .order_by(AgentToolCall.sequence_no)
                .all()
            )

            assert saved_run is not None
            assert saved_run.status == "completed"
            assert saved_run.started_at is not None
            assert saved_run.finished_at is not None
            assert saved_run.model_turns == 2
            assert saved_run.error_code is None
            assert [call.tool_name for call in saved_calls] == [
                "list_workspaces",
                "search_files",
            ]
            assert [call.status for call in saved_calls] == [
                "succeeded",
                "rejected",
            ]
            assert saved_calls[1].error_code == "invalid_arguments"
    finally:
        engine.dispose()


def test_agent_run_accepts_async_lifecycle_statuses(tmp_path: Path) -> None:
    engine = _engine(tmp_path, "async-agent-run-statuses.db")
    statuses = [
        "pending",
        "running",
        "waiting_approval",
        "completed",
        "max_steps_reached",
        "timed_out",
        "cancelled",
        "failed",
    ]
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            session.add_all(AgentRun(status=status) for status in statuses)
            session.commit()

            saved_statuses = [
                agent_run.status
                for agent_run in session.query(AgentRun)
                .order_by(AgentRun.id)
                .all()
            ]

        assert saved_statuses == statuses
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    ("duplicate_sequence", "duplicate_model_call_id"),
    [(True, False), (False, True)],
)
def test_tool_call_identity_is_unique_within_agent_run(
    tmp_path: Path,
    duplicate_sequence: bool,
    duplicate_model_call_id: bool,
) -> None:
    engine = _engine(tmp_path, "tool-call-identity.db")
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            agent_run = AgentRun()
            session.add(agent_run)
            session.flush()
            session.add(
                AgentToolCall(
                    agent_run_id=agent_run.id,
                    sequence_no=1,
                    model_call_id="call_1",
                    tool_name="list_workspaces",
                )
            )
            session.commit()

            session.add(
                AgentToolCall(
                    agent_run_id=agent_run.id,
                    sequence_no=1 if duplicate_sequence else 2,
                    model_call_id=(
                        "call_1" if duplicate_model_call_id else "call_2"
                    ),
                    tool_name="search_files",
                )
            )

            with pytest.raises(IntegrityError):
                session.commit()

            session.rollback()
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "agent_run",
    [
        AgentRun(status="unknown"),
        AgentRun(model_turns=-1),
    ],
)
def test_agent_run_rejects_invalid_lifecycle_values(
    tmp_path: Path,
    agent_run: AgentRun,
) -> None:
    engine = _engine(tmp_path, "invalid-agent-run.db")
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            session.add(agent_run)

            with pytest.raises(IntegrityError):
                session.commit()

            session.rollback()
    finally:
        engine.dispose()


def test_agent_runs_store_resume_context_without_tool_payload_columns(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, "safe-observability-schema.db")
    Base.metadata.create_all(bind=engine)

    try:
        schema = inspect(engine)
        run_columns = {
            column["name"] for column in schema.get_columns("agent_runs")
        }
        tool_call_columns = {
            column["name"]
            for column in schema.get_columns("agent_tool_calls")
        }

        assert run_columns == {
            "id",
            "workspace_id",
            "request_text",
            "context_json",
            "status",
            "started_at",
            "finished_at",
            "model_turns",
            "error_code",
            "final_answer",
            "sources_json",
        }
        assert tool_call_columns == {
            "id",
            "agent_run_id",
            "sequence_no",
            "model_call_id",
            "tool_name",
            "status",
            "started_at",
            "finished_at",
            "error_code",
        }
    finally:
        engine.dispose()
