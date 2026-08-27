from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.app.agent_observability import (
    AgentObservabilityError,
    SqlAlchemyAgentRunRecorder,
)
from backend.app.database import Base
from backend.app.models import AgentRun, AgentToolCall


def _timestamps() -> list[datetime]:
    return [
        datetime(2026, 8, 30, 12, minute, tzinfo=timezone.utc)
        for minute in range(4)
    ]


def test_recorder_persists_only_safe_run_and_tool_lifecycle(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'observability.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)
    timestamps = iter(_timestamps())
    raw_call_id = "call-secret-must-not-be-stored"

    try:
        with Session(engine, expire_on_commit=False) as session:
            recorder = SqlAlchemyAgentRunRecorder(
                session,
                clock=lambda: next(timestamps),
            )
            run_id = recorder.start_run()
            tool_record_id = recorder.start_tool_call(
                agent_run_id=run_id,
                sequence_no=1,
                model_call_id=raw_call_id,
                tool_name="search_files",
            )
            recorder.finish_tool_call(
                agent_run_id=run_id,
                tool_call_record_id=tool_record_id,
                status="succeeded",
                error_code=None,
            )
            recorder.finish_run(
                agent_run_id=run_id,
                status="completed",
                model_turns=2,
                error_code=None,
            )
            recorder.record_result(
                agent_run_id=run_id,
                final_answer="找到报告。",
                sources_json=(
                    '[{"workspace_id":1,"file_id":2,'
                    '"name":"report.txt","relative_path":"report.txt"}]'
                ),
            )

        with Session(engine) as session:
            saved_run = session.get(AgentRun, run_id)
            saved_call = session.get(AgentToolCall, tool_record_id)

            assert saved_run is not None
            assert saved_run.status == "completed"
            assert saved_run.model_turns == 2
            assert saved_run.finished_at is not None
            assert saved_run.final_answer == "找到报告。"
            assert saved_run.sources_json == (
                '[{"workspace_id":1,"file_id":2,'
                '"name":"report.txt","relative_path":"report.txt"}]'
            )
            assert saved_call is not None
            assert saved_call.status == "succeeded"
            assert saved_call.finished_at is not None
            assert saved_call.model_call_id == sha256(
                raw_call_id.encode("utf-8")
            ).hexdigest()
            assert raw_call_id not in saved_call.model_call_id
            assert saved_call.error_code is None
    finally:
        engine.dispose()


def test_recorder_keeps_safe_error_code_without_payload(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'observable-error.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine, expire_on_commit=False) as session:
            recorder = SqlAlchemyAgentRunRecorder(session)
            run_id = recorder.start_run()
            tool_record_id = recorder.start_tool_call(
                agent_run_id=run_id,
                sequence_no=1,
                model_call_id="call_invalid_1",
                tool_name="search_files",
            )
            recorder.finish_tool_call(
                agent_run_id=run_id,
                tool_call_record_id=tool_record_id,
                status="rejected",
                error_code="invalid_arguments",
            )
            recorder.finish_run(
                agent_run_id=run_id,
                status="failed",
                model_turns=1,
                error_code="model_request_rejected",
            )

        with Session(engine) as session:
            saved_run = session.get(AgentRun, run_id)
            saved_call = session.get(AgentToolCall, tool_record_id)

            assert saved_run is not None
            assert saved_run.error_code == "model_request_rejected"
            assert saved_call is not None
            assert saved_call.error_code == "invalid_arguments"
    finally:
        engine.dispose()


def test_recorder_rejects_malformed_sources_result(tmp_path: Path) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'malformed-sources.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine, expire_on_commit=False) as session:
            recorder = SqlAlchemyAgentRunRecorder(session)
            run_id = recorder.start_run()

            with pytest.raises(
                AgentObservabilityError,
                match="引用结果不可持久化",
            ):
                recorder.record_result(
                    agent_run_id=run_id,
                    final_answer="不应写入",
                    sources_json='{"not":"a list"}',
                )

            saved_run = session.get(AgentRun, run_id)
            assert saved_run is not None
            assert saved_run.final_answer is None
            assert saved_run.sources_json is None
    finally:
        engine.dispose()


def test_recorder_rolls_back_failed_event_without_losing_prior_events(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'observability-rollback.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine, expire_on_commit=False) as session:
            recorder = SqlAlchemyAgentRunRecorder(session)
            run_id = recorder.start_run()
            first_record_id = recorder.start_tool_call(
                agent_run_id=run_id,
                sequence_no=1,
                model_call_id="call_1",
                tool_name="list_workspaces",
            )

            with pytest.raises(
                AgentObservabilityError,
                match="Agent 可观察记录写入失败",
            ):
                recorder.start_tool_call(
                    agent_run_id=run_id,
                    sequence_no=1,
                    model_call_id="call_2",
                    tool_name="search_files",
                )

            saved_calls = list(
                session.scalars(
                    select(AgentToolCall).where(
                        AgentToolCall.agent_run_id == run_id
                    )
                )
            )
            assert [call.id for call in saved_calls] == [first_record_id]
    finally:
        engine.dispose()


def test_recorder_rejects_a_clock_without_timezone(tmp_path: Path) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'naive-clock.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as session:
            recorder = SqlAlchemyAgentRunRecorder(
                session,
                clock=lambda: datetime(2026, 8, 30, 12, 0),
            )

            with pytest.raises(
                ValueError,
                match="observability clock must return an aware datetime",
            ):
                recorder.start_run()

            assert session.scalar(select(AgentRun)) is None
    finally:
        engine.dispose()


def test_recorder_rejects_unapproved_error_text(tmp_path: Path) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'unsafe-error-code.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine, expire_on_commit=False) as session:
            recorder = SqlAlchemyAgentRunRecorder(session)
            run_id = recorder.start_run()
            tool_record_id = recorder.start_tool_call(
                agent_run_id=run_id,
                sequence_no=1,
                model_call_id="call_unsafe_error_1",
                tool_name="search_files",
            )

            with pytest.raises(
                ValueError,
                match="record contains an unsupported error code",
            ):
                recorder.finish_tool_call(
                    agent_run_id=run_id,
                    tool_call_record_id=tool_record_id,
                    status="failed",
                    error_code="secret_payload_must_not_be_stored",
                )

            saved_call = session.get(AgentToolCall, tool_record_id)
            assert saved_call is not None
            assert saved_call.status == "requested"
            assert saved_call.error_code is None
    finally:
        engine.dispose()
