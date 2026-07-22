from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import AgentRun, AgentToolCall


def _engine(tmp_path: Path, name: str):
    return create_engine(f"sqlite:///{(tmp_path / name).as_posix()}")


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


def test_observability_tables_have_no_raw_prompt_or_payload_columns(
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
            "status",
            "started_at",
            "finished_at",
            "model_turns",
            "error_code",
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
