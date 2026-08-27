import json
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from backend.app.events import (
    AgentRunStatusChangedEvent,
    AgentToolCallStatusChangedEvent,
    ApprovalStatusChangedEvent,
    WorkflowStatusChangedEvent,
    build_agent_run_event_stream,
    encode_sse_event,
)
from backend.app.models import AgentRun


OCCURRED_AT = datetime(2026, 8, 31, 20, 30, tzinfo=timezone(timedelta(hours=8)))
WORKFLOW_ID = UUID("12345678-1234-5678-1234-567812345678")


def test_existing_business_statuses_encode_as_sse_events() -> None:
    events = (
        AgentRunStatusChangedEvent(
            occurred_at=OCCURRED_AT,
            run_id=7,
            status="completed",
            model_turns=2,
        ),
        AgentToolCallStatusChangedEvent(
            occurred_at=OCCURRED_AT,
            run_id=7,
            tool_call_id=11,
            sequence_no=1,
            tool_name="search_files",
            status="succeeded",
        ),
        WorkflowStatusChangedEvent(
            occurred_at=OCCURRED_AT,
            workflow_id=WORKFLOW_ID,
            status="waiting",
            revision=3,
        ),
        ApprovalStatusChangedEvent(
            occurred_at=OCCURRED_AT,
            audit_event_id=13,
            approval_request_id=5,
            workflow_id=WORKFLOW_ID,
            action="approve",
            previous_status="WAITING_APPROVAL",
            next_status="APPROVED",
        ),
    )

    encoded_events = [encode_sse_event(event) for event in events]
    payloads = []
    for event, encoded in zip(events, encoded_events, strict=True):
        lines = encoded.splitlines()
        assert lines[0] == f"event: {event.kind}"
        assert lines[1].startswith("data: ")
        assert lines[2] == ""
        assert "\nid:" not in encoded
        payloads.append(json.loads(lines[1].removeprefix("data: ")))

    assert [payload["kind"] for payload in payloads] == [
        "agent_run.status_changed",
        "agent_tool_call.status_changed",
        "workflow.status_changed",
        "approval.status_changed",
    ]
    assert {payload["occurred_at"] for payload in payloads} == {
        "2026-08-31T12:30:00Z"
    }
    assert payloads[0]["run_id"] == 7
    assert payloads[1]["tool_call_id"] == 11
    assert payloads[2]["revision"] == 3
    assert payloads[3]["next_status"] == "APPROVED"


def test_business_event_rejects_timestamp_without_timezone() -> None:
    with pytest.raises(
        ValidationError,
        match="business event timestamp must be timezone-aware",
    ):
        AgentRunStatusChangedEvent(
            occurred_at=datetime(2026, 8, 31, 12, 30),
            run_id=1,
            status="running",
        )


def test_agent_terminal_sse_only_notifies_result_readiness() -> None:
    started_at = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
    finished_at = datetime(2026, 8, 31, 12, 1, tzinfo=timezone.utc)
    agent_run = AgentRun(
        id=7,
        status="completed",
        started_at=started_at,
        finished_at=finished_at,
        model_turns=2,
        final_answer="答案不应进入 SSE",
        sources_json="[]",
    )

    events = build_agent_run_event_stream(agent_run, [])

    assert [event.data.kind for event in events] == [
        "agent_run.status_changed",
        "agent.started",
        "agent_run.status_changed",
    ]
    terminal_payload = events[-1].data.model_dump()
    assert terminal_payload["status"] == "completed"
    assert "final_answer" not in terminal_payload
    assert "sources" not in terminal_payload
