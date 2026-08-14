from fastapi.testclient import TestClient

from backend.app.main import app


def test_minimal_ui_serves_safe_read_only_request_form() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    html = "\n".join(response.text.splitlines())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="workspace-select"' in html
    assert 'id="request-text"' in html
    assert 'id="result-status"' in html
    assert 'id="result-answer"' in html
    assert 'id="timeline-panel"' in html
    assert 'id="timeline-list"' in html
    assert 'id="plan-form"' in html
    assert 'id="plan-status"' in html
    assert 'id="approve-plan"' in html
    assert 'id="reject-plan"' in html
    assert 'fetch("/api/v1/workspaces")' in html
    assert 'fetch("/api/v1/agent-runs"' in html
    assert "new EventSource" in html
    assert 'fetch("/api/v1/workflows"' in html
    assert "/decisions" in html
    assert "expected_plan_id" in html
    assert 'action,\n              expected_plan_id' in html
    assert 'source_file_id' in html
    assert "textContent" in html
    assert "innerHTML" not in html


def test_minimal_ui_wires_plan_preview_and_decision_path() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    html = "\n".join(response.text.splitlines())
    assert 'planForm.addEventListener("submit"' in html
    assert 'fetch("/api/v1/workflows"' in html
    assert 'approvePlanButton.addEventListener("click"' in html
    assert 'rejectPlanButton.addEventListener("click"' in html
    assert 'encodeURIComponent(currentWorkflow.workflowId)' in html
    assert 'expected_plan_id: currentWorkflow.planId' in html
    assert 'submitPlanDecision("approve")' in html
    assert 'submitPlanDecision("reject")' in html
    assert 'addEventListener("agent_run.status_changed"' in html
    assert 'addEventListener(\n        "agent_tool_call.status_changed"' in html
    assert "eventSource.close()" in html
    assert "eventSource.onerror" in html
