from fastapi.testclient import TestClient

from backend.app.main import app


def test_minimal_ui_serves_safe_read_only_request_form() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="workspace-select"' in response.text
    assert 'id="request-text"' in response.text
    assert 'id="result-status"' in response.text
    assert 'id="result-answer"' in response.text
    assert 'id="plan-form"' in response.text
    assert 'id="plan-status"' in response.text
    assert 'id="approve-plan"' in response.text
    assert 'id="reject-plan"' in response.text
    assert 'fetch("/api/v1/workspaces")' in response.text
    assert 'fetch("/api/v1/agent-runs"' in response.text
    assert 'fetch("/api/v1/workflows"' in response.text
    assert "/decisions" in response.text
    assert "expected_plan_id" in response.text
    assert 'action,\n              expected_plan_id' in response.text
    assert 'source_file_id' in response.text
    assert "textContent" in response.text
    assert "innerHTML" not in response.text


def test_minimal_ui_wires_plan_preview_and_decision_path() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    html = response.text
    assert 'planForm.addEventListener("submit"' in html
    assert 'fetch("/api/v1/workflows"' in html
    assert 'approvePlanButton.addEventListener("click"' in html
    assert 'rejectPlanButton.addEventListener("click"' in html
    assert 'encodeURIComponent(currentWorkflow.workflowId)' in html
    assert 'expected_plan_id: currentWorkflow.planId' in html
    assert 'submitPlanDecision("approve")' in html
    assert 'submitPlanDecision("reject")' in html
