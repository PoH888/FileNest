from fastapi.testclient import TestClient

from backend.app.main import app


def test_minimal_ui_serves_safe_read_only_request_form() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    html = "\n".join(response.text.splitlines())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="workspace-form"' in html
    assert 'id="workspace-name"' in html
    assert 'id="workspace-root-path"' in html
    assert 'id="add-workspace"' in html
    assert 'id="workspace-status"' in html
    assert 'id="workspace-actions"' in html
    assert 'id="scan-workspace"' in html
    assert 'id="index-documents"' in html
    assert 'id="job-status"' in html
    assert 'id="pending-approvals-panel"' in html
    assert 'id="pending-approval-detail"' in html
    assert 'id="pending-approve"' in html
    assert 'id="pending-reject"' in html
    assert 'id="pending-cancel"' in html
    assert 'id="pending-edit-form"' in html
    assert 'id="pending-execute"' in html
    assert 'id="pending-undo"' in html
    assert 'id="workspace-select"' in html
    assert 'id="request-text"' in html
    assert 'id="result-status"' in html
    assert 'id="result-answer"' in html
    assert 'id="timeline-panel"' in html
    assert 'id="timeline-list"' in html
    assert 'id="recent-runs-panel"' in html
    assert 'id="recent-runs-status"' in html
    assert 'id="recent-runs-list"' in html
    assert 'id="plan-form"' in html
    assert 'id="plan-status"' in html
    assert 'id="approve-plan"' in html
    assert 'id="reject-plan"' in html
    assert 'fetch("/api/v1/workspaces")' in html
    assert 'method: "POST"' in html
    assert 'workspaceForm.addEventListener("submit"' in html
    assert "root_path: rootPath" in html
    assert "/scan`" in html
    assert "/documents/index`" in html
    assert "restoreWorkspaceJobs" in html
    assert "pollJobStatus" in html
    assert "pending" in html
    assert "running" in html
    assert "completed" in html
    assert "failed" in html
    assert "cancelled" in html
    assert "error_code" in html
    assert 'fetch("/api/v1/agent-runs"' in html
    assert "/api/v1/approvals/pending?" in html
    assert "/api/v1/operation-plans/" in html
    assert "loadPendingApprovals" in html
    assert "submitPendingDecision" in html
    assert "submitPendingFileAction" in html
    assert 'submitAgentRunAction(run, "resume"' in html
    assert 'submitAgentRunAction(run, "cancel"' in html
    assert "loadRecentRuns" in html
    assert "workspace_id: workspaceId" in html
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
    assert 'encodeURIComponent(workflowId)' in html
    assert 'expected_plan_id: snapshot.planId' in html
    assert 'expected_revision: snapshot.revision' in html
    assert 'submitPlanDecision("approve")' in html
    assert 'submitPlanDecision("reject")' in html
    assert 'addEventListener("agent_run.status_changed"' in html
    assert 'addEventListener(\n        "agent_tool_call.status_changed"' in html
    assert "eventSource.close()" in html
    assert "eventSource.onerror" in html
    assert "renderRecentRuns" in html
    assert "recentRunStatusLabel" in html
    assert 'workspaceSelect.addEventListener("change"' in html
    assert "new URLSearchParams" in html
    assert "recent-run-button" in html
