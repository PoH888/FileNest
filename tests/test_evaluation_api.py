from fastapi.testclient import TestClient
from time import monotonic, sleep

from backend.app import evaluation_api
from backend.app.main import app


def test_create_evaluation_returns_accepted_run_id() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/evaluations")

    assert response.status_code == 202
    assert response.json()["id"] >= 1


def test_create_evaluation_reports_unavailable_executor(
    monkeypatch,
) -> None:
    class RejectedExecutor:
        def submit(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("executor stopped")

    monkeypatch.setattr(
        evaluation_api,
        "_evaluation_executor",
        RejectedExecutor(),
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/evaluations")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "evaluation_unavailable"


def test_get_evaluation_status_returns_terminal_state() -> None:
    with TestClient(app) as client:
        create_response = client.post("/api/v1/evaluations")
        run_id = create_response.json()["id"]
        deadline = monotonic() + 10
        response = client.get(f"/api/v1/evaluations/{run_id}")
        while (
            response.json().get("status") in {"pending", "running"}
            and monotonic() < deadline
        ):
            sleep(0.05)
            response = client.get(f"/api/v1/evaluations/{run_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["error_code"] is None


def test_get_evaluation_status_rejects_unknown_id() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/evaluations/999999")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "evaluation_not_found"


def test_get_evaluation_results_returns_required_sections() -> None:
    with TestClient(app) as client:
        create_response = client.post("/api/v1/evaluations")
        run_id = create_response.json()["id"]
        deadline = monotonic() + 10
        status_response = client.get(f"/api/v1/evaluations/{run_id}")
        while (
            status_response.json().get("status") in {"pending", "running"}
            and monotonic() < deadline
        ):
            sleep(0.05)
            status_response = client.get(f"/api/v1/evaluations/{run_id}")
        response = client.get(f"/api/v1/evaluations/{run_id}/results")

    assert response.status_code == 200
    payload = response.json()
    assert {
        "summary",
        "individual_cases",
        "failures",
        "metrics",
        "version_info",
    } <= payload.keys()
    assert len(payload["individual_cases"]) == 6
    assert payload["failures"] == []
    assert payload["metrics"] == payload["summary"]["metrics"]
    assert payload["version_info"] == payload["summary"]["version_info"]


def test_get_evaluation_results_rejects_incomplete_run() -> None:
    run = evaluation_api._create_evaluation_run()
    try:
        with TestClient(app) as client:
            response = client.get(f"/api/v1/evaluations/{run.id}/results")
    finally:
        with evaluation_api._evaluation_runs_lock:
            evaluation_api._evaluation_runs.pop(run.id, None)

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "evaluation_results_not_ready"
