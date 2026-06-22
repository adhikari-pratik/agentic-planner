from fastapi.testclient import TestClient

from resource_constrained_agent.schemas import AgentResult
from resource_constrained_agent.web import app


def test_web_health() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_web_homepage_lists_all_assignment_tasks() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert "Assignment test cases" in html
    assert "Submitted by Pratik Adhikari" in html
    assert "Find a current source explaining Docker multi-stage builds" in html
    assert "Compute 17291 * 483" in html
    assert "If the first source cannot be fetched" in html
    assert "Find an integer that is both even and odd" in html
    assert "Read 30 different sources about agent frameworks" in html


def test_web_run_endpoint(monkeypatch) -> None:
    def fake_execute_task(task, settings, max_steps=None, verbose=False):
        _ = settings, verbose
        return AgentResult(
            task=task,
            status="solved",
            answer=f"handled with max_steps={max_steps}",
            steps=[],
            budget={
                "calls_made": 1,
                "max_calls": 10,
                "total_cost_usd": 0.001,
                "max_cost_usd": 0.2,
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
        )

    monkeypatch.setattr("resource_constrained_agent.web.execute_task", fake_execute_task)
    client = TestClient(app)

    response = client.post("/api/run", json={"task": "Compute 2 + 2", "max_steps": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "solved"
    assert payload["answer"] == "handled with max_steps=3"
    assert payload["budget"]["calls_made"] == 1


def test_web_stream_endpoint_emits_progress_and_result(monkeypatch) -> None:
    def fake_execute_task(task, settings, max_steps=None, verbose=False, on_progress=None):
        _ = settings, verbose
        if on_progress is not None:
            on_progress("llm_start", {"step": 1, "estimated_cost_usd": 0.001})
            on_progress("observation", {"step": 1, "ok": True, "summary": "done"})
        return AgentResult(
            task=task,
            status="solved",
            answer=f"streamed with max_steps={max_steps}",
            steps=[],
            budget={
                "calls_made": 1,
                "max_calls": 10,
                "total_cost_usd": 0.001,
                "max_cost_usd": 0.2,
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
        )

    monkeypatch.setattr("resource_constrained_agent.web.execute_task", fake_execute_task)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/api/run/stream",
        json={"task": "Compute 2 + 2", "max_steps": 3},
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert '"type": "progress"' in body
    assert '"event": "llm_start"' in body
    assert '"type": "result"' in body
    assert "streamed with max_steps=3" in body
