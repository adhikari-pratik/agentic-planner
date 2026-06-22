"""Small FastAPI demo surface for the resource-constrained agent."""

# ruff: noqa: E501

from __future__ import annotations

import json
import os
from queue import Queue
from threading import Thread
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from resource_constrained_agent.cli import execute_task
from resource_constrained_agent.settings import Settings


class RunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=2000)
    max_steps: int | None = Field(default=None, ge=1, le=20)


app = FastAPI(title="Resource-Constrained Agent Demo")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.post("/api/run")
def run_agent(request: RunRequest) -> dict[str, Any]:
    try:
        result = execute_task(
            request.task,
            Settings(),
            max_steps=request.max_steps,
            verbose=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return result.model_dump(mode="json")


@app.post("/api/run/stream")
def run_agent_stream(request: RunRequest) -> StreamingResponse:
    events: Queue[dict[str, Any] | None] = Queue()

    def on_progress(event: str, payload: dict[str, Any]) -> None:
        events.put({"type": "progress", "event": event, "payload": payload})

    def worker() -> None:
        try:
            result = execute_task(
                request.task,
                Settings(),
                max_steps=request.max_steps,
                on_progress=on_progress,
            )
            events.put({"type": "result", "result": result.model_dump(mode="json")})
        except Exception as exc:
            events.put({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})
        finally:
            events.put(None)

    Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            event = events.get()
            if event is None:
                break
            yield json.dumps(event) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("resource_constrained_agent.web:app", host=host, port=port)


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Resource-Constrained Agent Demo</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --border: #d8dde6;
      --text: #18202a;
      --muted: #596575;
      --accent: #176b87;
      --ok: #1c7c54;
      --warn: #a16207;
      --bad: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      border-bottom: 1px solid var(--border);
      background: var(--panel);
    }
    .wrap {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
    }
    header .wrap {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 64px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    main {
      padding: 24px 0;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 420px) minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }
    label {
      display: block;
      font-size: 13px;
      font-weight: 650;
      margin-bottom: 8px;
    }
    textarea, input {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
      font-size: 14px;
      padding: 10px 12px;
    }
    textarea {
      min-height: 142px;
      resize: vertical;
      line-height: 1.45;
    }
    .row {
      display: flex;
      gap: 10px;
      align-items: end;
      margin-top: 12px;
    }
    .row > div {
      flex: 1;
    }
    button {
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font: inherit;
      font-size: 14px;
      font-weight: 700;
      height: 40px;
      padding: 0 16px;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.65;
      cursor: wait;
    }
    .examples {
      display: grid;
      gap: 8px;
      margin-top: 14px;
    }
    .examples-title {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-top: 14px;
      text-transform: uppercase;
    }
    .example {
      width: 100%;
      height: auto;
      min-height: 36px;
      text-align: left;
      color: var(--text);
      background: #f8fafc;
      border-color: var(--border);
      font-weight: 500;
    }
    .result-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }
    h2 {
      margin: 0;
      font-size: 16px;
      letter-spacing: 0;
    }
    .badge {
      border-radius: 999px;
      padding: 4px 10px;
      background: #eef2f7;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .badge.solved { color: var(--ok); background: #eaf7f0; }
    .badge.stopped { color: var(--warn); background: #fff7e6; }
    .badge.failed { color: var(--bad); background: #fff1f0; }
    .answer {
      white-space: pre-wrap;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      background: #fbfcfe;
      min-height: 78px;
      line-height: 1.5;
    }
    .budget {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin: 12px 0;
    }
    .metric {
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      background: #fff;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .metric strong {
      font-size: 14px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    footer {
      color: var(--muted);
      font-size: 12px;
      padding: 18px 0 0;
      text-align: center;
    }
    th, td {
      text-align: left;
      vertical-align: top;
      padding: 9px 8px;
      border-bottom: 1px solid var(--border);
    }
    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    @media (max-width: 860px) {
      header .wrap {
        align-items: flex-start;
        flex-direction: column;
        justify-content: center;
        padding: 12px 0;
      }
      .meta {
        white-space: normal;
      }
      .grid {
        grid-template-columns: 1fr;
      }
      .budget {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .row {
        align-items: stretch;
        flex-direction: column;
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>Resource-Constrained Agent Demo</h1>
      <div class="meta">ReAct loop &middot; 10-call budget &middot; tool-grounded answers</div>
    </div>
  </header>
  <main class="wrap">
    <div class="grid">
      <section>
        <label for="task">Task</label>
        <textarea id="task">Find a current source explaining Docker multi-stage builds and summarize it.</textarea>
        <div class="row">
          <div>
            <label for="maxSteps">Max steps</label>
            <input id="maxSteps" type="number" min="1" max="20" value="10" />
          </div>
          <button id="runBtn" type="button">Run</button>
        </div>
        <div class="examples-title">Assignment test cases</div>
        <div class="examples">
          <button class="example" type="button">Find a current source explaining Docker multi-stage builds and summarize it.</button>
          <button class="example" type="button">Compute 17291 * 483 and report the result.</button>
          <button class="example" type="button">Find an official Python source explaining argparse. If the first source cannot be fetched, recover by finding another official source and summarize what argparse is used for in one sentence.</button>
          <button class="example" type="button">Find an integer that is both even and odd.</button>
          <button class="example" type="button">Read 30 different sources about agent frameworks before answering.</button>
        </div>
      </section>
      <section>
        <div class="result-head">
          <h2>Result</h2>
          <span id="status" class="badge">idle</span>
        </div>
        <div id="answer" class="answer">Run a task to see the answer, budget, and trace.</div>
        <div id="budget" class="budget"></div>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Action</th>
              <th>Progress</th>
              <th>Observation</th>
            </tr>
          </thead>
          <tbody id="trace"></tbody>
        </table>
      </section>
    </div>
    <footer>Submitted by Pratik Adhikari</footer>
  </main>
  <script>
    const taskEl = document.querySelector("#task");
    const maxStepsEl = document.querySelector("#maxSteps");
    const runBtn = document.querySelector("#runBtn");
    const statusEl = document.querySelector("#status");
    const answerEl = document.querySelector("#answer");
    const budgetEl = document.querySelector("#budget");
    const traceEl = document.querySelector("#trace");

    document.querySelectorAll(".example").forEach((button) => {
      button.addEventListener("click", () => {
        taskEl.value = button.textContent.trim();
      });
    });

    function setStatus(status) {
      statusEl.className = "badge " + status;
      statusEl.textContent = status;
    }

    function renderBudget(budget) {
      if (!budget) {
        budgetEl.innerHTML = "";
        return;
      }
      const rows = [
        ["Calls", `${budget.calls_made}/${budget.max_calls}`],
        ["Cost", `$${Number(budget.total_cost_usd).toFixed(4)}/$${Number(budget.max_cost_usd).toFixed(2)}`],
        ["Prompt tokens", budget.prompt_tokens],
        ["Completion tokens", budget.completion_tokens],
      ];
      budgetEl.innerHTML = rows.map(([label, value]) => (
        `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`
      )).join("");
    }

    function renderTrace(steps) {
      traceEl.innerHTML = (steps || []).map((step) => {
        const action = step.action?.tool_call?.tool_name || step.action?.kind || "";
        const observation = step.observation?.summary || "final";
        return `<tr>
          <td>${step.step_number}</td>
          <td><code>${action}</code></td>
          <td>${step.progress_assessment || ""}</td>
          <td>${observation}</td>
        </tr>`;
      }).join("");
    }

    function appendTraceRow(step, action, progress, observation) {
      const row = document.createElement("tr");
      const cells = [step, action, progress, observation];
      cells.forEach((value, index) => {
        const cell = document.createElement("td");
        if (index === 1) {
          const code = document.createElement("code");
          code.textContent = value || "";
          cell.appendChild(code);
        } else {
          cell.textContent = value || "";
        }
        row.appendChild(cell);
      });
      traceEl.appendChild(row);
    }

    function renderProgressEvent(event, payload) {
      const step = payload.step || "";
      if (event === "llm_start") {
        appendTraceRow(step, "llm", "Calling LLM", `estimated cost $${payload.estimated_cost_usd}`);
      } else if (event === "llm_complete") {
        appendTraceRow(step, "llm", "LLM returned", `${payload.completion_tokens} completion tokens`);
      } else if (event === "tool_start") {
        appendTraceRow(step, payload.tool_name, "Running tool", "started");
      } else if (event === "observation") {
        appendTraceRow(step, "observation", payload.ok ? "Observation ok" : "Observation failed", payload.summary);
      } else if (event === "rejected_final") {
        appendTraceRow(step, "final_answer", "Rejected by controller", payload.summary);
      } else if (event === "final_answer") {
        appendTraceRow(step, "final_answer", "Final answer ready", payload.status || "ready");
      } else if (event === "stopped") {
        appendTraceRow(step, "stop", "Stopped", payload.reason);
      }
    }

    async function runTask() {
      const task = taskEl.value.trim();
      if (!task) return;
      runBtn.disabled = true;
      setStatus("running");
      answerEl.textContent = "Running...";
      budgetEl.innerHTML = "";
      traceEl.innerHTML = "";
      try {
        const response = await fetch("/api/run/stream", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            task,
            max_steps: Number(maxStepsEl.value || 10),
          }),
        });
        if (!response.ok) {
          const payload = await response.json();
          throw new Error(payload.detail || "Request failed");
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const {value, done} = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, {stream: true});
          const lines = buffer.split("\\n");
          buffer = lines.pop() || "";
          for (const line of lines) {
            if (!line.trim()) continue;
            const event = JSON.parse(line);
            if (event.type === "progress") {
              renderProgressEvent(event.event, event.payload);
            } else if (event.type === "result") {
              const payload = event.result;
              setStatus(payload.status);
              answerEl.textContent = payload.answer;
              renderBudget(payload.budget);
              renderTrace(payload.steps);
            } else if (event.type === "error") {
              throw new Error(event.detail || "Request failed");
            }
          }
        }
      } catch (error) {
        setStatus("failed");
        answerEl.textContent = error.message;
      } finally {
        runBtn.disabled = false;
      }
    }

    runBtn.addEventListener("click", runTask);
  </script>
</body>
</html>
"""
