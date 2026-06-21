"""Deterministic evaluation harness for the five assignment tasks."""

from __future__ import annotations

import json
from pathlib import Path

from resource_constrained_agent.agent import ReActAgent
from resource_constrained_agent.budget import BudgetEnforcer
from resource_constrained_agent.providers import ScriptedProvider
from resource_constrained_agent.schemas import AgentResult, ToolName, ToolObservation
from resource_constrained_agent.tools import ToolRegistry


def step(
    thought: str,
    progress: str,
    action: dict[str, object],
    is_stuck: bool = False,
    new_plan: str | None = None,
) -> str:
    return json.dumps(
        {
            "thought": thought,
            "progress_assessment": progress,
            "is_stuck": is_stuck,
            "new_plan": new_plan,
            "action": action,
        }
    )


def final(text: str, thought: str = "I can now answer.") -> dict[str, object]:
    return {"kind": "final_answer", "final_answer": text}


def tool(name: str, tool_input: dict[str, object]) -> dict[str, object]:
    return {"kind": "tool_call", "tool_call": {"tool_name": name, "tool_input": tool_input}}


class HarnessToolRegistry(ToolRegistry):
    """Deterministic web tools for reproducible assignment traces."""

    def web_search(self, raw_input: dict[str, object]) -> ToolObservation:
        query = str(raw_input.get("query", ""))
        raw_max_results = raw_input.get("max_results", 3)
        max_results = raw_max_results if isinstance(raw_max_results, int) else 3
        results = [
            {
                "title": "Docker multi-stage builds",
                "url": "https://docs.docker.com/build/building/multi-stage/",
                "snippet": "Official Docker documentation explaining multi-stage builds.",
            },
            {
                "title": "Python argparse documentation",
                "url": "https://docs.python.org/3/library/argparse.html",
                "snippet": "Official Python documentation for command-line argument parsing.",
            },
        ][:max_results]
        return ToolObservation(
            tool_name=ToolName.WEB_SEARCH,
            ok=True,
            summary=f"Found {len(results)} deterministic search result(s).",
            data={"query": query, "results": results},
        )

    def evidence_fetcher(self, raw_input: dict[str, object]) -> ToolObservation:
        url = raw_input.get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            return ToolObservation(
                tool_name=ToolName.EVIDENCE_FETCHER,
                ok=False,
                summary="Invalid evidence_fetcher input.",
                data={"url": url},
            )
        return ToolObservation(
            tool_name=ToolName.EVIDENCE_FETCHER,
            ok=True,
            summary="Fetched deterministic source excerpt.",
            data={
                "url": url,
                "status_code": 200,
                "title": "Official technical documentation",
                "excerpt": (
                    "The source explains the requested technical topic and provides enough "
                    "evidence for a concise grounded answer."
                ),
            },
        )


def scripted_tasks() -> list[tuple[str, list[str], float]]:
    return [
        (
            "Find a current source explaining Docker multi-stage builds and summarize it.",
            [
                step(
                    "I need candidate sources first.",
                    "No sources yet.",
                    tool(
                        "web_search",
                        {"query": "Docker multi-stage builds official docs", "max_results": 3},
                    ),
                ),
                step(
                    "The search result should be verified from the source page.",
                    "Search found candidate URLs; fetch the official docs next.",
                    tool(
                        "evidence_fetcher",
                        {"url": "https://docs.docker.com/build/building/multi-stage/"},
                    ),
                ),
                step(
                    "The fetched source is enough.",
                    "The answer is grounded in a fetched page rather than a snippet.",
                    final(
                        "Docker multi-stage builds use multiple FROM stages so build tools can "
                        "stay out of the final runtime image."
                    ),
                ),
            ],
            0.001,
        ),
        (
            "Compute 17291 * 483 and report the result.",
            [
                step(
                    "This is deterministic arithmetic, so code execution is appropriate.",
                    "No computation has been run yet.",
                    tool("code_exec", {"code": "print(17291 * 483)"}),
                ),
                step(
                    "The code output contains the product.",
                    "The computation completed.",
                    final("17291 * 483 = 8351553."),
                ),
            ],
            0.001,
        ),
        (
            (
                "First try to fetch this bad source URL: not-a-url. If it fails, "
                "search for the official Python argparse documentation, fetch a working "
                "source, and summarize what argparse is used for in one sentence."
            ),
            [
                step(
                    "The task explicitly asks me to try the bad URL first.",
                    "No source has been checked yet.",
                    tool("evidence_fetcher", {"url": "not-a-url"}),
                ),
                step(
                    "The provided URL failed, so I need to search for a valid official source.",
                    "The bad source URL failed; pivot to web discovery.",
                    tool(
                        "web_search",
                        {"query": "Python argparse official docs", "max_results": 2},
                    ),
                    new_plan=(
                        "Search for the official Python documentation after the bad URL fails."
                    ),
                ),
                step(
                    "The search result should be verified from the actual page.",
                    "Search found candidate URLs; fetch the official docs next.",
                    tool(
                        "evidence_fetcher",
                        {"url": "https://docs.python.org/3/library/argparse.html"},
                    ),
                ),
                step(
                    "The fetched official source is enough for a concise answer.",
                    "The answer is grounded in the fetched Python documentation.",
                    final(
                        "argparse is Python's standard-library module for building "
                        "command-line interfaces that parse options, arguments, and subcommands."
                    ),
                ),
            ],
            0.001,
        ),
        (
            "Find an integer that is both even and odd.",
            [
                step(
                    "The request is mathematically contradictory.",
                    "No tool can produce an integer satisfying both parity definitions.",
                    final(
                        "No such integer exists; parity cannot be both even and odd.",
                        "Stop honestly.",
                    ),
                    is_stuck=True,
                )
            ],
            0.001,
        ),
        (
            "Read 30 different sources about agent frameworks before answering.",
            [
                step(
                    "The user asks for many sources, but the budget is strict.",
                    "Start with a small discovery query.",
                    tool("web_search", {"query": "agent framework comparison", "max_results": 10}),
                )
                for _ in range(12)
            ],
            0.03,
        ),
    ]


def run_harness(output_path: Path = Path("test_results.md")) -> list[AgentResult]:
    results: list[AgentResult] = []
    for task, responses, cost_per_call in scripted_tasks():
        agent = ReActAgent(
            provider=ScriptedProvider(responses, cost_per_call=cost_per_call),
            budget=BudgetEnforcer(max_calls=10, max_cost_usd=0.20),
            tools=HarnessToolRegistry(timeout_seconds=4),
            max_steps=10,
        )
        results.append(agent.run(task))
    write_results(results, output_path)
    return results


def write_results(results: list[AgentResult], output_path: Path) -> None:
    lines = ["# Test Results", ""]
    for index, result in enumerate(results, start=1):
        outcome = result.answer.replace("\n", "\n  ")
        lines.extend(
            [
                f"## Task {index}: {result.task}",
                f"- Status: `{result.status}`",
                f"- Budget: `{result.budget}`",
                f"- Outcome: {outcome}",
                "- Trace:",
            ]
        )
        for step_result in result.steps:
            action = step_result.action.kind
            observation = step_result.observation.summary if step_result.observation else "final"
            details = f"; progress={step_result.progress_assessment}"
            if step_result.new_plan:
                details += f"; new_plan={step_result.new_plan}"
            if step_result.is_stuck and step_result.action.kind == "final_answer":
                details += "; terminal_status=honest stop"
            if step_result.observation and not step_result.observation.ok:
                details += "; replanning_trigger=failed observation"
            lines.append(
                f"  - Step {step_result.step_number}: {action}; "
                f"stuck={step_result.is_stuck}; observation={observation}{details}"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
