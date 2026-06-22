"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from resource_constrained_agent.agent import ProgressCallback, ReActAgent
from resource_constrained_agent.budget import BudgetEnforcer
from resource_constrained_agent.harness import run_harness
from resource_constrained_agent.providers import OllamaProvider, OpenAIProvider, ScriptedProvider
from resource_constrained_agent.schemas import AgentResult
from resource_constrained_agent.settings import Settings
from resource_constrained_agent.tools import ToolRegistry

console = Console()


def build_provider(settings: Settings):
    provider = settings.agent_llm_provider.lower()
    if provider == "ollama":
        return OllamaProvider(
            model=settings.ollama_model,
            host=settings.ollama_host,
            price_per_1k_tokens=settings.local_model_price_per_1k_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
            max_completion_tokens=settings.llm_max_completion_tokens,
        )
    if provider == "openai":
        if settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required when AGENT_LLM_PROVIDER=openai")
        return OpenAIProvider(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            input_price_per_1m_tokens=settings.openai_input_price_per_1m_tokens,
            output_price_per_1m_tokens=settings.openai_output_price_per_1m_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
            max_completion_tokens=settings.llm_max_completion_tokens,
        )
    return ScriptedProvider(
        [
            json.dumps(
                {
                    "thought": "This default scripted provider is only for smoke tests.",
                    "progress_assessment": "No real LLM provider was configured.",
                    "is_stuck": True,
                    "new_plan": None,
                    "action": {
                        "kind": "final_answer",
                        "final_answer": (
                            "Configure AGENT_LLM_PROVIDER=ollama or openai for real runs."
                        ),
                    },
                }
            )
        ]
    )


def run_task(
    task: str,
    max_steps: int | None = None,
    emit_json: bool = False,
    verbose: bool = False,
) -> None:
    settings = Settings()
    result = execute_task(task, settings, max_steps=max_steps, verbose=verbose)
    if emit_json:
        console.print_json(result.model_dump_json())
    else:
        render_result(result)


def execute_task(
    task: str,
    settings: Settings,
    max_steps: int | None = None,
    verbose: bool = False,
    on_progress: ProgressCallback | None = None,
) -> AgentResult:
    progress_callback = on_progress or (render_progress if verbose else None)
    agent = ReActAgent(
        provider=build_provider(settings),
        budget=BudgetEnforcer(
            max_calls=settings.max_llm_calls,
            max_cost_usd=settings.max_task_cost_usd,
        ),
        tools=ToolRegistry(
            timeout_seconds=settings.tool_timeout_seconds,
            search_provider=settings.web_search_provider,
            tavily_api_key=settings.tavily_api_key,
        ),
        max_steps=max_steps or settings.max_agent_steps,
        on_progress=progress_callback,
    )
    return agent.run(task)


def chat(
    max_steps: int | None = None,
    emit_json: bool = False,
    verbose: bool = False,
) -> None:
    settings = Settings()
    console.print("[bold cyan]Agentic Planner[/bold cyan]")
    console.print("Type a task and press Enter. Use /exit or /quit to leave.")
    while True:
        try:
            task = console.input("\n[bold]task>[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting.")
            return
        if not task:
            continue
        if task.lower() in {"/exit", "/quit", "exit", "quit"}:
            console.print("Exiting.")
            return
        result = execute_task(task, settings, max_steps=max_steps, verbose=verbose)
        if emit_json:
            console.print_json(result.model_dump_json())
        else:
            render_result(result)


def render_progress(event: str, payload: dict[str, object]) -> None:
    step = payload.get("step", "?")
    if event == "llm_start":
        console.print(
            f"[dim]step {step}: calling LLM "
            f"(estimated cost ${payload.get('estimated_cost_usd')})...[/dim]"
        )
    elif event == "llm_complete":
        console.print(
            f"[dim]step {step}: LLM returned "
            f"{payload.get('completion_tokens')} completion tokens "
            f"(${payload.get('cost_usd')})[/dim]"
        )
    elif event == "tool_start":
        console.print(f"[dim]step {step}: running {payload.get('tool_name')}...[/dim]")
    elif event == "observation":
        status = "ok" if payload.get("ok") else "failed"
        console.print(f"[dim]step {step}: observation {status}: {payload.get('summary')}[/dim]")
    elif event == "rejected_final":
        console.print(
            f"[yellow]step {step}: rejected final answer: {payload.get('summary')}[/yellow]"
        )
    elif event == "final_answer":
        console.print(f"[green]step {step}: final answer ready[/green]")
    elif event == "stopped":
        console.print(f"[yellow]step {step}: stopped: {payload.get('reason')}[/yellow]")


def render_result(result: AgentResult) -> None:
    status_style = {
        "solved": "green",
        "stopped": "yellow",
        "failed": "red",
    }.get(result.status, "white")
    console.print(f"\n[{status_style}]Status:[/{status_style}] {result.status}")
    console.print("[bold]Answer:[/bold]")
    console.print(result.answer)

    budget = result.budget
    console.print(
        f"\n[dim]Budget: calls {budget['calls_made']}/{budget['max_calls']}, "
        f"cost ${budget['total_cost_usd']:.4f}/${budget['max_cost_usd']:.2f}, "
        f"tokens prompt={budget['prompt_tokens']} completion={budget['completion_tokens']}[/dim]"
    )
    console.print(f"[dim]Steps: {len(result.steps)}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resource-constrained ReAct agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one task")
    run_parser.add_argument("task", help="Task for the agent")
    run_parser.add_argument("--max-steps", type=int, default=None)
    run_parser.add_argument("--json", action="store_true", help="Print raw JSON result")
    run_parser.add_argument("--verbose", action="store_true", help="Print live loop progress")

    chat_parser = subparsers.add_parser("chat", help="Start an interactive task session")
    chat_parser.add_argument("--max-steps", type=int, default=None)
    chat_parser.add_argument("--json", action="store_true", help="Print raw JSON result")
    chat_parser.add_argument("--verbose", action="store_true", help="Print live loop progress")

    harness_parser = subparsers.add_parser("run-tests", help="Run deterministic five-task harness")
    harness_parser.add_argument("--output", default="test_results.md")

    args = parser.parse_args()
    if args.command == "run":
        run_task(
            args.task,
            max_steps=args.max_steps,
            emit_json=args.json,
            verbose=args.verbose,
        )
    elif args.command == "chat":
        chat(max_steps=args.max_steps, emit_json=args.json, verbose=args.verbose)
    elif args.command == "run-tests":
        run_harness(Path(args.output))
        console.print(f"Wrote {args.output}")
