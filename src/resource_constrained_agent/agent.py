"""Hand-rolled ReAct loop with budget and progress enforcement."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from resource_constrained_agent.budget import BudgetEnforcer, BudgetExceededError
from resource_constrained_agent.providers import ChatMessage, LLMProvider, LLMProviderError
from resource_constrained_agent.schemas import (
    AgentAction,
    AgentResult,
    AgentStep,
    StepOutput,
    ToolName,
    ToolObservation,
)
from resource_constrained_agent.tools import ToolRegistry

SYSTEM_PROMPT = """You are a resource-constrained ReAct agent.
You work in a ReAct loop: write a brief public thought, choose one action, wait for the
observation, then reassess progress before choosing the next action.
Do not invent observations. Observations come only from the controller after a real tool run.

Return exactly one JSON object and no surrounding text. The object must match this shape:
{
  "thought": "brief reasoning",
  "progress_assessment": "what changed since the last observation",
  "is_stuck": false,
  "new_plan": null,
  "action": {
    "kind": "tool_call",
    "tool_call": {
      "tool_name": "web_search",
      "tool_input": {"query": "search terms", "max_results": 3}
    }
  }
}
or use {"kind": "final_answer", "final_answer": "..."}.

Tool input schemas:
- web_search: {"query": "search terms", "max_results": 3}
  Use for discovery only. It returns candidate URLs and snippets. Do not treat snippets as
  verified evidence for a final answer.
- evidence_fetcher: {"url": "https://example.com/page", "max_chars": 4000}
  Use after web_search to verify one selected source URL. Web-sourced final answers must be
  grounded in fetched evidence, not search snippets.
- code_exec: {"code": "print(17291 * 483)"}
  Use for deterministic computation, arithmetic, parsing, or small code checks. Arithmetic and
  exact computation final answers must be based on successful stdout from code_exec.
  The code must be complete and must print the value needed for the final answer. Do not put
  code in new_plan.

Decision rules:
- If the task asks for current or external facts, call web_search first, then evidence_fetcher
  on the best source before answering.
- If the task asks for a specific number of sources, fetch and verify that many distinct URLs
  before giving a final answer. If the budget prevents this, report partial progress only after
  the controller stops the run.
- If the task gives a specific URL and asks to fetch it, check reachability, or validate the
  page, call evidence_fetcher directly. Do not use web_search for a provided URL.
- If the task asks to compute, calculate, multiply, divide, parse, transform, or verify code,
  call code_exec with a complete Python snippet that prints the result.
- If a tool observation has ok=false, do not answer confidently from memory. Correct the tool
  input, choose a different tool, or stop honestly with is_stuck=true.
- If the same action is not making progress, change strategy and explain the new_plan.
- When giving a final answer, follow the user's requested format, length, and sentence count.
- If the task has multiple requirements, the final answer must address each completed requirement,
  not only the last computation.
- new_plan must be a short string or null, never a JSON object.
- Use exact tool input schemas. Do not invent arguments.

Example code_exec for sum-and-prime tasks:
{"code": "total=sum(range(1,11)); prime=all(total%i for i in range(2,total)); print(total, prime)"}
"""

CODE_SYSTEM_PROMPT = """You are a resource-constrained ReAct agent.
Return exactly one JSON object and no surrounding text.

For computation, arithmetic, sorting, counting, parsing, or primality tasks, call code_exec.
The code_exec input must be a complete Python snippet that prints the final value needed.
Do not put code in new_plan. new_plan must be a short string or null.

Use this shape:
{
  "thought": "brief reason",
  "progress_assessment": "what changed",
  "is_stuck": false,
  "new_plan": null,
  "action": {
    "kind": "tool_call",
    "tool_call": {
      "tool_name": "code_exec",
      "tool_input": {"code": "print(1 + 1)"}
    }
  }
}
Final answers use:
{"kind": "final_answer", "final_answer": "..."}

Example for summing 1..10 and testing prime:
{"code": "total=sum(range(1,11)); print(f'{total} prime=False')"}
"""

ARITHMETIC_PATTERN = re.compile(r"\b\d[\d,]*\s*(?:\*|x|\+|-|/)\s*\d[\d,]*\b", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://[^\s)>\]\"']+", re.IGNORECASE)
PROMPT_SNIPPET_CHARS = 180
PROMPT_EXCERPT_CHARS = 1200
ProgressCallback = Callable[[str, dict[str, Any]], None]


class ReActAgent:
    def __init__(
        self,
        provider: LLMProvider,
        budget: BudgetEnforcer,
        tools: ToolRegistry,
        max_steps: int = 10,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.provider = provider
        self.budget = budget
        self.tools = tools
        self.max_steps = max_steps
        self.on_progress = on_progress

    def run(self, task: str) -> AgentResult:
        messages: list[ChatMessage] = [
            {"role": "system", "content": self._system_prompt_for_task(task)},
            {"role": "user", "content": task},
        ]
        steps: list[AgentStep] = []
        seen_tool_calls: set[str] = set()
        last_observation: ToolObservation | None = None

        for step_number in range(1, self.max_steps + 1):
            try:
                estimated_cost = self.provider.estimate_cost(messages)
                self._emit(
                    "llm_start",
                    {
                        "step": step_number,
                        "estimated_cost_usd": round(estimated_cost, 6),
                        "calls_made": self.budget.calls_made,
                    },
                )
                self.budget.preflight(estimated_cost_usd=estimated_cost)
                response = self.provider.complete(messages)
                self.budget.record(response.usage)
                self._emit(
                    "llm_complete",
                    {
                        "step": step_number,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "cost_usd": round(response.usage.cost_usd, 6),
                    },
                )
            except BudgetExceededError as exc:
                self._emit("stopped", {"step": step_number, "reason": str(exc)})
                return self._stopped(task, steps, f"Budget stopped execution: {exc}")
            except LLMProviderError as exc:
                fallback = self._provider_error_fallback(task, steps, exc)
                if fallback is not None:
                    self._emit(
                        "final_answer",
                        {"step": step_number, "status": "controller_fallback"},
                    )
                    return fallback
                self._emit("stopped", {"step": step_number, "reason": str(exc)})
                return self._stopped(task, steps, f"LLM provider stopped execution: {exc}")

            output, validation_error = self._parse_step_output(response.text)
            if output is not None:
                output = self._repair_contextual_tool_input(output, task)
            if output is None:
                observation = ToolObservation(
                    tool_name="agent_validation",
                    ok=False,
                    summary="LLM output failed schema validation.",
                    data={
                        "errors": validation_error.errors() if validation_error else [],
                        "raw_text": response.text[:2000],
                    },
                )
                steps.append(
                    AgentStep(
                        step_number=step_number,
                        thought="Validation failed.",
                        progress_assessment="No useful progress; output was not valid JSON.",
                        is_stuck=True,
                        new_plan="Ask the model for a valid structured action on the next turn.",
                        action=AgentAction(
                            kind="final_answer",
                            final_answer="Invalid LLM output was converted into an observation.",
                        ),
                        observation=observation,
                    )
                )
                messages.append({"role": "assistant", "content": response.text})
                messages.append(self._observation_message(observation))
                self._emit(
                    "observation",
                    {"step": step_number, "ok": observation.ok, "summary": observation.summary},
                )
                continue

            if output.action.kind == "final_answer":
                observation = self._final_answer_rejection(task, steps, last_observation, output)
                if observation is not None:
                    steps.append(
                        AgentStep(
                            step_number=step_number,
                            thought=output.thought,
                            progress_assessment=output.progress_assessment,
                            is_stuck=True,
                            new_plan=output.new_plan,
                            action=output.action,
                            observation=observation,
                        )
                    )
                    messages.append({"role": "assistant", "content": response.text})
                    messages.append(self._observation_message(observation))
                    last_observation = observation
                    self._emit(
                        "rejected_final",
                        {
                            "step": step_number,
                            "summary": observation.summary,
                        },
                    )
                    continue
                steps.append(
                    AgentStep(
                        step_number=step_number,
                        thought=output.thought,
                        progress_assessment=output.progress_assessment,
                        is_stuck=output.is_stuck,
                        new_plan=output.new_plan,
                        action=output.action,
                    )
                )
                self._emit("final_answer", {"step": step_number, "status": "ready"})
                return AgentResult(
                    task=task,
                    status="solved" if not output.is_stuck else "failed",
                    answer=output.action.final_answer or "",
                    steps=steps,
                    budget=self.budget.snapshot(),
                )

            tool_call = output.action.tool_call
            if tool_call is None:
                observation = ToolObservation(
                    tool_name="agent_validation",
                    ok=False,
                    summary="Tool action did not include a tool call.",
                )
            else:
                call_signature = tool_call.model_dump_json()
                if call_signature in seen_tool_calls:
                    observation = ToolObservation(
                        tool_name="progress_monitor",
                        ok=False,
                        summary="Repeated identical tool call; replanning required.",
                        data={"repeated_call": tool_call.model_dump(mode="json")},
                    )
                else:
                    seen_tool_calls.add(call_signature)
                    self._emit(
                        "tool_start",
                        {
                            "step": step_number,
                            "tool_name": tool_call.tool_name.value,
                        },
                    )
                    observation = self.tools.run(tool_call.tool_name, tool_call.tool_input)
                    observation = self._validate_tool_progress(
                        task,
                        tool_call.tool_name,
                        observation,
                    )

            steps.append(
                AgentStep(
                    step_number=step_number,
                    thought=output.thought,
                    progress_assessment=output.progress_assessment,
                    is_stuck=output.is_stuck,
                    new_plan=output.new_plan,
                    action=output.action,
                    observation=observation,
                )
            )
            messages.append({"role": "assistant", "content": response.text})
            messages.append(self._observation_message(observation))
            last_observation = observation
            self._emit(
                "observation",
                {"step": step_number, "ok": observation.ok, "summary": observation.summary},
            )
            arithmetic_result = self._arithmetic_tool_result(task, observation)
            if arithmetic_result is not None and not self._requires_external_support(task):
                fallback_step = AgentStep(
                    step_number=step_number + 1,
                    thought="Controller finalized exact arithmetic from code_exec stdout.",
                    progress_assessment=(
                        "The deterministic computation completed successfully; no more LLM "
                        "calls are needed."
                    ),
                    is_stuck=False,
                    new_plan=None,
                    action=AgentAction(kind="final_answer", final_answer=arithmetic_result),
                )
                steps.append(fallback_step)
                self._emit("final_answer", {"step": step_number + 1, "status": "tool_verified"})
                return AgentResult(
                    task=task,
                    status="solved",
                    answer=arithmetic_result,
                    steps=steps,
                    budget=self.budget.snapshot(),
                )
            if output.is_stuck and output.new_plan is None:
                self._emit(
                    "stopped",
                    {"step": step_number, "reason": "Agent reported stuck without new plan."},
                )
                return self._stopped(task, steps, "Agent reported it is stuck without a new plan.")
        self._emit("stopped", {"step": self.max_steps, "reason": "Reached max step limit."})
        return self._stopped(task, steps, f"Reached max step limit of {self.max_steps}.")

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.on_progress is not None:
            self.on_progress(event, payload)

    def _system_prompt_for_task(self, task: str) -> str:
        if self._requires_code_support(task) and not self._requires_external_support(task):
            return CODE_SYSTEM_PROMPT
        return SYSTEM_PROMPT

    def _requires_external_support(self, task: str) -> bool:
        lowered = task.lower()
        external_keywords = (
            "research",
            "find",
            "source",
            "current",
            "search",
            "web",
            "url",
            "fetch",
            "official",
            "documentation",
            "docs",
            "explain",
            "summarize",
        )
        return any(
            re.search(rf"\b{re.escape(keyword)}\b", lowered) for keyword in external_keywords
        )

    def _stopped(self, task: str, steps: list[AgentStep], reason: str) -> AgentResult:
        completed = [step.observation.summary for step in steps if step.observation is not None]
        answer = reason
        if completed:
            answer += "\nCompleted so far:\n- " + "\n- ".join(completed)
        return AgentResult(
            task=task,
            status="stopped",
            answer=answer,
            steps=steps,
            budget=self.budget.snapshot(),
        )

    def _provider_error_fallback(
        self,
        task: str,
        steps: list[AgentStep],
        exc: LLMProviderError,
    ) -> AgentResult | None:
        code_stdout = self._latest_successful_code_stdout(steps)
        if not code_stdout or not self._requires_code_support(task):
            return None
        answer = code_stdout.strip()
        fallback_step = AgentStep(
            step_number=len(steps) + 1,
            thought="Controller used verified code_exec output after provider failure.",
            progress_assessment=(
                f"The LLM provider failed, but code_exec stdout was available: {exc}"
            ),
            is_stuck=False,
            new_plan=None,
            action=AgentAction(kind="final_answer", final_answer=answer),
        )
        return AgentResult(
            task=task,
            status="solved",
            answer=answer,
            steps=[*steps, fallback_step],
            budget=self.budget.snapshot(),
        )

    def _validate_tool_progress(
        self,
        task: str,
        tool_name: ToolName,
        observation: ToolObservation,
    ) -> ToolObservation:
        if (
            tool_name == ToolName.CODE_EXEC
            and observation.ok
            and self._requires_code_support(task)
            and not str(observation.data.get("stdout", "")).strip()
        ):
            return ToolObservation(
                tool_name=ToolName.CODE_EXEC,
                ok=False,
                summary="Code executed but printed no stdout; code_exec must print the result.",
                data=observation.data,
            )
        return observation

    def _observation_message(self, observation: ToolObservation) -> ChatMessage:
        compact_observation = self._compact_observation_for_prompt(observation)
        return {
            "role": "user",
            "content": (
                "Observation from the controller. Use it as the source of truth and return "
                f"the next JSON action:\n{compact_observation.model_dump_json()}"
            ),
        }

    def _compact_observation_for_prompt(self, observation: ToolObservation) -> ToolObservation:
        if observation.tool_name == ToolName.WEB_SEARCH:
            raw_results = observation.data.get("results", [])
            results: list[dict[str, str]] = []
            if isinstance(raw_results, list):
                for row in raw_results[:3]:
                    if not isinstance(row, dict):
                        continue
                    results.append(
                        {
                            "title": str(row.get("title", ""))[:120],
                            "url": str(row.get("url", "")),
                            "snippet": str(row.get("snippet", ""))[:PROMPT_SNIPPET_CHARS],
                        }
                    )
            return ToolObservation(
                tool_name=observation.tool_name,
                ok=observation.ok,
                summary=observation.summary,
                data={"results": results},
            )
        if observation.tool_name == ToolName.EVIDENCE_FETCHER:
            return ToolObservation(
                tool_name=observation.tool_name,
                ok=observation.ok,
                summary=observation.summary,
                data={
                    "url": str(observation.data.get("url", "")),
                    "title": str(observation.data.get("title", ""))[:160],
                    "excerpt": str(observation.data.get("excerpt", ""))[:PROMPT_EXCERPT_CHARS],
                },
            )
        if observation.tool_name == ToolName.CODE_EXEC:
            return ToolObservation(
                tool_name=observation.tool_name,
                ok=observation.ok,
                summary=observation.summary,
                data={
                    "stdout": str(observation.data.get("stdout", ""))[-1000:],
                    "stderr": str(observation.data.get("stderr", ""))[-1000:],
                    "returncode": observation.data.get("returncode"),
                },
            )
        return observation

    def _final_answer_rejection(
        self,
        task: str,
        steps: list[AgentStep],
        last_observation: ToolObservation | None,
        output: StepOutput,
    ) -> ToolObservation | None:
        final_answer = output.action.final_answer or ""
        supported_code_stdout = self._latest_successful_code_stdout(steps)
        if supported_code_stdout and supported_code_stdout in final_answer.replace(",", ""):
            return None

        if supported_code_stdout and self._requires_code_support(task):
            if self._is_exact_arithmetic_task(task):
                return ToolObservation(
                    tool_name="progress_monitor",
                    ok=False,
                    summary=(
                        "Rejected final answer for arithmetic task; answer must match successful "
                        "code_exec stdout."
                    ),
                    data={
                        "rejected_answer": final_answer,
                        "latest_code_stdout": supported_code_stdout,
                    },
                )
            return None

        if self._requires_code_support(task):
            return ToolObservation(
                tool_name="progress_monitor",
                ok=False,
                summary=(
                    "Rejected final answer for arithmetic task; answer must match successful "
                    "code_exec stdout."
                ),
                data={
                    "rejected_answer": final_answer,
                    "latest_code_stdout": supported_code_stdout,
                },
            )

        if last_observation is not None and not last_observation.ok and not output.is_stuck:
            return ToolObservation(
                tool_name="progress_monitor",
                ok=False,
                summary="Rejected final answer after failed tool observation; replanning required.",
                data={"rejected_answer": final_answer},
            )
        required_source_count = self._required_source_count(task)
        fetched_source_count = self._successful_evidence_source_count(steps)
        if required_source_count is not None and fetched_source_count < required_source_count:
            return ToolObservation(
                tool_name="progress_monitor",
                ok=False,
                summary=(
                    "Rejected final answer before satisfying requested source count: "
                    f"fetched {fetched_source_count}/{required_source_count} distinct source(s)."
                ),
                data={
                    "rejected_answer": final_answer,
                    "required_sources": required_source_count,
                    "fetched_sources": fetched_source_count,
                },
            )
        if self._has_successful_search_without_evidence(steps):
            return ToolObservation(
                tool_name="progress_monitor",
                ok=False,
                summary=(
                    "Rejected final answer after web_search; evidence_fetcher must verify "
                    "a source before answering."
                ),
                data={"rejected_answer": final_answer},
            )
        return None

    def _parse_step_output(self, text: str) -> tuple[StepOutput | None, ValidationError | None]:
        try:
            return StepOutput.model_validate_json(text), None
        except ValidationError as first_error:
            repaired = self._repair_common_json_mistakes(text)
            if repaired is None:
                return None, first_error
            try:
                return StepOutput.model_validate(repaired), None
            except ValidationError as second_error:
                return None, second_error

    def _repair_contextual_tool_input(self, output: StepOutput, task: str) -> StepOutput:
        tool_call = output.action.tool_call
        if output.action.kind != "tool_call" or tool_call is None:
            return output
        if tool_call.tool_name == ToolName.WEB_SEARCH and self._is_url_fetch_task(task):
            url = self._extract_first_url(task) or str(tool_call.tool_input.get("query", ""))
            if URL_PATTERN.fullmatch(url):
                tool_call.tool_name = ToolName.EVIDENCE_FETCHER
                tool_call.tool_input = {"url": url}
        if tool_call.tool_name == ToolName.WEB_SEARCH and not tool_call.tool_input:
            tool_call.tool_input = {"query": task, "max_results": 3}
        return output

    def _repair_common_json_mistakes(self, text: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = self._extract_json_object(text)
            if payload is None:
                return None
        if not isinstance(payload, dict):
            return None

        if "kind" in payload and "action" not in payload:
            abbreviated_action: dict[str, Any] = {"kind": payload["kind"]}
            if "tool_call" in payload:
                abbreviated_action["tool_call"] = payload["tool_call"]
            if "final_answer" in payload:
                abbreviated_action["final_answer"] = payload["final_answer"]
            payload = {
                "thought": "Model returned an abbreviated action; controller normalized it.",
                "progress_assessment": "No explicit progress assessment was provided.",
                "is_stuck": False,
                "new_plan": None,
                "action": abbreviated_action,
            }

        if isinstance(payload.get("new_plan"), dict):
            candidate_action = self._normalize_action_like(payload["new_plan"])
            current_action = payload.get("action")
            if candidate_action is not None and self._action_has_empty_tool_input(current_action):
                payload["action"] = candidate_action
                payload["new_plan"] = "Controller normalized structured new_plan into action."

        if payload.get("progress_assessment") is None:
            payload["progress_assessment"] = "No explicit progress assessment was provided."
        if payload.get("thought") is None:
            payload["thought"] = "No explicit thought was provided."
        if payload.get("is_stuck") is None:
            payload["is_stuck"] = False
        if "new_plan" not in payload:
            payload["new_plan"] = None

        action = payload.get("action")
        normalized_action = self._normalize_action_like(action)
        if normalized_action is not None:
            payload["action"] = normalized_action
            action = normalized_action
        if not isinstance(action, dict) and isinstance(payload.get("new_plan"), dict):
            payload["action"] = payload["new_plan"]
            payload["new_plan"] = "Controller normalized structured new_plan into action."
            action = payload["action"]
        if not isinstance(action, dict):
            return payload
        tool_call = action.get("tool_call")
        if not isinstance(tool_call, dict):
            return payload

        tool_input = tool_call.get("tool_input")
        if isinstance(tool_input, str):
            try:
                decoded = json.loads(tool_input)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                tool_call["tool_input"] = decoded

        tool_name = tool_call.get("tool_name")
        if isinstance(tool_name, str) and "|" in tool_name:
            parts = [part.strip() for part in tool_name.split("|")]
            valid_parts = [part for part in parts if part in {name.value for name in ToolName}]
            if len(valid_parts) == 1:
                tool_call["tool_name"] = valid_parts[0]
            elif isinstance(tool_call.get("tool_input"), dict):
                inferred = self._infer_tool_name_from_input(tool_call["tool_input"])
                if inferred in valid_parts:
                    tool_call["tool_name"] = inferred
                    tool_call["tool_input"] = self._sanitize_tool_input(
                        inferred,
                        tool_call["tool_input"],
                    )
                elif ToolName.WEB_SEARCH.value in valid_parts:
                    tool_call["tool_name"] = ToolName.WEB_SEARCH.value

        return payload

    def _normalize_action_like(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        kind = value.get("kind")
        if kind == "final_answer" and value.get("final_answer"):
            return {"kind": "final_answer", "final_answer": value["final_answer"]}
        if kind == "tool_call":
            raw_tool_call = value.get("tool_call")
        elif isinstance(kind, str) and kind in {name.value for name in ToolName}:
            shorthand_args = {
                key: value[key]
                for key in ("code", "query", "url", "max_results", "max_chars")
                if key in value
            }
            nested_tool_call = value.get("tool_call")
            nested_args = nested_tool_call if isinstance(nested_tool_call, dict) else {}
            raw_tool_call = {"tool_name": kind, **shorthand_args, **nested_args}
        else:
            raw_tool_call = value.get("tool_call")
        if not isinstance(raw_tool_call, dict):
            return None

        tool_name = raw_tool_call.get("tool_name")
        if not isinstance(tool_name, str):
            return None
        tool_input = raw_tool_call.get("tool_input")
        if not isinstance(tool_input, dict):
            tool_input = {
                key: raw_tool_call[key]
                for key in ("code", "query", "url", "max_results", "max_chars")
                if key in raw_tool_call
            }
            if not tool_input:
                tool_input = raw_tool_call.get("tool_input")
        return {
            "kind": "tool_call",
            "tool_call": {
                "tool_name": tool_name,
                "tool_input": tool_input,
            },
        }

    def _action_has_empty_tool_input(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return True
        if value.get("kind") != "tool_call":
            return False
        tool_call = value.get("tool_call")
        if not isinstance(tool_call, dict):
            return True
        tool_input = tool_call.get("tool_input")
        return not isinstance(tool_input, dict) or not tool_input

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def _infer_tool_name_from_input(self, tool_input: dict[str, Any]) -> str | None:
        if "code" in tool_input:
            return ToolName.CODE_EXEC.value
        if "url" in tool_input:
            return ToolName.EVIDENCE_FETCHER.value
        if "query" in tool_input:
            return ToolName.WEB_SEARCH.value
        return None

    def _extract_first_url(self, text: str) -> str | None:
        match = URL_PATTERN.search(text)
        return match.group(0) if match else None

    def _is_url_fetch_task(self, task: str) -> bool:
        if self._extract_first_url(task) is None:
            return False
        lowered = task.lower()
        return any(
            keyword in lowered
            for keyword in ("fetch", "reachable", "valid page", "validate", "check url", "page")
        )

    def _sanitize_tool_input(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        if tool_name == ToolName.CODE_EXEC.value:
            return {"code": tool_input["code"]}
        if tool_name == ToolName.EVIDENCE_FETCHER.value:
            sanitized = {"url": tool_input["url"]}
            if "max_chars" in tool_input:
                sanitized["max_chars"] = tool_input["max_chars"]
            return sanitized
        if tool_name == ToolName.WEB_SEARCH.value:
            sanitized = {"query": tool_input["query"]}
            if "max_results" in tool_input:
                sanitized["max_results"] = tool_input["max_results"]
            return sanitized
        return tool_input

    def _requires_code_support(self, task: str) -> bool:
        lowered = task.lower()
        computation_keywords = (
            "compute",
            "calculate",
            "multiply",
            "divide",
            "product",
            "difference",
            "average",
            "add",
            "sum",
            "prime",
            "sort",
            "count",
        )
        return bool(ARITHMETIC_PATTERN.search(task)) or any(
            re.search(rf"\b{re.escape(keyword)}\b", lowered) for keyword in computation_keywords
        )

    def _is_exact_arithmetic_task(self, task: str) -> bool:
        return bool(ARITHMETIC_PATTERN.search(task))

    def _arithmetic_tool_result(
        self,
        task: str,
        observation: ToolObservation,
    ) -> str | None:
        if not self._is_exact_arithmetic_task(task):
            return None
        if observation.tool_name != ToolName.CODE_EXEC or not observation.ok:
            return None
        stdout = str(observation.data.get("stdout", "")).strip()
        return stdout or None

    def _latest_successful_code_stdout(self, steps: list[AgentStep]) -> str | None:
        for step in reversed(steps):
            observation = step.observation
            if (
                observation is not None
                and observation.tool_name == ToolName.CODE_EXEC
                and observation.ok
            ):
                stdout = str(observation.data.get("stdout", "")).strip()
                if stdout:
                    return stdout.replace(",", "")
        return None

    def _has_successful_search_without_evidence(self, steps: list[AgentStep]) -> bool:
        saw_successful_search = False
        for step in steps:
            observation = step.observation
            if observation is None or not observation.ok:
                continue
            if observation.tool_name == ToolName.EVIDENCE_FETCHER:
                return False
            if observation.tool_name == ToolName.WEB_SEARCH:
                saw_successful_search = True
        return saw_successful_search

    def _required_source_count(self, task: str) -> int | None:
        match = re.search(
            r"\b(\d{1,3})\s+(?:different\s+|separate\s+|distinct\s+)?sources?\b",
            task,
            re.IGNORECASE,
        )
        if match is None:
            return None
        return int(match.group(1))

    def _successful_evidence_source_count(self, steps: list[AgentStep]) -> int:
        urls: set[str] = set()
        for step in steps:
            observation = step.observation
            if (
                observation is None
                or not observation.ok
                or observation.tool_name != ToolName.EVIDENCE_FETCHER
            ):
                continue
            url = str(observation.data.get("url", "")).strip()
            urls.add(url or f"step:{step.step_number}")
        return len(urls)
