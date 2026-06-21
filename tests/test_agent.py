import json

from resource_constrained_agent.agent import ReActAgent
from resource_constrained_agent.budget import BudgetEnforcer
from resource_constrained_agent.providers import ChatMessage, LLMProviderError, ScriptedProvider
from resource_constrained_agent.schemas import LLMResponse, TokenUsage, ToolName, ToolObservation
from resource_constrained_agent.tools import ToolRegistry


class FailingProvider:
    def estimate_cost(self, messages: list[ChatMessage]) -> float:
        _ = messages
        return 0.001

    def complete(self, messages: list[ChatMessage]) -> LLMResponse:
        _ = messages
        raise LLMProviderError("provider timed out")


class ProviderFailsAfterOneResponse:
    def __init__(self, first_response: str) -> None:
        self.first_response = first_response
        self.calls = 0

    def estimate_cost(self, messages: list[ChatMessage]) -> float:
        _ = messages
        return 0.001

    def complete(self, messages: list[ChatMessage]) -> LLMResponse:
        _ = messages
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text=self.first_response,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, cost_usd=0.001),
            )
        raise LLMProviderError("provider timed out")


class FakeWebToolRegistry(ToolRegistry):
    def run(self, name: ToolName, tool_input: dict[str, object]) -> ToolObservation:
        _ = tool_input
        if name == ToolName.WEB_SEARCH:
            return ToolObservation(
                tool_name=ToolName.WEB_SEARCH,
                ok=True,
                summary="Found 1 fake result.",
                data={"results": [{"url": "https://docs.docker.com/build/building/multi-stage/"}]},
            )
        if name == ToolName.EVIDENCE_FETCHER:
            return ToolObservation(
                tool_name=ToolName.EVIDENCE_FETCHER,
                ok=True,
                summary="Fetched fake source.",
                data={"excerpt": "Docker multi-stage builds use multiple build stages."},
            )
        return super().run(name, tool_input)


def test_agent_turns_invalid_json_into_observation() -> None:
    provider = ScriptedProvider(
        [
            "not json",
            json.dumps(
                {
                    "thought": "Stop after validation.",
                    "progress_assessment": "Validation failure was observed.",
                    "is_stuck": True,
                    "new_plan": None,
                    "action": {"kind": "final_answer", "final_answer": "Stopped cleanly."},
                }
            ),
        ]
    )
    agent = ReActAgent(provider, BudgetEnforcer(), ToolRegistry(timeout_seconds=1), max_steps=2)

    result = agent.run("test")

    assert result.steps[0].observation is not None
    assert result.steps[0].observation.tool_name == "agent_validation"
    assert result.answer == "Stopped cleanly."


def test_provider_error_stops_without_traceback() -> None:
    agent = ReActAgent(FailingProvider(), BudgetEnforcer(), ToolRegistry(timeout_seconds=1))

    result = agent.run("test")

    assert result.status == "stopped"
    assert "LLM provider stopped execution" in result.answer


def test_provider_timeout_after_code_exec_returns_verified_stdout() -> None:
    code = json.dumps(
        {
            "thought": "Use deterministic computation.",
            "progress_assessment": "No computation yet.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "code_exec",
                    "tool_input": {"code": "print(17291 * 483)"},
                },
            },
        }
    )
    agent = ReActAgent(
        ProviderFailsAfterOneResponse(code),
        BudgetEnforcer(),
        ToolRegistry(timeout_seconds=1),
        max_steps=2,
    )

    result = agent.run("Calculate the product from the code.")

    assert result.status == "solved"
    assert result.answer == "8351553"
    assert result.steps[-1].action.kind == "final_answer"


def test_exact_arithmetic_finalizes_after_llm_selects_code_exec() -> None:
    code = json.dumps(
        {
            "thought": "Use deterministic computation.",
            "progress_assessment": "No computation yet.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "code_exec",
                    "tool_input": {"code": "print(17291 * 443)"},
                },
            },
        }
    )
    wrong_final = json.dumps(
        {
            "thought": "Ignore stdout.",
            "progress_assessment": "This should not be reached.",
            "is_stuck": False,
            "new_plan": None,
            "action": {"kind": "final_answer", "final_answer": "17291 * 443 = 8351553"},
        }
    )
    provider = ScriptedProvider([code, wrong_final])
    agent = ReActAgent(
        provider,
        BudgetEnforcer(),
        ToolRegistry(timeout_seconds=1),
        max_steps=2,
    )

    result = agent.run("Compute 17291 * 443 and report the result.")

    assert result.status == "solved"
    assert result.answer == "7659913"
    assert len(result.steps) == 2
    assert result.budget["calls_made"] == 1
    assert result.steps[0].thought == "Use deterministic computation."
    assert (
        result.steps[-1].thought == "Controller finalized exact arithmetic from code_exec stdout."
    )


def test_repeated_tool_call_creates_replanning_observation() -> None:
    repeated = json.dumps(
        {
            "thought": "Try search.",
            "progress_assessment": "No progress yet.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {"tool_name": "code_exec", "tool_input": {"code": "print(1)"}},
            },
        }
    )
    final = json.dumps(
        {
            "thought": "Stop.",
            "progress_assessment": "Repeated call was detected.",
            "is_stuck": True,
            "new_plan": None,
            "action": {"kind": "final_answer", "final_answer": "Detected no-progress repeat."},
        }
    )
    agent = ReActAgent(
        ScriptedProvider([repeated, repeated, final]),
        BudgetEnforcer(),
        ToolRegistry(timeout_seconds=1),
        max_steps=3,
    )

    result = agent.run("repeat")

    assert result.steps[1].observation is not None
    assert result.steps[1].observation.tool_name == "progress_monitor"


def test_failed_tool_observation_blocks_confident_final_answer() -> None:
    bad_tool = json.dumps(
        {
            "thought": "Use code execution.",
            "progress_assessment": "No calculation yet.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {"tool_name": "code_exec", "tool_input": {"a": 17291, "b": 483}},
            },
        }
    )
    unsupported_final = json.dumps(
        {
            "thought": "I will answer from memory.",
            "progress_assessment": "The tool failed, but I am guessing.",
            "is_stuck": False,
            "new_plan": None,
            "action": {"kind": "final_answer", "final_answer": "8361093"},
        }
    )
    corrected_tool = json.dumps(
        {
            "thought": "Correct the tool input schema.",
            "progress_assessment": "The prior tool call failed validation.",
            "is_stuck": False,
            "new_plan": "Use code_exec with a code string.",
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "code_exec",
                    "tool_input": {"code": "print(17291 * 483)"},
                },
            },
        }
    )
    final = json.dumps(
        {
            "thought": "The code output has the answer.",
            "progress_assessment": "The corrected computation succeeded.",
            "is_stuck": False,
            "new_plan": None,
            "action": {"kind": "final_answer", "final_answer": "8351553"},
        }
    )
    agent = ReActAgent(
        ScriptedProvider([bad_tool, unsupported_final, corrected_tool, final]),
        BudgetEnforcer(),
        ToolRegistry(timeout_seconds=1),
        max_steps=4,
    )

    result = agent.run("Compute 17291 * 483")

    assert result.answer == "8351553"
    assert result.steps[1].observation is not None
    assert "Rejected final answer" in result.steps[1].observation.summary


def test_arithmetic_final_answer_must_match_code_stdout() -> None:
    code = json.dumps(
        {
            "thought": "Use deterministic computation.",
            "progress_assessment": "No computation yet.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "code_exec",
                    "tool_input": {"code": "print(17291 * 483)"},
                },
            },
        }
    )
    wrong_final = json.dumps(
        {
            "thought": "The evidence fetcher tool was used to validate the result.",
            "progress_assessment": "The final answer is correct and valid.",
            "is_stuck": False,
            "new_plan": None,
            "action": {"kind": "final_answer", "final_answer": "17291 * 483 = 8361273"},
        }
    )
    right_final = json.dumps(
        {
            "thought": "The code stdout is the source of truth.",
            "progress_assessment": "The final answer now matches code_exec stdout.",
            "is_stuck": False,
            "new_plan": None,
            "action": {"kind": "final_answer", "final_answer": "17291 * 483 = 8351553"},
        }
    )
    agent = ReActAgent(
        ScriptedProvider([code, wrong_final, right_final]),
        BudgetEnforcer(),
        ToolRegistry(timeout_seconds=1),
        max_steps=3,
    )

    result = agent.run("Compute 17291 * 483 and report the result.")

    assert result.answer == "8351553"
    assert (
        result.steps[-1].thought == "Controller finalized exact arithmetic from code_exec stdout."
    )


def test_repairs_abbreviated_top_level_final_answer_but_still_applies_guards() -> None:
    abbreviated_wrong_final = json.dumps({"kind": "final_answer", "final_answer": "8331319"})
    code = json.dumps(
        {
            "thought": "Use deterministic computation.",
            "progress_assessment": "The prior answer was unsupported.",
            "is_stuck": False,
            "new_plan": "Run code_exec.",
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "code_exec",
                    "tool_input": {"code": "print(17291 * 483)"},
                },
            },
        }
    )
    right_final = json.dumps(
        {
            "thought": "Use stdout.",
            "progress_assessment": "The code output supports the answer.",
            "is_stuck": False,
            "new_plan": None,
            "action": {"kind": "final_answer", "final_answer": "8351553"},
        }
    )
    agent = ReActAgent(
        ScriptedProvider([abbreviated_wrong_final, code, right_final]),
        BudgetEnforcer(),
        ToolRegistry(timeout_seconds=1),
        max_steps=3,
    )

    result = agent.run("Compute 17291 * 483")

    assert result.answer == "8351553"
    assert result.steps[0].observation is not None
    assert "must match successful code_exec stdout" in result.steps[0].observation.summary


def test_repairs_json_string_tool_input_and_single_valid_pipe_tool_name() -> None:
    repaired_tool = json.dumps(
        {
            "thought": "Computation performed using code_exec",
            "progress_assessment": "Executed with code: print(17291 * 483)",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "code_exec|print",
                    "tool_input": '{"code":"print(17291 * 483)"}',
                },
            },
        }
    )
    final = json.dumps(
        {
            "thought": "Use stdout.",
            "progress_assessment": "The code output supports the answer.",
            "is_stuck": False,
            "new_plan": None,
            "action": {"kind": "final_answer", "final_answer": "8351553"},
        }
    )
    agent = ReActAgent(
        ScriptedProvider([repaired_tool, final]),
        BudgetEnforcer(),
        ToolRegistry(timeout_seconds=1),
        max_steps=2,
    )

    result = agent.run("Compute 17291 * 483")

    assert result.answer == "8351553"
    assert result.steps[0].observation is not None
    assert result.steps[0].observation.ok


def test_repairs_copied_tool_union_and_empty_search_input() -> None:
    copied_prompt_shape = json.dumps(
        {
            "thought": "Use web_search.",
            "progress_assessment": "No sources yet.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "web_search|code_exec|evidence_fetcher",
                    "tool_input": {},
                },
            },
        }
    )
    fetch = json.dumps(
        {
            "thought": "Fetch source.",
            "progress_assessment": "Search ran with repaired input.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "evidence_fetcher",
                    "tool_input": {"url": "https://docs.docker.com/build/building/multi-stage/"},
                },
            },
        }
    )
    final = json.dumps(
        {
            "thought": "Use fetched source.",
            "progress_assessment": "Evidence fetch ran after repaired search.",
            "is_stuck": False,
            "new_plan": None,
            "action": {"kind": "final_answer", "final_answer": "Search input was repaired."},
        }
    )
    agent = ReActAgent(
        ScriptedProvider([copied_prompt_shape, fetch, final]),
        BudgetEnforcer(),
        FakeWebToolRegistry(timeout_seconds=1),
        max_steps=3,
    )

    result = agent.run("Find a source about Docker multi-stage builds.")

    assert result.answer == "Search input was repaired."
    assert result.steps[0].observation is not None
    assert result.steps[0].observation.ok
    assert result.steps[0].action.tool_call is not None
    assert result.steps[0].action.tool_call.tool_input["query"] == (
        "Find a source about Docker multi-stage builds."
    )


def test_repairs_tool_name_as_action_kind_and_routes_url_fetch_to_evidence() -> None:
    malformed_url_fetch = json.dumps(
        {
            "thought": "Check URL reachability.",
            "progress_assessment": "No prior observations.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "web_search",
                "tool_call": {
                    "query": "https://example.com",
                    "max_results": 1,
                },
            },
        }
    )
    final = json.dumps(
        {
            "thought": "Use fetch result.",
            "progress_assessment": "The URL was fetched.",
            "is_stuck": False,
            "new_plan": None,
            "action": {"kind": "final_answer", "final_answer": "The page is reachable."},
        }
    )
    agent = ReActAgent(
        ScriptedProvider([malformed_url_fetch, final]),
        BudgetEnforcer(),
        FakeWebToolRegistry(timeout_seconds=1),
        max_steps=2,
    )

    result = agent.run("Fetch https://example.com and tell me whether it is reachable.")

    assert result.answer == "The page is reachable."
    assert result.steps[0].action.tool_call is not None
    assert result.steps[0].action.tool_call.tool_name == ToolName.EVIDENCE_FETCHER
    assert result.steps[0].action.tool_call.tool_input == {"url": "https://example.com"}


def test_repairs_tool_name_kind_with_top_level_url_argument() -> None:
    shorthand_fetch = json.dumps(
        {
            "thought": "Fetch the source.",
            "progress_assessment": "Search returned a source.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "evidence_fetcher",
                "url": "https://docs.python.org/3/library/argparse.html",
            },
        }
    )
    final = json.dumps(
        {
            "thought": "Use fetched source.",
            "progress_assessment": "The source was fetched.",
            "is_stuck": False,
            "new_plan": None,
            "action": {"kind": "final_answer", "final_answer": "Source fetched."},
        }
    )
    agent = ReActAgent(
        ScriptedProvider([shorthand_fetch, final]),
        BudgetEnforcer(),
        FakeWebToolRegistry(timeout_seconds=1),
        max_steps=2,
    )

    result = agent.run("Fetch a source about Python argparse.")

    assert result.answer == "Source fetched."
    assert result.steps[0].action.tool_call is not None
    assert result.steps[0].action.tool_call.tool_name == ToolName.EVIDENCE_FETCHER
    assert result.steps[0].action.tool_call.tool_input == {
        "url": "https://docs.python.org/3/library/argparse.html"
    }


def test_repairs_json_object_with_surrounding_text_and_null_progress() -> None:
    noisy_tool = """
Here is the next action:
{
  "thought": "Use code execution.",
  "progress_assessment": null,
  "is_stuck": false,
  "new_plan": null,
  "action": {
    "kind": "tool_call",
    "tool_call": {
      "tool_name": "code_exec",
      "tool_input": {"code": "print(17291 * 443)"}
    }
  }
}
"""
    agent = ReActAgent(
        ScriptedProvider([noisy_tool]),
        BudgetEnforcer(),
        ToolRegistry(timeout_seconds=1),
        max_steps=1,
    )

    result = agent.run("Compute 17291 * 443")

    assert result.status == "solved"
    assert result.answer == "7659913"
    assert result.steps[0].progress_assessment == "No explicit progress assessment was provided."


def test_repairs_structured_new_plan_when_action_input_is_empty() -> None:
    malformed_tool = json.dumps(
        {
            "thought": "Use code_exec for arithmetic computation",
            "progress_assessment": None,
            "is_stuck": False,
            "new_plan": {
                "kind": "code_exec",
                "tool_call": {
                    "tool_name": "code_exec",
                    "code": (
                        "total = sum(range(1, 11))\n"
                        "is_prime = total > 1 and all(total % i for i in range(2, total))\n"
                        "print(f'{total} prime={is_prime}')"
                    ),
                },
            },
            "action": {
                "kind": "tool_call",
                "tool_call": {"tool_name": "code_exec", "tool_input": {}},
            },
        }
    )
    agent = ReActAgent(
        ScriptedProvider([malformed_tool]),
        BudgetEnforcer(),
        ToolRegistry(timeout_seconds=1),
        max_steps=1,
    )

    result = agent.run("Add numbers from 1 to 10 and determine if the result is prime.")

    assert result.status == "stopped"
    assert result.steps[0].observation is not None
    assert result.steps[0].observation.ok
    assert result.steps[0].observation.data["stdout"].strip() == "55 prime=False"


def test_code_exec_without_stdout_is_not_progress_for_computation() -> None:
    no_stdout_tool = json.dumps(
        {
            "thought": "Define a helper but forget to print.",
            "progress_assessment": "No computation yet.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "code_exec",
                    "tool_input": {"code": "def is_prime(n):\n    return n > 1"},
                },
            },
        }
    )
    agent = ReActAgent(
        ScriptedProvider([no_stdout_tool]),
        BudgetEnforcer(),
        ToolRegistry(timeout_seconds=1),
        max_steps=1,
    )

    result = agent.run("Add numbers from 1 to 10 and determine if the result is prime.")

    assert result.steps[0].observation is not None
    assert not result.steps[0].observation.ok
    assert "printed no stdout" in result.steps[0].observation.summary


def test_non_exact_computation_final_can_summarize_code_stdout() -> None:
    code = json.dumps(
        {
            "thought": "Use code for the sum and primality check.",
            "progress_assessment": "No computation yet.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "code_exec",
                    "tool_input": {"code": "print('55 False')"},
                },
            },
        }
    )
    final = json.dumps(
        {
            "thought": "Use the code result.",
            "progress_assessment": "The code result shows 55 is not prime.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "final_answer",
                "final_answer": "The sum is 55, and 55 is not prime.",
            },
        }
    )
    agent = ReActAgent(
        ScriptedProvider([code, final]),
        BudgetEnforcer(),
        ToolRegistry(timeout_seconds=1),
        max_steps=2,
    )

    result = agent.run("Add numbers from 1 to 10 and determine if the result is prime.")

    assert result.status == "solved"
    assert result.answer == "The sum is 55, and 55 is not prime."


def test_mixed_research_and_compute_task_does_not_use_code_only_prompt() -> None:
    search = json.dumps(
        {
            "thought": "Find source first.",
            "progress_assessment": "No source yet.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "web_search",
                    "tool_input": {"query": "Python argparse official docs", "max_results": 1},
                },
            },
        }
    )
    fetch = json.dumps(
        {
            "thought": "Verify source.",
            "progress_assessment": "Search returned a candidate.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "evidence_fetcher",
                    "tool_input": {"url": "https://docs.python.org/3/library/argparse.html"},
                },
            },
        }
    )
    code = json.dumps(
        {
            "thought": "Compute remaining calls.",
            "progress_assessment": "Source is verified.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {"tool_name": "code_exec", "tool_input": {"code": "print(10 - 4)"}},
            },
        }
    )
    final = json.dumps(
        {
            "thought": "Answer with both parts.",
            "progress_assessment": "All required work is complete.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "final_answer",
                "final_answer": "Argparse source verified; 6 calls remain.",
            },
        }
    )
    agent = ReActAgent(
        ScriptedProvider([search, fetch, code, final]),
        BudgetEnforcer(),
        FakeWebToolRegistry(timeout_seconds=1),
        max_steps=4,
    )

    result = agent.run(
        "Research Python argparse, recover from a bad source URL, then compute remaining calls."
    )

    assert result.answer == "Argparse source verified; 6 calls remain."
    tool_names = [
        step.action.tool_call.tool_name for step in result.steps[:3] if step.action.tool_call
    ]
    assert tool_names == [
        ToolName.WEB_SEARCH,
        ToolName.EVIDENCE_FETCHER,
        ToolName.CODE_EXEC,
    ]


def test_mixed_research_and_compute_task_does_not_auto_finalize_with_stdout_only() -> None:
    search = json.dumps(
        {
            "thought": "Find source first.",
            "progress_assessment": "No source yet.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "web_search",
                    "tool_input": {"query": "Python argparse official docs", "max_results": 1},
                },
            },
        }
    )
    fetch = json.dumps(
        {
            "thought": "Verify source.",
            "progress_assessment": "Search returned a candidate.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "evidence_fetcher",
                    "tool_input": {"url": "https://docs.python.org/3/library/argparse.html"},
                },
            },
        }
    )
    code = json.dumps(
        {
            "thought": "Compute remaining calls.",
            "progress_assessment": "Source is verified.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {"tool_name": "code_exec", "tool_input": {"code": "print(10 - 4)"}},
            },
        }
    )
    final_answer = "The argparse source was verified, and 6 LLM calls remain."
    final_step = json.dumps(
        {
            "thought": "Answer both requirements.",
            "progress_assessment": "The source fetch and computation are complete.",
            "is_stuck": False,
            "new_plan": None,
            "action": {"kind": "final_answer", "final_answer": final_answer},
        }
    )
    agent = ReActAgent(
        ScriptedProvider([search, fetch, code, final_step]),
        BudgetEnforcer(),
        FakeWebToolRegistry(timeout_seconds=1),
        max_steps=4,
    )

    result = agent.run("Find a source about Python argparse, then compute 10 - 4 remaining calls.")

    assert result.answer == final_answer
    assert len(result.steps) == 4


def test_repairs_structured_new_plan_into_action_and_infers_tool_from_input() -> None:
    new_plan_tool = json.dumps(
        {
            "thought": "web_search failed to find relevant URLs",
            "progress_assessment": "web_search failed with ok=false",
            "is_stuck": False,
            "new_plan": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "code_exec|evidence_fetcher",
                    "tool_input": {"code": "print(17291 * 483)", "max_chars": 4000},
                },
            },
        }
    )
    final = json.dumps(
        {
            "thought": "Use stdout.",
            "progress_assessment": "The code output supports the answer.",
            "is_stuck": False,
            "new_plan": None,
            "action": {"kind": "final_answer", "final_answer": "8351553"},
        }
    )
    agent = ReActAgent(
        ScriptedProvider([new_plan_tool, final]),
        BudgetEnforcer(),
        ToolRegistry(timeout_seconds=1),
        max_steps=2,
    )

    result = agent.run("Compute 17291 * 483")

    assert result.answer == "8351553"
    assert result.steps[0].observation is not None
    assert result.steps[0].observation.ok


def test_web_answer_requires_evidence_fetch_after_search() -> None:
    search = json.dumps(
        {
            "thought": "Find candidate sources.",
            "progress_assessment": "No source candidates yet.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "web_search",
                    "tool_input": {"query": "Docker multi-stage builds", "max_results": 2},
                },
            },
        }
    )
    unsupported_final = json.dumps(
        {
            "thought": "Answer from snippets.",
            "progress_assessment": "Search returned candidates.",
            "is_stuck": False,
            "new_plan": None,
            "action": {"kind": "final_answer", "final_answer": "Multi-stage builds use stages."},
        }
    )
    fetch = json.dumps(
        {
            "thought": "Verify a source.",
            "progress_assessment": "Search snippets are not enough.",
            "is_stuck": False,
            "new_plan": "Fetch one source.",
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "evidence_fetcher",
                    "tool_input": {"url": "https://docs.docker.com/build/building/multi-stage/"},
                },
            },
        }
    )
    final = json.dumps(
        {
            "thought": "Answer from fetched evidence.",
            "progress_assessment": "A source was fetched successfully.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "final_answer",
                "final_answer": "Docker multi-stage builds use multiple build stages.",
            },
        }
    )
    agent = ReActAgent(
        ScriptedProvider([search, unsupported_final, fetch, final]),
        BudgetEnforcer(),
        FakeWebToolRegistry(timeout_seconds=4),
        max_steps=4,
    )

    result = agent.run("Find one source about Docker multi-stage builds and summarize it.")

    assert result.answer == "Docker multi-stage builds use multiple build stages."
    assert result.steps[1].observation is not None
    assert "evidence_fetcher must verify" in result.steps[1].observation.summary


def test_source_count_request_rejects_final_answer_before_enough_distinct_fetches() -> None:
    search = json.dumps(
        {
            "thought": "Find candidate sources.",
            "progress_assessment": "No source candidates yet.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "web_search",
                    "tool_input": {"query": "agent frameworks", "max_results": 3},
                },
            },
        }
    )
    fetch = json.dumps(
        {
            "thought": "Fetch one source.",
            "progress_assessment": "Search returned candidates.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "tool_call",
                "tool_call": {
                    "tool_name": "evidence_fetcher",
                    "tool_input": {"url": "https://docs.docker.com/build/building/multi-stage/"},
                },
            },
        }
    )
    premature_final = json.dumps(
        {
            "thought": "Answer now.",
            "progress_assessment": "One source was fetched.",
            "is_stuck": False,
            "new_plan": None,
            "action": {
                "kind": "final_answer",
                "final_answer": "Agent frameworks help developers build agents.",
            },
        }
    )
    agent = ReActAgent(
        ScriptedProvider([search, fetch, premature_final]),
        BudgetEnforcer(),
        FakeWebToolRegistry(timeout_seconds=1),
        max_steps=3,
    )

    result = agent.run("Read 30 different sources about agent frameworks before answering.")

    assert result.status == "stopped"
    assert result.steps[2].observation is not None
    assert "fetched 1/30" in result.steps[2].observation.summary
