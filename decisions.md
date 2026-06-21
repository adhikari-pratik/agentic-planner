# Decisions

- "I considered LangChain but chose a hand-rolled ReAct loop because the assignment evaluates explicit budget enforcement, tool routing, and replanning behavior."

- "I considered LangGraph but chose a small direct loop because graph orchestration would hide the state transitions the reviewer needs to inspect."

- "I considered separate reflection LLM calls but chose reflection folded into each step because the task has only 10 LLM calls and every extra reflection call consumes scarce budget."

- "I considered return-flag budget enforcement but chose exception-based enforcement because a raised `BudgetExceededError` cannot be silently ignored by a call site."

- "I considered a calculator custom tool but chose an evidence fetcher because search should discover sources while a separate fetch verifies the actual page content."

- "I considered merging `web_search` and `evidence_fetcher` into one web tool but chose separate tools because search snippets are only leads, while fetched page content is stronger evidence for grounded answers."

- "I considered letting the LLM do arithmetic directly but chose `code_exec` for deterministic computation because small models can choose the right operation and still copy the final number incorrectly."

- "I considered accepting any final answer after a tool call but chose controller-side answer guards because the assignment expects graceful behavior when the model guesses, undersatisfies source-count requests, or answers after a failed observation."

- "I considered failing immediately on malformed LLM JSON but chose narrow repair plus Pydantic validation because real models often make small serialization mistakes, and recoverable observations demonstrate replanning without accepting unsupported facts."

- "I considered forcing every unknown task into a tool call but chose honest stuck/failure states because unsupported, contradictory, or underspecified tasks should stop with partial progress instead of pretending to solve them."
