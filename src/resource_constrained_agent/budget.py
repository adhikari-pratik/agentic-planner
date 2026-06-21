"""Budget enforcement for LLM calls."""

from __future__ import annotations

from dataclasses import dataclass

from resource_constrained_agent.schemas import TokenUsage


class BudgetExceededError(RuntimeError):
    """Raised when the next LLM call would exceed the configured task budget."""


@dataclass
class BudgetEnforcer:
    max_calls: int = 10
    max_cost_usd: float = 0.20
    calls_made: int = 0
    total_cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def preflight(self, estimated_cost_usd: float = 0.0) -> None:
        """Fail before an LLM call if the next call cannot fit in the budget."""
        if self.calls_made + 1 > self.max_calls:
            raise BudgetExceededError(
                f"LLM call budget exceeded: next call would be "
                f"{self.calls_made + 1}/{self.max_calls}."
            )
        if self.total_cost_usd + estimated_cost_usd > self.max_cost_usd:
            raise BudgetExceededError(
                f"Cost budget exceeded: next call estimate would bring total to "
                f"${self.total_cost_usd + estimated_cost_usd:.4f}/"
                f"${self.max_cost_usd:.2f}."
            )

    def record(self, usage: TokenUsage) -> None:
        """Record an attempted LLM call after the provider returns usage."""
        next_cost = self.total_cost_usd + usage.cost_usd
        if next_cost > self.max_cost_usd:
            raise BudgetExceededError(
                f"Cost budget exceeded after provider usage: ${next_cost:.4f}/"
                f"${self.max_cost_usd:.2f}."
            )
        self.calls_made += 1
        self.total_cost_usd = next_cost
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens

    def snapshot(self) -> dict[str, int | float]:
        return {
            "calls_made": self.calls_made,
            "max_calls": self.max_calls,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "max_cost_usd": self.max_cost_usd,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
