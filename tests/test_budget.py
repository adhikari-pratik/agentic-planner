import pytest

from resource_constrained_agent.budget import BudgetEnforcer, BudgetExceededError
from resource_constrained_agent.schemas import TokenUsage


def test_preflight_stops_before_call_limit() -> None:
    budget = BudgetEnforcer(max_calls=1)
    budget.record(TokenUsage(prompt_tokens=1, completion_tokens=1, cost_usd=0.01))

    with pytest.raises(BudgetExceededError):
        budget.preflight()


def test_record_stops_before_cost_limit_is_crossed() -> None:
    budget = BudgetEnforcer(max_cost_usd=0.02)

    with pytest.raises(BudgetExceededError):
        budget.record(TokenUsage(prompt_tokens=1, completion_tokens=1, cost_usd=0.03))
