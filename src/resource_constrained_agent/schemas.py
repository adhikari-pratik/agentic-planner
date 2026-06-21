"""Pydantic v2 schemas shared by the agent, tools, and providers."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TokenUsage(StrictModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMResponse(StrictModel):
    text: str
    usage: TokenUsage


class ToolName(StrEnum):
    WEB_SEARCH = "web_search"
    CODE_EXEC = "code_exec"
    EVIDENCE_FETCHER = "evidence_fetcher"


class ToolCall(StrictModel):
    tool_name: ToolName
    tool_input: dict[str, Any] = Field(default_factory=dict)


class ToolObservation(StrictModel):
    tool_name: ToolName | Literal["agent_validation", "progress_monitor"]
    ok: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class AgentAction(StrictModel):
    kind: Literal["tool_call", "final_answer"]
    tool_call: ToolCall | None = None
    final_answer: str | None = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> AgentAction:
        if self.kind == "tool_call" and self.tool_call is None:
            raise ValueError("tool_call action requires tool_call")
        if self.kind == "final_answer" and not self.final_answer:
            raise ValueError("final_answer action requires final_answer")
        return self


class StepOutput(StrictModel):
    thought: str
    progress_assessment: str
    is_stuck: bool
    new_plan: str | None = None
    action: AgentAction

    @field_validator("thought", "progress_assessment")
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field cannot be empty")
        return stripped


class AgentStep(StrictModel):
    step_number: int
    thought: str
    progress_assessment: str
    is_stuck: bool
    new_plan: str | None = None
    action: AgentAction
    observation: ToolObservation | None = None


class AgentResult(StrictModel):
    task: str
    status: Literal["solved", "stopped", "failed"]
    answer: str
    steps: list[AgentStep]
    budget: dict[str, int | float]


class WebSearchInput(StrictModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=10)


class WebSearchResult(StrictModel):
    title: str
    url: HttpUrl
    snippet: str = ""


class CodeExecInput(StrictModel):
    code: str


class EvidenceFetchInput(StrictModel):
    url: HttpUrl
    max_chars: int = Field(default=4000, ge=500, le=12000)
