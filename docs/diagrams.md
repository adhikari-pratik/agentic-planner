# Architecture And Flow Diagrams

These diagrams are the presentation-friendly version of the system design. They are intentionally small enough to explain in an interview without hiding the core control flow.

## Final Architecture

```mermaid
flowchart TD
    CLI[CLI] --> Agent[ReActAgent]
    Web[FastAPI Browser Demo] --> Agent
    Harness[Test Harness] --> Agent

    Agent --> Budget[BudgetEnforcer]
    Agent --> Provider[LLMProvider]
    Agent --> Validator[Pydantic v2 Validation]
    Agent --> Tools[ToolRegistry]

    Provider --> Ollama[OllamaProvider<br/>default local model]
    Provider --> OpenAI[OpenAIProvider<br/>optional paid model]
    Provider --> Scripted[ScriptedProvider<br/>deterministic tests]

    Tools --> Search[web_search<br/>discover candidate URLs]
    Tools --> Code[code_exec<br/>deterministic computation]
    Tools --> Fetch[evidence_fetcher<br/>verify one URL]

    Budget --> Stop[Graceful stop<br/>partial progress report]
    Validator --> Observation[Structured observation]
    Search --> Observation
    Code --> Observation
    Fetch --> Observation
    Observation --> Agent
```

## Runtime Step Flow

```mermaid
sequenceDiagram
    participant U as User / Harness
    participant A as ReActAgent
    participant B as BudgetEnforcer
    participant L as LLMProvider
    participant V as Pydantic Validator
    participant T as ToolRegistry

    U->>A: task
    loop until final answer, budget stop, or max steps
        A->>B: preflight estimated next LLM call
        alt budget would be exceeded
            B-->>A: BudgetExceededError
            A-->>U: stopped result with partial progress
        else budget ok
            A->>L: prompt messages
            L-->>A: response text + token usage
            A->>B: record actual usage
            A->>V: validate StepOutput
            alt invalid or repairable output
                V-->>A: validation errors or repaired object
                A->>A: convert failures to observations
            else tool call
                A->>T: run selected tool with timeout
                T-->>A: structured observation
            else final answer
                A->>A: verify answer support
                A-->>U: solved / failed result
            end
        end
    end
```

## Tool Boundary

```mermaid
flowchart LR
    Task[Task] --> Need{What does the next step need?}

    Need -->|Find sources| Search[web_search]
    Need -->|Verify one source| Fetch[evidence_fetcher]
    Need -->|Compute or execute code| Code[code_exec]

    Search --> SearchObs[Candidate titles, snippets, URLs]
    Fetch --> FetchObs[Fetched page title, URL, excerpt, status]
    Code --> CodeObs[stdout, stderr, return code]

    SearchObs --> Decide[Progress assessment]
    FetchObs --> Decide
    CodeObs --> Decide
    Decide -->|enough| Final[Final answer]
    Decide -->|not enough| Need
```

## Arithmetic Guardrail Flow

```mermaid
flowchart TD
    MathTask[Exact arithmetic task] --> LLM[LLM proposes action or answer]
    LLM --> Guess{Final answer without code_exec support?}
    Guess -->|yes| Reject[Reject final answer<br/>create progress_monitor observation]
    Reject --> Replan[Replan]
    Replan --> Code[code_exec: print(expression)]
    Code --> Stdout[stdout value]
    Stdout --> ControllerFinal[Controller can finalize from verified stdout]
    Guess -->|no, uses code_exec| Code
```

## Brainstorming / Tradeoff Map

```mermaid
mindmap
  root((Design Choices))
    Loop
      Hand-rolled ReAct
        Explicit budget gate
        Easy to inspect
        Small enough for take-home
      LangGraph
        Viable in production
        More framework surface area
        Could distract from assignment mechanics
    Custom Tool
      Evidence fetcher
        Verifies actual page
        Separates discovery from evidence
        Stronger than trusting snippets
      Calculator
        Simple
        Less distinctive
      Data profiler
        Useful
        Less aligned with web-grounded agent traces
    Output
      Readable CLI
        Good demo experience
        Shows concise trace
      JSON mode
        Audit/debug output
        Machine-readable
```

## Reviewer Talking Point

The core claim is:

> The LLM proposes, but the controller enforces.

That means the model can make mistakes, but the system still has explicit checks for budget, schemas, tool failures, answer support, and graceful stopping.
