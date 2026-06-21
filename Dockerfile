FROM ghcr.io/astral-sh/uv:latest AS uv

FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY docs ./docs
COPY decisions.md .env.example ./

RUN uv sync --locked --no-dev

CMD ["/app/.venv/bin/agentic-planner", "run-tests"]
