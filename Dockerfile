FROM ghcr.io/astral-sh/uv:latest AS uv

FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}"

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY docs ./docs
COPY decisions.md .env.example ./

RUN uv sync --locked --no-dev

EXPOSE 8000

CMD ["agentic-planner", "run-tests"]
