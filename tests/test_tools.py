import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from resource_constrained_agent.schemas import ToolName, ToolObservation
from resource_constrained_agent.tools import ToolRegistry


class FixtureHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = (
            b"<html><head><title>Fixture</title></head>"
            b"<body><main>Hello evidence.</main></body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        _ = format, args


def test_code_exec_success() -> None:
    registry = ToolRegistry(timeout_seconds=2)

    result = registry.run(ToolName.CODE_EXEC, {"code": "print(2 + 2)"})

    assert result.ok
    assert result.data["stdout"].strip() == "4"


def test_code_exec_timeout_returns_structured_error() -> None:
    registry = ToolRegistry(timeout_seconds=0.1)

    result = registry.run(ToolName.CODE_EXEC, {"code": "while True:\n    pass"})

    assert not result.ok
    assert "timed out" in result.summary


def test_code_exec_rejects_dangerous_filesystem_access() -> None:
    registry = ToolRegistry(timeout_seconds=1)

    result = registry.run(ToolName.CODE_EXEC, {"code": "print(open('secret.txt').read())"})

    assert not result.ok
    assert "sandbox policy" in result.summary


def test_tool_registry_outer_timeout_returns_quickly() -> None:
    registry = ToolRegistry(timeout_seconds=0.1)

    def slow_tool(_: dict[str, object]) -> ToolObservation:
        time.sleep(2)
        return ToolObservation(tool_name=ToolName.CODE_EXEC, ok=True, summary="too late")

    registry._tools[ToolName.CODE_EXEC] = slow_tool
    started = time.monotonic()

    result = registry.run(ToolName.CODE_EXEC, {})

    assert not result.ok
    assert "timed out" in result.summary
    assert time.monotonic() - started < 1


def test_web_search_rejects_invalid_input() -> None:
    registry = ToolRegistry(timeout_seconds=1)

    result = registry.run(ToolName.WEB_SEARCH, {"max_results": 2})

    assert not result.ok
    assert result.summary == "Invalid web_search input."


def test_tavily_without_key_returns_structured_error() -> None:
    registry = ToolRegistry(timeout_seconds=1, search_provider="tavily")

    result = registry.run(ToolName.WEB_SEARCH, {"query": "Docker docs", "max_results": 2})

    assert not result.ok
    assert "TAVILY_API_KEY" in result.summary


def test_evidence_fetcher_rejects_invalid_input() -> None:
    registry = ToolRegistry(timeout_seconds=1)

    result = registry.run(ToolName.EVIDENCE_FETCHER, {"url": "not a url"})

    assert not result.ok
    assert result.summary == "Invalid evidence_fetcher input."


def test_evidence_fetcher_extracts_local_html() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        registry = ToolRegistry(timeout_seconds=2)
        url = f"http://127.0.0.1:{server.server_port}/"

        result = registry.run(ToolName.EVIDENCE_FETCHER, {"url": url, "max_chars": 500})
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert result.ok
    assert result.data["title"] == "Fixture"
    assert "Hello evidence." in result.data["excerpt"]
