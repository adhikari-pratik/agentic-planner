"""Timeout-protected tools available to the agent."""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from pydantic import ValidationError

from resource_constrained_agent.schemas import (
    CodeExecInput,
    EvidenceFetchInput,
    ToolName,
    ToolObservation,
    WebSearchInput,
)

DANGEROUS_MODULES = {
    "os",
    "pathlib",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
    "httpx",
    "requests",
}
DANGEROUS_CALLS = {"open", "input", "eval", "exec", "compile", "__import__"}
SEARCH_SNIPPET_CHARS = 220


class ToolRegistry:
    def __init__(
        self,
        timeout_seconds: float = 8.0,
        search_provider: str = "ddgs",
        tavily_api_key: str | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.search_provider = search_provider.lower()
        self.tavily_api_key = tavily_api_key
        self._tools: dict[ToolName, Callable[[dict[str, Any]], ToolObservation]] = {
            ToolName.WEB_SEARCH: self.web_search,
            ToolName.CODE_EXEC: self.code_exec,
            ToolName.EVIDENCE_FETCHER: self.evidence_fetcher,
        }

    def run(self, name: ToolName, tool_input: dict[str, Any]) -> ToolObservation:
        tool = self._tools[name]
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(tool, tool_input)
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            return ToolObservation(
                tool_name=name,
                ok=False,
                summary=f"{name} timed out after {self.timeout_seconds:.1f}s.",
                data={"timeout_seconds": self.timeout_seconds},
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def web_search(self, raw_input: dict[str, Any]) -> ToolObservation:
        try:
            params = WebSearchInput.model_validate(raw_input)
        except ValidationError as exc:
            return ToolObservation(
                tool_name=ToolName.WEB_SEARCH,
                ok=False,
                summary="Invalid web_search input.",
                data={"errors": exc.errors()},
            )
        if self.search_provider == "tavily":
            return self._tavily_search(params)
        try:
            with DDGS(timeout=int(self.timeout_seconds)) as ddgs:
                rows = list(ddgs.text(params.query, max_results=params.max_results))
        except (TimeoutError, httpx.HTTPError, RuntimeError, ValueError) as exc:
            return ToolObservation(
                tool_name=ToolName.WEB_SEARCH,
                ok=False,
                summary=f"Search failed: {type(exc).__name__}: {exc}",
            )
        results = [
            {
                "title": str(row.get("title", "")),
                "url": str(row.get("href") or row.get("url") or ""),
                "snippet": str(row.get("body", ""))[:SEARCH_SNIPPET_CHARS],
            }
            for row in rows
        ]
        return ToolObservation(
            tool_name=ToolName.WEB_SEARCH,
            ok=bool(results),
            summary=f"Found {len(results)} search result(s).",
            data={"query": params.query, "results": results},
        )

    def _tavily_search(self, params: WebSearchInput) -> ToolObservation:
        if not self.tavily_api_key:
            return ToolObservation(
                tool_name=ToolName.WEB_SEARCH,
                ok=False,
                summary="Tavily search selected but TAVILY_API_KEY is not set.",
            )
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.tavily_api_key,
                        "query": params.query,
                        "max_results": params.max_results,
                        "search_depth": "basic",
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.TimeoutException as exc:
            return ToolObservation(
                tool_name=ToolName.WEB_SEARCH,
                ok=False,
                summary=f"Tavily search timed out: {exc}",
            )
        except (httpx.HTTPError, ValueError) as exc:
            return ToolObservation(
                tool_name=ToolName.WEB_SEARCH,
                ok=False,
                summary=f"Tavily search failed: {type(exc).__name__}: {exc}",
            )
        rows = payload.get("results", [])
        results = [
            {
                "title": str(row.get("title", "")),
                "url": str(row.get("url", "")),
                "snippet": str(row.get("content", ""))[:SEARCH_SNIPPET_CHARS],
            }
            for row in rows
            if isinstance(row, dict)
        ]
        return ToolObservation(
            tool_name=ToolName.WEB_SEARCH,
            ok=bool(results),
            summary=f"Found {len(results)} Tavily result(s).",
            data={"query": params.query, "results": results},
        )

    def code_exec(self, raw_input: dict[str, Any]) -> ToolObservation:
        try:
            params = CodeExecInput.model_validate(raw_input)
        except ValidationError as exc:
            return ToolObservation(
                tool_name=ToolName.CODE_EXEC,
                ok=False,
                summary="Invalid code_exec input.",
                data={"errors": exc.errors()},
            )
        safety_error = validate_code_safety(params.code)
        if safety_error is not None:
            return ToolObservation(
                tool_name=ToolName.CODE_EXEC,
                ok=False,
                summary=f"Code rejected by sandbox policy: {safety_error}",
            )
        with tempfile.TemporaryDirectory(prefix="agent_code_") as tmpdir:
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", "-c", params.code],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return ToolObservation(
                    tool_name=ToolName.CODE_EXEC,
                    ok=False,
                    summary=f"Code execution timed out after {self.timeout_seconds:.1f}s.",
                    data={"stdout": exc.stdout or "", "stderr": exc.stderr or ""},
                )
            except OSError as exc:
                return ToolObservation(
                    tool_name=ToolName.CODE_EXEC,
                    ok=False,
                    summary=f"Code execution failed to start: {exc}",
                )
        return ToolObservation(
            tool_name=ToolName.CODE_EXEC,
            ok=completed.returncode == 0,
            summary=f"Process exited with code {completed.returncode}.",
            data={
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
                "returncode": completed.returncode,
            },
        )

    def evidence_fetcher(self, raw_input: dict[str, Any]) -> ToolObservation:
        try:
            params = EvidenceFetchInput.model_validate(raw_input)
        except ValidationError as exc:
            return ToolObservation(
                tool_name=ToolName.EVIDENCE_FETCHER,
                ok=False,
                summary="Invalid evidence_fetcher input.",
                data={"errors": exc.errors()},
            )
        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                follow_redirects=True,
                headers={"User-Agent": "resource-constrained-agent/0.1"},
            ) as client:
                response = client.get(str(params.url))
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            return ToolObservation(
                tool_name=ToolName.EVIDENCE_FETCHER,
                ok=False,
                summary=f"Fetch timed out: {exc}",
            )
        except httpx.HTTPError as exc:
            return ToolObservation(
                tool_name=ToolName.EVIDENCE_FETCHER,
                ok=False,
                summary=f"Fetch failed: {type(exc).__name__}: {exc}",
            )
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = " ".join(soup.get_text(" ").split())
        excerpt = text[: params.max_chars]
        return ToolObservation(
            tool_name=ToolName.EVIDENCE_FETCHER,
            ok=bool(excerpt),
            summary=f"Fetched {len(excerpt)} characters from source.",
            data={
                "url": str(response.url),
                "status_code": response.status_code,
                "title": title,
                "excerpt": excerpt,
            },
        )


def validate_code_safety(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", maxsplit=1)[0]
                if root in DANGEROUS_MODULES:
                    return f"import of '{root}' is not allowed"
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", maxsplit=1)[0]
            if root in DANGEROUS_MODULES:
                return f"import from '{root}' is not allowed"
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in DANGEROUS_CALLS:
                return f"call to '{func.id}' is not allowed"
    return None
