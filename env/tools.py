"""The agent's four tools over an Incident Desk world: ls, read, grep, submit.

All outputs are plain, deterministic strings (same call -> same bytes), which keeps the
prompt prefix stable across policies and runs. Paths are virtual: "/" is the world root.
"""
from __future__ import annotations

import re
from pathlib import Path

READ_DEFAULT_LINES = 40
READ_MAX_LINES = 60
GREP_MAX_SHOWN = 25


class ToolError(Exception):
    pass


class IncidentDesk:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"world not found at {self.root}; run `python -m env.generate_world`")

    # ------------------------------------------------------------------ #
    def _resolve(self, path: str) -> Path:
        p = (self.root / str(path).strip().lstrip("/")).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"path outside the world: {path}")
        if not p.exists():
            parent = next((q for q in p.parents if q.exists()), self.root)
            raise ToolError(f"no such file or directory: {path} (use ls('{self._vpath(parent)}') to see what exists)")
        return p

    def _vpath(self, p: Path) -> str:
        rel = p.relative_to(self.root).as_posix()
        return "/" + rel if rel != "." else "/"

    def _lines(self, p: Path) -> list[str]:
        return p.read_text(encoding="utf-8").splitlines()

    # ------------------------------------------------------------------ #
    def ls(self, path: str = "/") -> str:
        p = self._resolve(path)
        if p.is_file():
            return f"{self._vpath(p)} (file, {len(self._lines(p))} lines)"
        rows = []
        for c in sorted(p.iterdir(), key=lambda x: x.name):
            rows.append(f"{c.name}/" if c.is_dir() else f"{c.name}  ({len(self._lines(c))} lines)")
        return f"{self._vpath(p)}:\n" + "\n".join(rows)

    def read(self, path: str, start_line: int = 1, n_lines: int = READ_DEFAULT_LINES) -> str:
        p = self._resolve(path)
        if p.is_dir():
            raise ToolError(f"{path} is a directory; use ls()")
        lines = self._lines(p)
        total = len(lines)
        start = max(1, int(start_line))
        n = max(1, min(int(n_lines), READ_MAX_LINES))
        if start > total:
            return f"== {self._vpath(p)} (empty range: file has {total} lines) =="
        end = min(total, start + n - 1)
        body = "\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(start, end + 1))
        return f"== {self._vpath(p)} (lines {start}-{end} of {total}) ==\n{body}"

    def grep(self, pattern: str, path: str) -> str:
        p = self._resolve(path)
        if p.is_dir():
            names = ", ".join(sorted(c.name for c in p.iterdir()))
            raise ToolError(f"{path} is a directory; grep takes a single file. Files here: {names}")
        try:
            rx = re.compile(pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))
        hits = [(i, l) for i, l in enumerate(self._lines(p), 1) if rx.search(l)]
        head = f"== grep {pattern!r} {self._vpath(p)}: {len(hits)} matching lines =="
        shown = "\n".join(f"{i:>5}  {l}" for i, l in hits[:GREP_MAX_SHOWN])
        more = f"\n... ({len(hits) - GREP_MAX_SHOWN} more not shown)" if len(hits) > GREP_MAX_SHOWN else ""
        return head + ("\n" + shown if hits else "") + more

    
    # ------------------------------------------------------------------ #
    def call(self, name: str, args: dict) -> str:
        """Dispatch a tool call; errors come back as text so the agent can recover."""
        try:
            if name == "ls":
                return self.ls(args.get("path", "/"))
            if name == "read":
                return self.read(args["path"], args.get("start_line", 1), args.get("n_lines", READ_DEFAULT_LINES))
            if name == "grep":
                return self.grep(args["pattern"], args["path"])
            if name == "submit":
                return f"submitted: {args.get('answer', '')}"
            return f"error: unknown tool {name!r}"
        except ToolError as e:
            return f"error: {e}"
        except (KeyError, TypeError, ValueError) as e:
            return f"error: bad arguments for {name}: {e}"


# OpenAI-compatible tool schemas. Order is fixed on purpose: reordering tools changes the
# prompt prefix and breaks the cache (that's policy P5's "shuffled tools" variant).
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "ls", "description": "List a directory (or describe a file). Paths start at '/'.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read", "description": f"Read lines of a file, with line numbers. Max {READ_MAX_LINES} lines per call.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "start_line": {"type": "integer"}, "n_lines": {"type": "integer"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": f"Regex search in ONE file. Returns the match count and the first {GREP_MAX_SHOWN} matching lines.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern", "path"]}}},
    {"type": "function", "function": {
        "name": "submit", "description": "Submit the final answer. Ends the episode.",
        "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}},
]