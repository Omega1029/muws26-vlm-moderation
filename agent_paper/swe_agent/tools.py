"""tools.py — agent tools, modelled on the competition's tool surface and caps.

Caps (read_file 150 lines / 10k chars, run_command 300 s) follow the competition
summary so behaviour measured here matches what the 31B agent will see.
WARNING: commands run as local subprocesses, NOT in a sandbox. Use a throwaway
checkout or a container for anything but the bundled toy tasks.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

READ_MAX_LINES, READ_MAX_CHARS, CMD_TIMEOUT_S, OUT_MAX_CHARS = 150, 10_000, 300, 10_000

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a bash command in the repo root. Returns exit code + output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": f"Read up to {READ_MAX_LINES} lines of a file starting at start_line (1-based).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace exactly one occurrence of old_str with new_str in a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_str": {"type": "string"},
                                                        "new_str": {"type": "string"}},
                       "required": ["path", "old_str", "new_str"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a file with content.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "get_status", "description": "Report remaining step budget.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "submit_patch", "description": "Finish: submit the current working-tree diff as your fix.",
        "parameters": {"type": "object", "properties": {}}}},
]
TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}


class ToolError(Exception):
    pass


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root != p and self.root not in p.parents:
            raise ToolError(f"path escapes repo root: {rel}")
        return p

    def run_command(self, command: str) -> str:
        try:
            r = subprocess.run(command, shell=True, cwd=self.root, capture_output=True,
                               text=True, timeout=CMD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return f"[timeout after {CMD_TIMEOUT_S}s]"
        out = (r.stdout + r.stderr)[-OUT_MAX_CHARS:]
        return f"[exit {r.returncode}]\n{out}"

    def read_file(self, path: str, start_line: int = 1) -> str:
        p = self._path(path)
        if not p.is_file():
            raise ToolError(f"no such file: {path}")
        lines = p.read_text(errors="replace").splitlines()
        s = max(1, int(start_line))
        chunk = lines[s - 1:s - 1 + READ_MAX_LINES]
        body = "\n".join(f"{s + i}\t{l}" for i, l in enumerate(chunk))[:READ_MAX_CHARS]
        return f"{body}\n[lines {s}-{s + len(chunk) - 1} of {len(lines)}]"

    def edit_file(self, path: str, old_str: str, new_str: str) -> str:
        p = self._path(path)
        if not p.is_file():
            raise ToolError(f"no such file: {path}")
        text = p.read_text()
        n = text.count(old_str)
        if n != 1:
            raise ToolError(f"old_str must match exactly once, matched {n} times")
        p.write_text(text.replace(old_str, new_str, 1))
        return "ok"

    def write_file(self, path: str, content: str) -> str:
        p = self._path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return "ok"

    def diff(self) -> str:
        r = subprocess.run(["git", "diff"], cwd=self.root, capture_output=True, text=True)
        return r.stdout

    def call(self, name: str, args: dict) -> str:
        if name not in ("run_command", "read_file", "edit_file", "write_file"):
            raise ToolError(f"unknown tool: {name}")
        try:
            return getattr(self, name)(**args)
        except TypeError as e:  # wrong / missing arguments
            raise ToolError(f"bad arguments for {name}: {e}") from e
