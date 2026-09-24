"""agent.py — the agent loop. Every step is logged so failures can be diagnosed, not just counted.

Supports native tool calls (OpenAI `tool_calls`) and a text fallback: a JSON object
{"tool": ..., "args": {...}} in the reply, optionally inside a ```json fence. Small
models often emit the fallback form even when native tools are offered; how often
that happens is one of the measured behaviours (see diagnostics.py).
"""
from __future__ import annotations

import json
import re
import time

from .tools import TOOL_NAMES, TOOL_SCHEMAS, ToolError, Workspace

SYSTEM_PROMPT = """You are an autonomous software engineer working in a git repository.
Fix the issue described by the user. Use the tools to inspect the code, make a minimal fix,
and run the tests to verify it. When the fix is done, call submit_patch.
Call exactly one tool per turn. If native tool calling is unavailable, reply with a single JSON
object: {"tool": "<name>", "args": {...}}."""

_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```|(\{.*\})", re.S)


def parse_text_tool_call(content: str) -> tuple[str, dict] | None:
    m = _JSON_RE.search(content or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(1) or m.group(2))
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
        return obj["tool"], obj.get("args") or {}
    return None


def run_agent(llm, workspace: Workspace, problem_statement: str, max_steps: int = 30,
              system_prompt: str = SYSTEM_PROMPT) -> dict:
    """Run one episode. Returns {"patch", "submitted", "steps", "exit_reason", "wall_s"}."""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": problem_statement}]
    steps, submitted, exit_reason, t0 = [], False, "max_steps", time.perf_counter()

    for i in range(max_steps):
        try:
            reply = llm.chat(messages, tools=TOOL_SCHEMAS)
        except Exception as e:  # server error / context overflow — record, stop
            steps.append({"i": i, "kind": "llm_error", "error": repr(e)})
            exit_reason = "llm_error"
            break

        step = {"i": i, "content": reply["content"], "latency_s": reply["latency_s"],
                "usage": reply["usage"]}
        if reply["tool_calls"]:
            tc = reply["tool_calls"][0]
            step["call_format"] = "native"
            step["n_calls_in_turn"] = len(reply["tool_calls"])
            name, raw = tc["function"]["name"], tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                args = None
            messages.append({"role": "assistant", "content": reply["content"], "tool_calls": [tc]})
            reply_to = {"role": "tool", "tool_call_id": tc.get("id", f"call_{i}")}
        else:
            parsed = parse_text_tool_call(reply["content"])
            messages.append({"role": "assistant", "content": reply["content"]})
            reply_to = {"role": "user"}
            if parsed is None:
                step.update(call_format="none", tool=None, ok=False, observation="no tool call")
                steps.append(step)
                messages.append({**reply_to, "content": "No tool call found. Call exactly one tool."})
                continue
            step["call_format"] = "text"
            name, args = parsed

        step["tool"], step["args"] = name, args
        if args is None:
            obs, ok = "arguments were not valid JSON", False
        elif name == "submit_patch":
            submitted, exit_reason, obs, ok = True, "submitted", "patch submitted", True
        elif name == "get_status":
            obs, ok = f"{max_steps - i - 1} steps remaining", True
        elif name not in TOOL_NAMES:
            obs, ok = f"unknown tool {name!r}; valid: {sorted(TOOL_NAMES)}", False
        else:
            try:
                obs, ok = workspace.call(name, args), True
            except ToolError as e:
                obs, ok = f"error: {e}", False
        step.update(ok=ok, observation=obs[:2000])
        steps.append(step)
        messages.append({**reply_to, "content": obs})
        if submitted:
            break

    return {"patch": workspace.diff(), "submitted": submitted, "steps": steps,
            "exit_reason": exit_reason, "wall_s": time.perf_counter() - t0}
