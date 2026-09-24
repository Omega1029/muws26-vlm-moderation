"""llm.py — chat backends. Any OpenAI-compatible server (vLLM, Ollama, llama.cpp) + a scripted mock.

    vllm serve <gemma-4 model id> --enable-auto-tool-choice --tool-call-parser <see vLLM docs>
    ollama serve   # base_url=http://localhost:11434/v1

Uses urllib only, so the harness runs with zero pip installs.
"""
from __future__ import annotations

import json
import time
import urllib.request


class OpenAICompatLLM:
    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1",
                 api_key: str = "EMPTY", temperature: float = 0.0, max_tokens: int = 2048,
                 native_tools: bool = True, timeout: float = 600.0):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key, self.temperature, self.max_tokens = api_key, temperature, max_tokens
        self.native_tools, self.timeout = native_tools, timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Return {"content": str, "tool_calls": [...], "usage": {...}, "latency_s": float}."""
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        if tools and self.native_tools:
            body["tools"], body["tool_choice"] = tools, "auto"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.loads(r.read())
        msg = resp["choices"][0]["message"]
        return {"content": msg.get("content") or "", "tool_calls": msg.get("tool_calls") or [],
                "usage": resp.get("usage", {}), "latency_s": time.perf_counter() - t0}


class ScriptedLLM:
    """Replays a fixed list of assistant turns — for smoke tests and reproducing trajectories."""

    def __init__(self, turns: list[dict]):
        self.turns, self.i = turns, 0

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        turn = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        return {"content": turn.get("content", ""), "tool_calls": turn.get("tool_calls", []),
                "usage": {}, "latency_s": 0.0}
