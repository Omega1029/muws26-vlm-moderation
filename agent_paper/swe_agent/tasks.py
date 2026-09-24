"""tasks.py — SWE-style task loading, workspace setup, and hidden-test verification.

Task JSONL, one object per line (paths relative to the JSONL file):
  {"instance_id": "...", "repo_dir": "toy_calc", "problem_statement": "...",
   "hidden_tests": {"tests/test_x.py": "toy_calc_hidden/test_x.py"}, "test_command": "python -m pytest -q"}

Verification mirrors the competition: apply the agent's diff to a FRESH copy of the
repo, drop in the hidden tests, run the test command. The agent never sees them.
SWE-bench instances can be converted to this format (see README) once repos are cloned.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def load_tasks(path: str | Path) -> list[dict]:
    path = Path(path).resolve()
    tasks = []
    for line in path.read_text().splitlines():
        if line.strip():
            t = json.loads(line)
            t["repo_dir"] = str(path.parent / t["repo_dir"])
            t["hidden_tests"] = {dst: str(path.parent / src) for dst, src in t.get("hidden_tests", {}).items()}
            tasks.append(t)
    return tasks


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def fresh_checkout(repo_dir: str, into: Path | None = None) -> Path:
    """Copy repo_dir to a temp dir and make it a one-commit git repo (so `git diff` = agent's patch)."""
    dst = Path(into or tempfile.mkdtemp(prefix="swe_")) / "repo"
    shutil.copytree(repo_dir, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".git"))
    _git(dst, "init", "-q")
    _git(dst, "add", "-A")
    _git(dst, "-c", "user.email=a@b", "-c", "user.name=agent", "commit", "-qm", "base")
    return dst


def verify(task: dict, patch: str, timeout: int = 600) -> dict:
    """Return {"resolved": bool, "applied": bool, "test_output": str}."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = fresh_checkout(task["repo_dir"], Path(tmp))
        if patch.strip():
            (repo / ".agent.patch").write_text(patch)
            r = subprocess.run(["git", "apply", ".agent.patch"], cwd=repo, capture_output=True, text=True)
            if r.returncode != 0:
                return {"resolved": False, "applied": False, "test_output": r.stderr[-2000:]}
        for dst, src in task["hidden_tests"].items():
            (repo / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, repo / dst)
        try:
            r = subprocess.run(task["test_command"], shell=True, cwd=repo, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"resolved": False, "applied": True, "test_output": "[timeout]"}
        return {"resolved": r.returncode == 0, "applied": True, "test_output": (r.stdout + r.stderr)[-2000:]}
