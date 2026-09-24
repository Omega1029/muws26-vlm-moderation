"""run_agent.py — run the agent over a task file, verify each patch, append results as JSONL.

  # toy tasks against a local Gemma 4 served by vLLM
  python scripts/run_agent.py --tasks tasks/toy.jsonl --model google/gemma-4-E4B-it \
      --base_url http://localhost:8000/v1 --tag e4b-native

  # same model, force the text-JSON tool protocol (ablation: native vs text calling)
  python scripts/run_agent.py ... --no_native_tools --tag e4b-text

Resumable: (tag, instance_id, seed) keys already in --out are skipped.
Full trajectories go to results/trajectories/<tag>/<instance_id>__s<seed>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swe_agent.agent import run_agent  # noqa: E402
from swe_agent.diagnostics import episode_metrics  # noqa: E402
from swe_agent.llm import OpenAICompatLLM  # noqa: E402
from swe_agent.tasks import fresh_checkout, load_tasks, verify  # noqa: E402
from swe_agent.tools import Workspace  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--tag", required=True, help="experiment cell name, e.g. e4b-native-t0")
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=1, help="repeats per task (needs temperature>0 to differ)")
    ap.add_argument("--no_native_tools", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on instance_id")
    ap.add_argument("--out", default=str(ROOT / "results" / "runs.jsonl"))
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["tag"], r["instance_id"], r["seed"]))
    traj_dir = out.parent / "trajectories" / a.tag
    traj_dir.mkdir(parents=True, exist_ok=True)

    llm = OpenAICompatLLM(a.model, a.base_url, temperature=a.temperature,
                          native_tools=not a.no_native_tools)
    tasks = [t for t in load_tasks(a.tasks) if not a.only or a.only in t["instance_id"]]
    for t in tasks:
        for seed in range(a.seeds):
            if (a.tag, t["instance_id"], seed) in done:
                continue
            ws = Workspace(fresh_checkout(t["repo_dir"]))
            res = run_agent(llm, ws, t["problem_statement"], max_steps=a.max_steps)
            v = verify(t, res["patch"])
            row = {"tag": a.tag, "model": a.model, "instance_id": t["instance_id"], "seed": seed,
                   "native_tools": not a.no_native_tools, "temperature": a.temperature,
                   "max_steps": a.max_steps, "resolved": v["resolved"], "patch_applied": v["applied"],
                   **episode_metrics(res)}
            (traj_dir / f"{t['instance_id']}__s{seed}.json").write_text(
                json.dumps({**row, "patch": res["patch"], "steps": res["steps"],
                            "test_output": v["test_output"]}, indent=1, default=str))
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{a.tag}] {t['instance_id']} s{seed}: resolved={v['resolved']} "
                  f"steps={row['n_steps']} valid={row['valid_call_rate']} exit={row['exit_reason']}", flush=True)


if __name__ == "__main__":
    main()
