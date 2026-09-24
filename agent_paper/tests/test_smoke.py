"""End-to-end smoke test with a scripted LLM — no GPU or server needed.

  python -m unittest discover -s tests -v      (from agent_paper/)
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swe_agent.agent import parse_text_tool_call, run_agent  # noqa: E402
from swe_agent.diagnostics import aggregate, episode_metrics  # noqa: E402
from swe_agent.llm import ScriptedLLM  # noqa: E402
from swe_agent.tasks import fresh_checkout, load_tasks, verify  # noqa: E402
from swe_agent.tools import Workspace  # noqa: E402

TASKS = {t["instance_id"]: t for t in load_tasks(ROOT / "tasks" / "toy.jsonl")}


def native(name, **args):
    return {"tool_calls": [{"id": "c", "function": {"name": name, "arguments": json.dumps(args)}}]}


def text(name, **args):
    return {"content": "```json\n" + json.dumps({"tool": name, "args": args}) + "\n```"}


class TestAgent(unittest.TestCase):
    def episode(self, task_id, turns, max_steps=10):
        task = TASKS[task_id]
        res = run_agent(ScriptedLLM(turns), Workspace(fresh_checkout(task["repo_dir"])),
                        task["problem_statement"], max_steps=max_steps)
        return res, verify(task, res["patch"])

    def test_good_trajectory_resolves(self):
        res, v = self.episode("toy-clamp", [
            native("read_file", path="calc/ops.py"),
            native("edit_file", path="calc/ops.py", old_str="max(hi, min(lo, x))", new_str="max(lo, min(hi, x))"),
            text("run_command", command="python -m unittest discover -s tests -q"),
            native("submit_patch"),
        ])
        m = episode_metrics(res)
        self.assertTrue(v["resolved"], v["test_output"])
        self.assertTrue(m["submitted"] and m["verified_before_submit"])
        self.assertEqual(m["valid_call_rate"], 1.0)
        self.assertAlmostEqual(m["text_fallback_rate"], 0.25)

    def test_degenerate_trajectory_is_diagnosed(self):
        # a looping model: same read over and over, never edits, runs out of steps
        res, v = self.episode("toy-mean", [native("read_file", path="calc/ops.py")], max_steps=6)
        m = episode_metrics(res)
        self.assertFalse(v["resolved"])
        self.assertEqual(m["exit_reason"], "max_steps")
        self.assertTrue(m["empty_patch"])
        self.assertEqual(m["dominant_tool_share"], 1.0)
        self.assertGreater(m["repeat_action_rate"], 0.8)

    def test_blind_wrong_submit_counts_as_false_submit(self):
        res, v = self.episode("toy-duration", [
            {"content": "I think the bug is obvious."},  # no tool call
            native("edit_file", path="calc/ops.py", old_str='"s": 1', new_str='"s": 2'),
            native("submit_patch"),
        ])
        m = {**episode_metrics(res), "resolved": v["resolved"]}
        self.assertFalse(m["verified_before_submit"])
        self.assertEqual(aggregate([m])["false_submit_rate"], 1.0)
        self.assertGreater(m["no_call_rate"], 0)

    def test_tool_errors_are_recorded_not_raised(self):
        res, _ = self.episode("toy-mean", [
            native("edit_file", path="calc/ops.py", old_str="does not exist", new_str="x"),
            native("read_file", path="../../etc/passwd"),
            native("frobnicate"),
            native("submit_patch"),
        ])
        oks = [s["ok"] for s in res["steps"]]
        self.assertEqual(oks, [False, False, False, True])

    def test_text_parser(self):
        self.assertEqual(parse_text_tool_call('{"tool": "get_status", "args": {}}'), ("get_status", {}))
        self.assertIsNone(parse_text_tool_call("no json here"))


if __name__ == "__main__":
    unittest.main()
