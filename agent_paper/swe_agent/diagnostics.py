"""diagnostics.py — trajectory-level metrics that resolve rate hides.

Carries over the MUWS26 lesson ("zero-shot models are broken scanners and F1 hides
it"): a single headline number (% resolved) can't tell a model that tries and
fails apart from one that is degenerate (loops, malformed calls, submits blind).
These per-episode numbers are what the paper reports next to resolve rate.
"""
from __future__ import annotations

import json
from collections import Counter

EDIT_TOOLS = {"edit_file", "write_file"}


def _key(step: dict) -> str:
    return json.dumps([step.get("tool"), step.get("args")], sort_keys=True, default=str)


def episode_metrics(result: dict) -> dict:
    steps = [s for s in result["steps"] if "call_format" in s]
    n = len(steps)
    tools = [s.get("tool") for s in steps]
    fmt = Counter(s["call_format"] for s in steps)
    counts = Counter(t for t in tools if t)
    keys = [_key(s) for s in steps if s.get("tool")]
    edit_idx = [i for i, s in enumerate(steps) if s.get("tool") in EDIT_TOOLS and s.get("ok")]
    last_edit = edit_idx[-1] if edit_idx else None
    ran_after_edit = last_edit is not None and any(
        s.get("tool") == "run_command" and s.get("ok") for s in steps[last_edit + 1:])
    return {
        "n_steps": n,
        "submitted": result["submitted"],
        "exit_reason": result["exit_reason"],
        "empty_patch": not result["patch"].strip(),
        # format health — the agent analogue of "is the scanner even emitting labels?"
        "valid_call_rate": round(sum(bool(s.get("ok")) for s in steps) / n, 3) if n else 0.0,
        "native_rate": round(fmt["native"] / n, 3) if n else 0.0,
        "text_fallback_rate": round(fmt["text"] / n, 3) if n else 0.0,
        "no_call_rate": round(fmt["none"] / n, 3) if n else 0.0,
        # collapse — the agent analogue of pred_pos_rate
        "dominant_tool": counts.most_common(1)[0][0] if counts else None,
        "dominant_tool_share": round(counts.most_common(1)[0][1] / n, 3) if counts else 0.0,
        "repeat_action_rate": round(1 - len(set(keys)) / len(keys), 3) if keys else 0.0,
        # process quality
        "steps_to_first_edit": edit_idx[0] if edit_idx else None,
        "verified_before_submit": bool(result["submitted"] and ran_after_edit),
        "total_tokens": sum((s.get("usage") or {}).get("total_tokens", 0) for s in result["steps"]),
        "wall_s": round(result["wall_s"], 2),
    }


def aggregate(rows: list[dict]) -> dict:
    """Mean over episodes; rows = episode_metrics(...) + {"resolved": bool}."""
    if not rows:
        return {}
    n = len(rows)
    mean = lambda k: round(sum(float(r[k] or 0) for r in rows) / n, 3)
    resolved = [r for r in rows if r.get("resolved")]
    return {
        "n": n,
        "resolve_rate": mean("resolved"),
        "submit_rate": mean("submitted"),
        "empty_patch_rate": mean("empty_patch"),
        # submitted-but-wrong: confident failure, the costliest mode for a real user
        "false_submit_rate": round(sum(r["submitted"] and not r["resolved"] for r in rows) / n, 3),
        "verified_before_submit_rate": mean("verified_before_submit"),
        "valid_call_rate": mean("valid_call_rate"),
        "text_fallback_rate": mean("text_fallback_rate"),
        "no_call_rate": mean("no_call_rate"),
        "dominant_tool_share": mean("dominant_tool_share"),
        "repeat_action_rate": mean("repeat_action_rate"),
        "mean_steps": mean("n_steps"),
        "mean_steps_resolved": round(sum(r["n_steps"] for r in resolved) / len(resolved), 2) if resolved else None,
        "mean_tokens": mean("total_tokens"),
    }
