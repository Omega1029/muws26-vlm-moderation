"""eval.py — score one (model, quant, adaptation, dataset) cell and return a record.

Mirrors the openvla quant-sweep contract: the caller loads the model once and passes
it in; we iterate the eval split, collect (label, pred, score), compute metrics, and
return a dict the sweep appends to the jsonl leaderboard. Peak VRAM during eval is
captured here; dedicated latency numbers come from benchmark.py.
"""
from __future__ import annotations

import time

import torch

from . import prompts
from . import metrics as M
from .data import Example


def _has_chat_template(processor) -> bool:
    return bool(getattr(processor, "chat_template", None)
                or getattr(getattr(processor, "tokenizer", None), "chat_template", None))


def _prepare_inputs(processor, messages, images, device):
    """Render the prompt (chat template if available, else a raw flattened prompt for
    untemplated processors like Gemma4) and bind images → tensors on `device`."""
    if _has_chat_template(processor):
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
    else:
        text = prompts.flatten_messages(processor, messages)
    inputs = processor(text=text, images=images, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def evaluate_cell(model, processor, task: str, primary_metric: str,
                  eval_set: list[Example], few_shot: list[dict] | None,
                  few_shot_images: list | None, device="cuda:0") -> dict:
    """Run the eval split for one cell. Uses the yes-token logit score for AUROC and
    threshold-0.5 for the hard label. few_shot/few_shot_images are the in-context
    demonstrations (None for zero-shot / lora)."""
    torch.cuda.reset_peak_memory_stats(device)
    y_true, y_pred, y_score = [], [], []
    t0 = time.time()
    for ex in eval_set:
        msgs = prompts.build_messages(task, ex.text, few_shot)
        images = (few_shot_images or []) + [ex.image]
        inputs = _prepare_inputs(processor, msgs, images, device)
        p_yes = prompts.score_yes_prob(model, processor, inputs, device)
        y_true.append(ex.label)
        y_score.append(p_yes)
        y_pred.append(int(p_yes >= 0.5))

    met = M.compute(primary_metric, y_true, y_pred, y_score)
    n = len(eval_set)
    return {
        "n_eval": n,
        "metrics": met,
        # bias diagnostics: small VLMs collapse to a constant answer (see yes-bias finding).
        # pred_pos_rate ≈ pos_rate means calibrated; pred_pos_rate ≈ 1 or 0 means collapse.
        "pos_rate": round(sum(y_true) / n, 3),
        "pred_pos_rate": round(sum(y_pred) / n, 3),
        "mean_score": round(sum(y_score) / n, 3),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1),
        "eval_seconds": round(time.time() - t0, 1),
    }
