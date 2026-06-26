"""benchmark.py — the latency/efficiency protocol. Lock this down early.

Reports, on one named GPU (A100-80GB), per (model, quant) point:
  • latency  : median + IQR ms/sample at batch=1 (single-sample moderation = the
               on-device/serving unit), after warmup, fixed generation length.
  • throughput: samples/s at a fixed batch (serving-cost view).
  • peak_vram: MB at batch=1.
  • footprint: static weight bytes (models.model_footprint_mb).

Protocol knobs are explicit and recorded so the table is reproducible: WARMUP runs
discarded, N timed runs, fixed max_new_tokens, torch.cuda.synchronize around each.
"""
from __future__ import annotations

import statistics
import time

import torch

from .models import model_footprint_mb

WARMUP = 5
N_TIMED = 50
MAX_NEW_TOKENS = 1          # we only need the yes/no answer token


@torch.no_grad()
def _time_once(model, inputs, device) -> float:
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000.0   # ms


@torch.no_grad()
def benchmark_latency(model, inputs, device="cuda:0") -> dict:
    """inputs: a single prepared sample (batch=1) on `device`."""
    for _ in range(WARMUP):
        _time_once(model, inputs, device)
    samples = [_time_once(model, inputs, device) for _ in range(N_TIMED)]
    samples.sort()
    q1, q3 = statistics.quantiles(samples, n=4)[0], statistics.quantiles(samples, n=4)[2]
    torch.cuda.reset_peak_memory_stats(device)
    _time_once(model, inputs, device)
    return {
        "latency_ms_median": round(statistics.median(samples), 2),
        "latency_ms_iqr": round(q3 - q1, 2),
        "latency_ms_p10": round(samples[len(samples) // 10], 2),
        "throughput_sps": round(1000.0 / statistics.median(samples), 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1),
        "footprint_mb": round(model_footprint_mb(model), 1),
        "protocol": {"warmup": WARMUP, "n_timed": N_TIMED, "max_new_tokens": MAX_NEW_TOKENS, "batch": 1},
    }
