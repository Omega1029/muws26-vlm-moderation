#!/usr/bin/env python3
"""make_figures.py — the two headline figures from results/leaderboard.jsonl.

  fig1 (pareto)      : primary metric vs. latency (and vs. VRAM), one point per
                       model+quant, frontier highlighted.
  fig2 (degradation) : primary metric vs. quant level (fp16→int8→int4) per dataset,
                       showing whether sarcasm degrades faster than hate.

USAGE:
  ../venv/bin/python -m scripts.make_figures --results results/leaderboard.jsonl --out paper/figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

QUANT_ORDER = {"fp16": 0, "int8": 1, "int4": 2}


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return [r for r in rows if "metrics" in r and r["metrics"].get("primary") is not None]


def pareto_front(points):
    """Lower latency + higher metric is better. Return indices on the frontier."""
    front, best = [], -1.0
    for i in sorted(range(len(points)), key=lambda i: points[i][0]):  # by latency asc
        if points[i][1] > best:
            best = points[i][1]
            front.append(i)
    return set(front)


def fig_pareto(rows, dataset, out):
    sub = [r for r in rows if r["dataset"] == dataset and "bench" in r]
    if not sub:
        return
    pts = [(r["bench"]["latency_ms_median"], r["metrics"]["primary"]) for r in sub]
    front = pareto_front(pts)
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for i, (r, (lat, met)) in enumerate(zip(sub, pts)):
        ax.scatter(lat, met, s=40, marker="*" if r["model"].endswith("7b") else "o",
                   edgecolor="k" if i in front else "none", zorder=3 if i in front else 2)
        ax.annotate(f'{r["model"].split("-")[-1]}/{r["quant"]}', (lat, met),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    fl = sorted([pts[i] for i in front])
    ax.plot([p[0] for p in fl], [p[1] for p in fl], "--", lw=1, color="0.4", zorder=1)
    ax.set_xlabel("latency (ms/sample, A100, bs=1)"); ax.set_ylabel(sub[0]["primary_metric"])
    ax.set_title(f"{dataset}: accuracy–latency Pareto"); fig.tight_layout()
    fig.savefig(Path(out) / f"pareto_{dataset}.pdf"); plt.close(fig)


def fig_degradation(rows, out):
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for dataset in sorted({r["dataset"] for r in rows}):
        for model in sorted({r["model"] for r in rows if r["dataset"] == dataset}):
            series = sorted([r for r in rows if r["dataset"] == dataset and r["model"] == model
                             and r["adaptation"] == "zeroshot"],
                            key=lambda r: QUANT_ORDER.get(r["quant"], 9))
            if len(series) < 2:
                continue
            ax.plot([r["quant"] for r in series], [r["metrics"]["primary"] for r in series],
                    marker="o", label=f"{dataset}/{model}")
    ax.set_xlabel("quantization"); ax.set_ylabel("primary metric")
    ax.set_title("Compression degradation"); ax.legend(fontsize=5)
    fig.tight_layout(); fig.savefig(Path(out) / "degradation.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/leaderboard.jsonl")
    ap.add_argument("--out", default="paper/figs")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    rows = load(args.results)
    for ds in sorted({r["dataset"] for r in rows}):
        fig_pareto(rows, ds, args.out)
    fig_degradation(rows, args.out)
    print(f"[figs] wrote pareto_*.pdf + degradation.pdf → {args.out}")


if __name__ == "__main__":
    main()
