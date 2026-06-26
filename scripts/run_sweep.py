#!/usr/bin/env python3
"""run_sweep.py — expand the experiment matrix and score every cell.

For each (model, quant) we load ONCE, then run each applicable (adaptation, dataset)
cell against it — amortizing the load like the openvla quant sweep. Each cell appends
one record to results/leaderboard.jsonl; reruns skip cells already present (by key).

USAGE:
  ../venv/bin/python -m scripts.run_sweep --config configs/experiment_matrix.yaml
  ../venv/bin/python -m scripts.run_sweep --smoke          # just the smoke cell(s)
  ../venv/bin/python -m scripts.run_sweep --eval_limit 200 # cap eval examples
  ../venv/bin/python -m scripts.run_sweep --list           # print grid and exit
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import load_model
from src.data import load_split
from src import eval as ev
from src import benchmark as bench


def cell_key(model, quant, adaptation, dataset) -> str:
    return f"{model}|{quant}|{adaptation}|{dataset}"


def skip_cell(cell, skips) -> bool:
    """A skip rule matches if every key it specifies equals the cell's value."""
    for rule in skips:
        if all(cell.get(k) == v for k, v in rule.items()):
            return True
    return False


def done_keys(path: Path) -> set[str]:
    """Keys of *successful* cells only — errored cells are retried on rerun."""
    if not path.exists():
        return set()
    done = set()
    for l in path.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if "key" in r and "metrics" in r:
            done.add(r["key"])
    return done


def expand(cfg, smoke=False):
    """Yield cell dicts. Smoke mode yields only the listed smoke cells."""
    if smoke:
        for c in cfg["grid"].get("smoke", []):
            yield c
        return
    models = {m["name"]: m for m in cfg["models"]}
    quant = {q["name"]: q for q in cfg["quant"]}
    adapt = {a["name"]: a for a in cfg["adaptation"]}
    data = {d["name"]: d for d in cfg["datasets"]}
    skips = cfg["grid"].get("skip", [])
    for mn, qn, an, dn in product(models, quant, adapt, data):
        cell = {"model": mn, "quant": qn, "adaptation": an, "dataset": dn}
        if not skip_cell(cell, skips):
            yield cell


def few_shot_pool(dataset, n_shots, split="train"):
    """Class-balanced in-context demonstrations for few-shot cells."""
    if n_shots == 0:
        return None, None
    pool = load_split(dataset, split, limit=200)
    pos = [e for e in pool if e.label == 1][: n_shots // 2]
    neg = [e for e in pool if e.label == 0][: n_shots - len(pos)]
    shots = pos + neg
    return [{"text": e.text, "label": e.label} for e in shots], [e.image for e in shots]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/experiment_matrix.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default=None,
                    help="substring filter on cell key, e.g. 'smolvlm-500m|fp16|zeroshot'")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--eval_limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    models = {m["name"]: m for m in cfg["models"]}
    quant = {q["name"]: q for q in cfg["quant"]}
    adapt = {a["name"]: a for a in cfg["adaptation"]}
    data = {d["name"]: d for d in cfg["datasets"]}

    cells = list(expand(cfg, smoke=args.smoke))
    if args.only:
        cells = [c for c in cells
                 if args.only in cell_key(c["model"], c["quant"], c["adaptation"], c["dataset"])]
    if args.list:
        for c in cells:
            print("  " + cell_key(c["model"], c["quant"], c["adaptation"], c["dataset"]))
        print(f"[sweep] {len(cells)} cells")
        return

    results_path = ROOT / cfg["meta"]["results_path"]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    already = done_keys(results_path)

    # Group cells by (model, quant) so we load each loaded-config exactly once.
    by_load: dict[tuple, list] = {}
    for c in cells:
        by_load.setdefault((c["model"], c["quant"]), []).append(c)

    for (mn, qn), group in by_load.items():
        m, q = models[mn], quant[qn]
        pending = [c for c in group
                   if cell_key(c["model"], c["quant"], c["adaptation"], c["dataset"]) not in already]
        if not pending:
            continue
        print(f"\n{'='*70}\n[sweep] LOAD {mn} @ {qn}\n{'='*70}")
        model, processor = load_model(m["hf_id"], q, m["family"], args.device)

        for c in pending:
            an, dn = c["adaptation"], c["dataset"]
            key = cell_key(mn, qn, an, dn)
            ds = data[dn]
            a = adapt[an]
            print(f"[sweep] CELL {key}")
            rec = {"key": key, "model": mn, "quant": qn, "adaptation": an, "dataset": dn,
                   "params": m["params"], "primary_metric": ds["primary_metric"],
                   "gpu": cfg["meta"]["gpu"], "timestamp": datetime.now().isoformat(timespec="seconds")}
            try:
                # NOTE: lora cells should load the merged ckpt instead of m["hf_id"];
                # wired once train_lora.py emits checkpoints (see grid.adaptation.ckpt_dir).
                fs, fs_imgs = few_shot_pool(dn, a.get("n_shots", 0), ds["splits"]["train"])
                eval_set = load_split(dn, ds["splits"]["eval"], args.eval_limit)
                # prompts are keyed by dataset name (dn); ds["task"] is the metric type.
                rec.update(ev.evaluate_cell(model, processor, dn, ds["primary_metric"],
                                            eval_set, fs, fs_imgs, args.device))
                # one latency point per (model,quant); attach on first dataset cell
                sample = eval_set[0]
                msgs = __import__("src.prompts", fromlist=["build_messages"]).build_messages(
                    dn, sample.text, None)
                inputs = ev._prepare_inputs(processor, msgs, [sample.image], args.device)
                rec["bench"] = bench.benchmark_latency(model, inputs, args.device)
            except Exception as e:
                import traceback; traceback.print_exc()
                rec["error"] = repr(e)
            with open(results_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if "metrics" in rec:
                print(f"[sweep] {key}: {ds['primary_metric']}={rec['metrics']['primary']} "
                      f"vram={rec.get('peak_vram_mb')}MB")
        del model
        import torch; torch.cuda.empty_cache()

    print(f"\n[sweep] done → {results_path}")


if __name__ == "__main__":
    main()
