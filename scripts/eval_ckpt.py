#!/usr/bin/env python3
"""eval_ckpt.py — evaluate one checkpoint on one dataset (transfer-matrix cell).

Decouples the trained-on dataset from the evaluated-on dataset, so the same script
covers in-domain (train==eval), cross-domain, and cross-task cells. Loads a base
hf_id OR a local merged-LoRA dir, applies a quant level, scores the eval split, and
appends a record keyed by (model, train_on, adaptation, quant, eval_on).

USAGE:
  # in-domain: LoRA-HM evaluated on HM
  ./venv/bin/python -m scripts.eval_ckpt --model checkpoints/smolvlm-500m_hateful_memes_lora \
      --base smolvlm-500m --train_on hateful_memes --adaptation lora \
      --eval_dataset hateful_memes --quant fp16 --eval_limit 500
  # cross-task: LoRA-HM evaluated on MMSD2
  ... --eval_dataset mmsd2 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from src.models import load_model
from src.data import load_split
from src import eval as ev
from src import prompts
from src import benchmark as bench

QUANTS = {
    "fp16": {"load_dtype": "float16"},
    "int8": {"bnb_8bit": True},
    "int4": {"bnb_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "float16"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="hf_id or local merged-ckpt dir")
    ap.add_argument("--base", required=True, help="base model name for the record, e.g. smolvlm-500m")
    ap.add_argument("--train_on", required=True, help="dataset the ckpt was trained on (or 'none')")
    ap.add_argument("--adaptation", default="lora", help="lora | zeroshot | fewshot")
    ap.add_argument("--eval_dataset", required=True)
    ap.add_argument("--quant", default="fp16", choices=list(QUANTS))
    ap.add_argument("--eval_limit", type=int, default=500)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", default=str(ROOT / "configs/experiment_matrix.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dmeta = {d["name"]: d for d in cfg["datasets"]}[args.eval_dataset]
    results_path = ROOT / cfg["meta"]["results_path"]
    results_path.parent.mkdir(parents=True, exist_ok=True)

    key = f"{args.base}|{args.quant}|{args.adaptation}@{args.train_on}|{args.eval_dataset}"
    print(f"[eval_ckpt] {key}\n[eval_ckpt] loading {args.model} @ {args.quant}")
    model, processor = load_model(args.model, QUANTS[args.quant], "auto", args.device)

    eval_set = load_split(args.eval_dataset, dmeta["splits"]["eval"], args.eval_limit)
    rec = {"key": key, "model": args.base, "train_on": args.train_on,
           "adaptation": args.adaptation, "quant": args.quant, "dataset": args.eval_dataset,
           "primary_metric": dmeta["primary_metric"], "gpu": cfg["meta"]["gpu"],
           "transfer": "in_domain" if args.train_on == args.eval_dataset else
                       ("cross_task" if args.train_on != "none" else "zeroshot"),
           "timestamp": datetime.now().isoformat(timespec="seconds")}
    rec.update(ev.evaluate_cell(model, processor, args.eval_dataset, dmeta["primary_metric"],
                                eval_set, None, None, args.device))
    sample = eval_set[0]
    msgs = prompts.build_messages(args.eval_dataset, sample.text, None)
    inputs = ev._prepare_inputs(processor, msgs, [sample.image], args.device)
    rec["bench"] = bench.benchmark_latency(model, inputs, args.device)

    with open(results_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    m = rec["metrics"]
    print(f"[eval_ckpt] {key}: {dmeta['primary_metric']}={m['primary']} "
          f"auroc={m['auroc']} pred_pos={rec['pred_pos_rate']} ({rec['transfer']})")


if __name__ == "__main__":
    main()
