"""data.py — uniform loaders for the two benchmarks.

Both return a list of Example(image: PIL.Image, text: str, label: int). The eval/train
loops never see dataset-specific structure beyond this.

Access notes (verify before the full run):
  • hateful_memes — gated (Meta license). Expects the official release unpacked at
    $HM_ROOT (img/*.png + {train,dev_seen,test_seen}.jsonl). Set HM_ROOT env var.
  • mmsd2        — MMSD2.0; loadable from HF (`coderchen01/MMSD2.0` or local mirror).
    Set MMSD2_ROOT for a local copy, else HF hub.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class Example:
    image: Image.Image
    text: str
    label: int


def load_hateful_memes(split: str, limit: int | None = None, seed: int = 0) -> list[Example]:
    root = Path(os.environ.get("HM_ROOT", "data/hateful_memes"))
    jsonl = root / f"{split}.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(
            f"{jsonl} not found. Download the gated Hateful Memes release and set "
            f"HM_ROOT, or point --hm_root at it.")
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    # dev/test jsonl are label-sorted — shuffle before limiting so any slice is balanced.
    random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]
    out = []
    for r in rows:
        img = Image.open(root / r["img"]).convert("RGB")
        out.append(Example(image=img, text=r["text"], label=int(r.get("label", 0))))
    return out


def load_mmsd2(split: str, limit: int | None = None, seed: int = 0) -> list[Example]:
    # Local mirror first, else HF hub. MMSD2.0 fields vary by mirror; map defensively.
    local = os.environ.get("MMSD2_ROOT")
    if local:
        root = Path(local)
        rows = [json.loads(l) for l in (root / f"{split}.jsonl").read_text().splitlines() if l.strip()]
        random.Random(seed).shuffle(rows)
        img_dir = root / "images"
        examples = ((r, img_dir / f'{r["image_id"]}.jpg') for r in rows)
    else:
        from datasets import load_dataset
        # MMSD2.0 ships 3 configs; "mmsd-v2" is the de-biased corpus the paper introduces.
        config = os.environ.get("MMSD2_CONFIG", "mmsd-v2")
        ds = load_dataset("coderchen01/MMSD2.0", config, split=split).shuffle(seed=seed)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return [Example(image=r["image"].convert("RGB"),
                        text=r.get("text", r.get("caption", "")),
                        label=int(r["label"])) for r in ds]
    out = []
    for i, (r, p) in enumerate(examples):
        if limit and i >= limit:
            break
        out.append(Example(image=Image.open(p).convert("RGB"),
                            text=r.get("text", ""), label=int(r["label"])))
    return out


LOADERS = {"hateful_memes": load_hateful_memes, "mmsd2": load_mmsd2}


def load_split(dataset: str, split: str, limit: int | None = None, seed: int = 0) -> list[Example]:
    return LOADERS[dataset](split, limit, seed)
