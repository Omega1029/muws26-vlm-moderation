"""train_lora.py — LoRA fine-tuning for the `adaptation: lora` cells.

Generative SFT: the model learns to emit "yes"/"no" for the task prompt (the SAME
prompt as eval, so train/eval are consistent). Loss is computed on the answer tokens
only — the prompt (instruction + image + question) is masked with -100.

Trained in bf16; we save the MERGED model so the int8/int4 LoRA sweep cells can load
it and quantize at load time, isolating "does LoRA recover accuracy?" from "does
quantization erode it?". Batch size 1 + gradient accumulation avoids the headache of
padding variable-length multimodal inputs (fine for SmolVLM-500M on an A100).

USAGE:
  CUDA_VISIBLE_DEVICES=0 ./venv/bin/python -m src.train_lora \
      --model_hf HuggingFaceTB/SmolVLM-500M-Instruct --dataset hateful_memes \
      --out checkpoints/smolvlm-500m_hateful_memes_lora --epochs 2 --train_limit 3000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor

try:  # transformers ≥5 renamed Vision2Seq → ImageTextToText
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVLM

from . import prompts
from .data import load_split

# Attention + MLP projections — the standard VLM LoRA target set.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_example_inputs(processor, task, ex, device):
    """One supervised example: prompt+answer tensors with prompt tokens masked (-100).

    `prompt` uses add_generation_prompt=True so it ends exactly where the answer begins;
    its token length is the mask boundary in the full (prompt+answer) sequence. Image
    tokens appear identically in both (same image), so the boundary is exact.
    """
    msgs = prompts.build_messages(task, ex.text, few_shot=None)
    answer = "yes" if ex.label == 1 else "no"
    full_msgs = msgs + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]

    full_text = processor.apply_chat_template(full_msgs, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(msgs, add_generation_prompt=True)

    full = processor(text=full_text, images=[ex.image], return_tensors="pt")
    plen = processor(text=prompt_text, images=[ex.image], return_tensors="pt")["input_ids"].shape[1]

    labels = full["input_ids"].clone()
    labels[:, :plen] = -100
    full["labels"] = labels
    return {k: v.to(device) for k, v in full.items()}


class SFTDataset(Dataset):
    """Builds masked-label tensors on CPU so DataLoader workers can parallelize the
    (expensive) image preprocessing across cores while the GPU trains — this is what
    makes the loop GPU-bound instead of CPU-bound."""
    def __init__(self, examples, processor, task):
        self.examples, self.processor, self.task = examples, processor, task

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return build_example_inputs(self.processor, self.task, self.examples[i], device="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_hf", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--train_limit", type=int, default=None)
    ap.add_argument("--task", default=None, help="prompt key; defaults to --dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=6)
    args = ap.parse_args()

    task = args.task or args.dataset
    device = "cuda:0"
    torch.manual_seed(args.seed)

    assert torch.cuda.is_available(), "CUDA not available — refusing to train on CPU"
    processor = AutoProcessor.from_pretrained(args.model_hf)
    model = AutoVLM.from_pretrained(
        args.model_hf, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    model.config.use_cache = False

    lcfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.train()

    # explicit GPU confirmation
    pdev = next(model.parameters()).device
    print(f"[train_lora] model on {pdev} | cuda:{torch.cuda.current_device()} "
          f"= {torch.cuda.get_device_name(0)}", flush=True)
    assert pdev.type == "cuda", f"model is on {pdev}, not GPU!"

    train = load_split(args.dataset, args.split, args.train_limit, seed=args.seed)
    ds = SFTDataset(train, processor, task)
    gen = torch.Generator().manual_seed(args.seed)
    # batch=1, but `num_workers` processes preprocess images in parallel → GPU-bound.
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=args.num_workers,
                        collate_fn=lambda b: b[0], pin_memory=True,
                        persistent_workers=args.num_workers > 0,
                        prefetch_factor=4 if args.num_workers > 0 else None, generator=gen)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    print(f"[train_lora] {len(train)} ex | task={task} | r={args.rank} | "
          f"bs1×accum{args.grad_accum} | {args.epochs} epochs | workers={args.num_workers}", flush=True)
    t0 = time.time()
    step, run_loss = 0, 0.0
    for epoch in range(args.epochs):
        for i, inputs in enumerate(loader):
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            loss = model(**inputs).loss / args.grad_accum
            loss.backward()
            run_loss += loss.item() * args.grad_accum
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
                step += 1
                if step % args.log_every == 0:
                    rate = (step * args.grad_accum) / (time.time() - t0)
                    print(f"  e{epoch} step {step} | loss {run_loss / (args.log_every * args.grad_accum):.4f} "
                          f"| {rate:.1f} ex/s | gpu_mem {torch.cuda.max_memory_allocated(device)/1e9:.1f}GB "
                          f"| {(time.time()-t0)/60:.1f}m", flush=True)
                    run_loss = 0.0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[train_lora] merging + saving → {out}")
    merged = model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    processor.save_pretrained(out)
    print(f"[train_lora] done in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
