# MUWS 2026 — Project Summary
**Workshop:** MUWS 2026 @ ACM Multimedia, Rio de Janeiro, Nov 10–14, 2026  
**Submission deadline:** July 16, 2026  
**Format:** ACM acmart `sigconf, nonacm` · pdfLaTeX · ~4 pages (current draft ~6–7 pages; trimming notes in combined `.tex`)

---

## Paper

**Title:** *Scale, Collapse, Adaptation, Transfer: What Governs Small Vision–Language Models on Multimodal Hate and Sarcasm Detection*

**Files (copy/paste into Overleaf):**

| File | Description |
|---|---|
| `paper/muws2026_combined.tex` | **Main deliverable** — all 4 findings unified |
| `paper/p1_collapse.tex` | Draft 1 — zero-shot collapse + metric failure standalone |
| `paper/p2_transfer.tex` | Draft 2 — asymmetric cross-task transfer standalone |
| `paper/p3_hardness.tex` | Draft 3 — 7× difficulty gap hate vs. sarcasm standalone |
| `paper/p4_efficiency.tex` | Draft 4 — Pareto/efficiency story standalone |
| `paper/refs.bib` | Shared bibliography |

**Overleaf instructions:** New Project → upload `.tex` + `refs.bib` → Compiler: pdfLaTeX → Compile twice.

---

## Four Findings (with real measured numbers)

### Finding 1 — Scale Is Not Enough
Zero-shot Pareto (fp16, n=500, A100):

| Model | Hate AUROC | Sarc. AUROC | Peak VRAM | Latency |
|---|---|---|---|---|
| SmolVLM-256M | 0.513 | 0.502 | ~961 MB | 70 ms |
| **SmolVLM-500M** | **0.581** | **0.600** | ~1,453 MB | 69 ms |
| Gemma-4-E2B | 0.528 | 0.527 | ~10,170 MB | 86 ms |

**SmolVLM-500M Pareto-dominates Gemma-4-E2B** — higher AUROC on both tasks at 1/7th the memory.

---

### Finding 2 — Zero-Shot Models Collapse; Metrics Hide It
- SmolVLM answers "yes" for **86–98%** of all inputs (`pred_pos_rate`)
- Gemma answers "no" for **100%** of inputs
- A small positive-skewed slice (n=50) yielded an **apparent F1 of 0.89** — a pure artifact of a "yes"-saying model meeting a mostly-positive slice
- On balanced n=500: same config → AUROC 0.50, accuracy below majority baseline
- **Standard reporting practice:** always report AUROC (threshold-free) + `pred_pos_rate`; never evaluate on small skewed slices

---

### Finding 3 — LoRA De-biases First, Then Exposes a 7× Difficulty Gap

| Task | Zero-shot AUROC | +LoRA AUROC | Δ | pred_pos (before→after) |
|---|---|---|---|---|
| Hateful Memes | 0.581 | **0.619** | +0.038 | 0.93 → 0.36 |
| MMSD2.0 | 0.600 | **0.881** | +0.281 | 0.98 → 0.29 |

- Training loss converges to ~0.1 for **both** tasks — the model can fit hate targets just as well
- Only sarcasm generalises to held-out data → fitting ≠ generalisation
- Root cause: Hateful Memes benign-confounder design makes learnable train patterns non-transferable to dev

---

### Finding 4 — Cross-Task Transfer Is Asymmetric

Transfer matrix (SmolVLM-500M, fp16, n=500):

| Train ↓ / Eval → | Hateful Memes AUROC | MMSD2.0 AUROC |
|---|---|---|
| zero-shot (floor) | 0.581 (pred_pos=0.93) | 0.600 (pred_pos=0.98) |
| LoRA-Hate | **0.619** (0.36) | 0.668 (0.61) |
| LoRA-Sarcasm | 0.594 (0.65) | **0.881** (0.29) |

- Hate-trained adapter raises sarcasm to **0.668** — above the sarcasm zero-shot floor AND above the hate in-domain score
- Sarcasm-trained adapter barely moves hate (0.581 → 0.594)
- **Practical takeaway:** when adaptation data is scarce, a hate adapter is a better warm start for sarcasm than the reverse

---

## Experimental Setup

**Models evaluated:**
- `HuggingFaceTB/SmolVLM-256M-Instruct` — 0.256B params, ~489 MB weights
- `HuggingFaceTB/SmolVLM-500M-Instruct` — 0.500B params, ~968 MB weights  
- `google/gemma-4-e2b` — ~2B effective params, ~9,736 MB weights

**Datasets:**
- **Hateful Memes** — binary hate detection, balanced dev split (n=500), metric: AUROC. Images + `.jsonl` labels in `data/hateful_memes/`.
- **MMSD2.0** — `coderchen01/MMSD2.0` config `mmsd-v2`, binary sarcasm, test split (n=500), metric: AUROC/F1.

**Prompting:** Each model receives task instruction + image + yes/no question. Continuous score = softmax P(yes) over answer tokens → AUROC. Gemma prompted with flattened raw prompt (no chat template).

**LoRA config:** r=16, alpha=32, targets: q/k/v/o + gate/up/down projections, 1.85% trainable. Generative SFT: loss on yes/no answer tokens only, prompt masked with -100. 3,000 train examples × 2 epochs, bf16.

**Hardware:** NVIDIA A100 SXM4-80GB. Latency = median of 50 timed single-sample runs (batch=1) after 5 warmup steps.

---

## File Structure

```
MUWS26/
├── configs/
│   └── experiment_matrix.yaml      # Full sweep grid (4 models × 3 quant × 3 adapt × 2 datasets)
├── data/
│   └── hateful_memes/              # img/ + train.jsonl, dev.jsonl (test.jsonl is unlabeled)
├── paper/
│   ├── muws2026_combined.tex       # MAIN PAPER — copy/paste to Overleaf
│   ├── p1_collapse.tex             # Standalone draft 1
│   ├── p2_transfer.tex             # Standalone draft 2
│   ├── p3_hardness.tex             # Standalone draft 3
│   ├── p4_efficiency.tex           # Standalone draft 4
│   ├── refs.bib                    # Shared references
│   └── README_drafts.md
├── results/
│   └── leaderboard.jsonl           # 10 records: 6 zero-shot + 4 LoRA transfer cells
├── src/
│   ├── models.py                   # load_model(), model_footprint_mb()
│   ├── prompts.py                  # TASK_SPEC, build_messages(), score_yes_prob()
│   ├── data.py                     # load_hateful_memes(), load_mmsd2(), load_split()
│   ├── eval.py                     # evaluate_cell() → metrics + pred_pos_rate + VRAM
│   ├── benchmark.py                # benchmark_latency() → median/IQR/p10 + throughput
│   └── train_lora.py               # SFTDataset, build_example_inputs(), LoRA training loop
├── scripts/
│   ├── run_sweep.py                # Groups by (model, quant), loads once, runs all cells
│   ├── eval_ckpt.py                # Evaluate any checkpoint on any dataset
│   └── make_figures.py             # Figure generation (currently stubs)
└── requirements.txt
```

---

## Environment / Reproducibility

**Python venvs (two required — transformers version conflict):**

| venv | transformers | For |
|---|---|---|
| `venv` (main) | 4.51.3 | SmolVLM-256M, SmolVLM-500M (Idefics3-based) |
| `venv-tf5` | 5.12.1 | Gemma-4-E2B (requires `gemma4` model_type in tf5) |

**Critical torch install (CUDA 12.2 driver constraint):**
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```
torch ≥ 2.5 ships cu124+ and fails on driver 535 / CUDA 12.2.

**Run zero-shot sweep:**
```bash
source venv/bin/activate
python scripts/run_sweep.py --smoke        # quick sanity check
python scripts/run_sweep.py --only smolvlm # full SmolVLM cells
```

**Run LoRA training:**
```bash
PYTHONUNBUFFERED=1 python -u src/train_lora.py \
  --task hateful_memes --n_train 3000 --epochs 2
```

**Evaluate a checkpoint:**
```bash
python scripts/eval_ckpt.py \
  --ckpt checkpoints/smolvlm-500m-lora-hate \
  --dataset mmsd2 --train_on hateful_memes
```

All results append to `results/leaderboard.jsonl` (resumable — already-completed keys are skipped).

---

## Current Results in `leaderboard.jsonl`

10 records total:
- 6 zero-shot cells: SmolVLM-256M × 2 datasets, SmolVLM-500M × 2, Gemma-4-E2B × 2
- 4 LoRA transfer cells: SmolVLM-500M trained on hate/sarcasm × evaluated on hate/sarcasm

---

## Known Issues / Diagnostic Notes

| Issue | Cause | Fix |
|---|---|---|
| AUROC=NaN on first 250 rows | `dev.jsonl` is label-sorted | `random.Random(seed).shuffle(rows)` before limiting in `data.py` |
| Apparent F1=0.89 on n=50 | "yes"-saying model + positive-skewed slice | Use balanced n=500 + AUROC |
| Gemma "no" 100% | No chat template → flattened prompt causes "no" bias | Known; reported in paper Limitations |
| CPU-bound training | Sequential per-example image decode | Fixed: `DataLoader(num_workers=6, pin_memory=True, prefetch_factor=4)` |
| `KeyError('binary')` | `run_sweep.py` was passing `ds["task"]` not dataset name `dn` | Fixed in `run_sweep.py` |

---

## What's Left (to strengthen the paper)

| Task | Priority | What it adds |
|---|---|---|
| **Generate real figures** | High — paper has placeholder `\fbox{}` | Pareto scatter (AUROC vs. VRAM, log-x) + P(yes) histograms per model |
| **Quantization sweep** (INT8/INT4) | Medium | Original efficiency thesis; compression-degradation curve |
| **Full SmolVLM ladder** (2.2B) + Qwen2-VL-7B anchor | Medium | Strengthens Pareto claim |
| **HarMeme cross-domain** (needs `hf auth login`) | Medium | True same-task cross-domain transfer |
| **More seeds + full splits** | Low | CI bars; reviewer ask |
| **Per-example score dump** | Low (needed for reliability diagrams) | Add `--dump_scores` flag to `eval.py` |

---

## References (key bibtex keys in `refs.bib`)

`kiela2020hateful` · `qin2023mmsd2` · `cai2019multimodal` · `pramanick2021harmeme` · `hu2022lora` · `dettmers2023qlora` · `laurencon2024idefics3` · `marafioti2025smolvlm` · `gemma2025` · `guo2017calibration` · `radford2021clip`
