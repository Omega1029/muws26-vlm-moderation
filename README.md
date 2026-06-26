# MUWS 2026 — On-Device Small VLM Harmful Content Detection

**Paper:** *No Cloud Required: On-Device Small Vision–Language Models for Offline Detection of Harmful Multimodal Content*  
**Workshop:** MUWS 2026 @ ACM Multimedia, Rio de Janeiro, Nov 10–14, 2026  
**Submission deadline:** July 16, 2026

---

## What this is

A controlled empirical study of small VLMs (≤2B parameters) as **offline, on-device harmful content scanners** — the kind needed by parental-control apps, air-gapped corporate endpoints, or school device management where there is no cloud connection. Hash-based tools (PhotoDNA etc.) only catch known contraband; semantic VLM-based scanning generalises to novel harmful content.

### Four findings

1. **Memory budget, not parameter count, governs viability** — SmolVLM-500M (~1.5 GB VRAM) Pareto-dominates Gemma-4-E2B (~10 GB) on detection quality at 1/7th the cost
2. **Zero-shot models are broken scanners** — they emit a near-constant label (SmolVLM: 86–98% "yes"; Gemma: 100% "no"), and standard F1/accuracy metrics hide this failure completely
3. **A cheap LoRA policy adapter (~2% extra params) fixes the scanner first, then improves it** — but gains differ 7× by content category (sarcasm +0.28 AUROC vs. hate +0.04)
4. **Cross-category transfer is asymmetric** — a hate adapter bootstraps sarcasm detection (useful for low-resource deployment); the reverse does not hold

## Results

All results in `results/leaderboard.jsonl`. Key numbers (fp16, n=500, A100):

| Model | Hate AUROC | Sarcasm AUROC | Peak VRAM |
|---|---|---|---|
| SmolVLM-256M (zero-shot) | 0.513 | 0.502 | ~960 MB |
| SmolVLM-500M (zero-shot) | 0.581 | 0.600 | ~1,450 MB |
| Gemma-4-E2B (zero-shot) | 0.528 | 0.527 | ~10,170 MB |
| SmolVLM-500M + LoRA-Hate | **0.619** | 0.668 | ~1,450 MB |
| SmolVLM-500M + LoRA-Sarcasm | 0.594 | **0.881** | ~1,450 MB |

## Paper files (Overleaf-ready)

| File | Description |
|---|---|
| `paper/muws2026_cyber.tex` | **Main paper** — offline/endpoint security framing |
| `paper/muws2026_combined.tex` | Combined paper — original academic framing |
| `paper/p1_collapse.tex` | Standalone: zero-shot collapse + metric failure |
| `paper/p2_transfer.tex` | Standalone: asymmetric cross-task transfer |
| `paper/p3_hardness.tex` | Standalone: 7× difficulty gap hate vs. sarcasm |
| `paper/p4_efficiency.tex` | Standalone: Pareto/efficiency story |
| `paper/refs.bib` | Shared bibliography |

Upload any `.tex` + `refs.bib` to Overleaf → Compiler: pdfLaTeX → compile twice.

## Repo layout

```
configs/experiment_matrix.yaml   full sweep grid (models × quant × adaptation × datasets)
src/models.py                    quantized model loading (bitsandbytes fp16/int8/int4)
src/prompts.py                   task prompts + P(yes) AUROC scoring
src/data.py                      Hateful Memes + MMSD2.0 loaders
src/eval.py                      evaluate one cell → metrics + pred_pos_rate + VRAM
src/benchmark.py                 latency protocol (warmup, median+IQR, throughput)
src/train_lora.py                LoRA SFT training loop (GPU-bound DataLoader)
scripts/run_sweep.py             grouped sweep runner → appends to leaderboard.jsonl
scripts/eval_ckpt.py             evaluate any checkpoint on any dataset
scripts/make_figures.py          figure generation (Pareto scatter, score histograms)
results/leaderboard.jsonl        one JSONL record per completed cell (resumable)
paper/                           all LaTeX drafts + refs.bib
```

## Reproducing

**Two venvs are required** (transformers version conflict between SmolVLM and Gemma):

```bash
# Main venv — SmolVLM-256M / 500M
python -m venv venv && source venv/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Sidecar venv — Gemma-4-E2B (needs transformers ≥5)
python -m venv venv-tf5 && source venv-tf5/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=5.0" accelerate peft datasets Pillow
```

> **torch must be cu121** — torch ≥2.5 ships cu124+ and fails on driver 535 / CUDA 12.2.

**Run zero-shot sweep:**
```bash
source venv/bin/activate
python scripts/run_sweep.py --smoke          # sanity check (smoke cells only)
python scripts/run_sweep.py --only smolvlm   # full SmolVLM cells
```

**LoRA training:**
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

## Datasets

- **Hateful Memes** — download from Meta AI; place images + jsonl files in `data/hateful_memes/`; set `HM_ROOT` env var
- **MMSD2.0** — loaded automatically from HuggingFace Hub: `coderchen01/MMSD2.0`, config `mmsd-v2`

## Hardware

NVIDIA A100 SXM4-80GB · CUDA 12.2 · driver 535
