# p4_efficiency.tex

**Title:** Bigger Is Not Better: A Parameter–Performance Pareto Analysis of Small Vision–Language Models for Multimodal Hate and Sarcasm Detection  
**Venue:** MUWS 2026 @ ACM Multimedia, Rio de Janeiro, Nov 10–14, 2026  
**Framing:** Efficiency / on-device deployment  
**Status:** Complete draft. Clean, direct efficiency story — works well as a short paper.

---

## Thesis

At the small-model scale relevant to on-device deployment, parameter count is a poor predictor of multimodal harmful-content detection quality. A 0.5B model Pareto-dominates a 2B model — higher detection quality at 1/7th the memory and lower latency. Cheap LoRA adaptation, not scale, is what moves the frontier.

## Key result — zero-shot Pareto (fp16, n=500, A100)

| Model | Hate AUROC | Sarcasm AUROC | Peak VRAM | Latency |
|---|---|---|---|---|
| SmolVLM-256M | 0.513 | 0.502 | ~960 MB | 70 ms |
| **SmolVLM-500M** | **0.581** | **0.600** | ~1,450 MB | 69 ms |
| Gemma-4-E2B | 0.528 | 0.527 | ~10,170 MB | 86 ms |

SmolVLM-500M **Pareto-dominates** Gemma-4-E2B: better AUROC on both tasks, ~1/7th the VRAM, lower latency.

**LoRA (SmolVLM-500M only, negligible inference overhead):**

| Condition | Hate AUROC | Sarcasm AUROC |
|---|---|---|
| zero-shot | 0.581 | 0.600 |
| + LoRA in-domain | 0.619 | 0.881 |

LoRA moves the frontier far beyond what scaling to 2B achieves.

## How to compile

1. Upload `p4_efficiency.tex` and `refs.bib` to Overleaf
2. Menu → Compiler → **pdfLaTeX**
3. Compile twice

## Figures

- **Figure 1** (`\label{fig:pareto}`) — AUROC (y) vs. peak VRAM (x, log scale) scatter; Pareto frontier dashed; star = SmolVLM-500M + LoRA — **placeholder** `\fbox{}`

Can be generated directly from `results/leaderboard.jsonl` using `scripts/make_figures.py` (currently a stub — needs ~20 lines of matplotlib).

## Tables

- **Table 1** (`\label{tab:pareto}`) — zero-shot quality vs. cost — real numbers ✓
- **Table 2** (`\label{tab:lora}`) — LoRA on best small model — real numbers ✓

## Extension opportunities

- Add INT8/INT4 quantization points (bitsandbytes) to extend the Pareto frontier leftward along the memory axis — the original efficiency thesis
- Add SmolVLM-2.2B and Qwen2-VL-7B as additional ladder points
- Plot VRAM vs. latency separately to show the two-dimensional cost surface
