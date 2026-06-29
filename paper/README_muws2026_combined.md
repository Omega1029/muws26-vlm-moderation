# muws2026_combined.tex

**Title:** Scale, Collapse, Adaptation, Transfer: What Governs Small Vision–Language Models on Multimodal Hate and Sarcasm Detection  
**Venue:** MUWS 2026 @ ACM Multimedia, Rio de Janeiro, Nov 10–14, 2026  
**Framing:** Academic / empirical analysis  
**Status:** Complete draft (~6–7 pages). See trimming notes at bottom of .tex if CFP enforces 4 pages.

---

## Thesis

At the small-model scale (≤2B parameters), raw parameter count is a poor predictor of multimodal harmful-content detection quality. Four interlocking findings characterise what actually governs behaviour: scale, measurement, adaptation, and cross-task transfer.

## Key results (all real measured numbers)

| Condition | Hate AUROC | Sarcasm AUROC | Peak VRAM |
|---|---|---|---|
| SmolVLM-256M zero-shot | 0.513 | 0.502 | ~960 MB |
| SmolVLM-500M zero-shot | 0.581 | 0.600 | ~1,450 MB |
| Gemma-4-E2B zero-shot | 0.528 | 0.527 | ~10,170 MB |
| SmolVLM-500M + LoRA-Hate | 0.619 | 0.668 | ~1,450 MB |
| SmolVLM-500M + LoRA-Sarcasm | 0.594 | 0.881 | ~1,450 MB |

**Four findings:**
1. **Scale not enough** — SmolVLM-500M Pareto-dominates Gemma-4-E2B
2. **Collapse + metric failure** — zero-shot models emit near-constant labels; F1 on skewed slices is a pure artifact (apparent F1=0.89 on n=50 → AUROC=0.50 on balanced n=500)
3. **Adaptation gap** — LoRA gains: sarcasm +0.28 vs. hate +0.04 (~7×); both tasks converge to train loss ~0.1
4. **Asymmetric transfer** — hate→sarcasm positive; sarcasm→hate essentially zero

## How to compile

1. Upload `muws2026_combined.tex` and `refs.bib` to Overleaf
2. Menu → Compiler → **pdfLaTeX**
3. Compile twice (second pass resolves bibliography)

## Figures

- **Figure 1** (`\label{fig:pareto}`) — accuracy–memory Pareto scatter — **placeholder** `\fbox{}`
- **Figure 2** (`\label{fig:hist}`) — P(yes) score histograms — **placeholder** `\fbox{}`

Both can be generated from `results/leaderboard.jsonl` using `scripts/make_figures.py`.

## Tables

- **Table 1** (`\label{tab:zs}`) — zero-shot quality vs. cost — real numbers ✓
- **Table 2** (`\label{tab:matrix}`) — cross-task transfer matrix — real numbers ✓

## Trimming to 4 pages

See comments at the bottom of the `.tex` file: merge §6+§7, cut a figure, compress Related Work, drop the itemized contributions list in the intro.

## Relationship to other drafts

This is the unified paper. The four `p*.tex` files are standalone single-finding versions — useful if you want to submit one tightly scoped story rather than the four-finding combined paper.
