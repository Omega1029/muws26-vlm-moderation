# muws2026_cyber.tex

**Title:** No Cloud Required: On-Device Small Vision–Language Models for Offline Detection of Harmful Multimodal Content  
**Venue:** MUWS 2026 @ ACM Multimedia, Rio de Janeiro, Nov 10–14, 2026  
**Framing:** Endpoint / cybersecurity — offline harmful media scanner  
**Status:** Complete draft (~6–7 pages). See trimming notes at bottom of .tex if CFP enforces 4 pages.

---

## Thesis

Cloud-based content moderation fails when a device goes offline. This paper asks: can a small VLM (~0.5B parameters) serve as a practical on-device harmful-content scanner — and what are its real limits?

## Key results (all real measured numbers)

| Condition | Hate AUROC | Sarcasm AUROC | Peak VRAM |
|---|---|---|---|
| SmolVLM-256M zero-shot | 0.513 | 0.502 | ~960 MB |
| SmolVLM-500M zero-shot | 0.581 | 0.600 | ~1,450 MB |
| Gemma-4-E2B zero-shot | 0.528 | 0.527 | ~10,170 MB |
| SmolVLM-500M + LoRA-Hate | 0.619 | 0.668 | ~1,450 MB |
| SmolVLM-500M + LoRA-Sarcasm | 0.594 | 0.881 | ~1,450 MB |

**Four findings:**
1. SmolVLM-500M Pareto-dominates Gemma-4-E2B (better quality, 1/7th the VRAM)
2. Zero-shot models are broken scanners — constant-label output, F1 hides it
3. LoRA adapter fixes the scanner first, then improves it — but 7× harder for hate than sarcasm
4. Hate adapter cross-transfers to sarcasm; sarcasm adapter does not transfer back

## How to compile

1. Upload `muws2026_cyber.tex` and `refs.bib` to Overleaf
2. Menu → Compiler → **pdfLaTeX**
3. Compile twice (second pass resolves bibliography)

## Figures

- **Figure 1** (`\label{fig:pareto}`) — AUROC vs. VRAM Pareto scatter — **placeholder** `\fbox{}`
- **Figure 2** (`\label{fig:hist}`) — P(yes) score histograms per model — **placeholder** `\fbox{}`

Both can be generated from `results/leaderboard.jsonl` using `scripts/make_figures.py` (currently stubs).

## Tables

- **Table 1** (`\label{tab:zs}`) — zero-shot quality vs. cost — real numbers ✓
- **Table 2** (`\label{tab:matrix}`) — cross-category transfer matrix — real numbers ✓

## Trimming to 4 pages

See comments at the bottom of the `.tex` file. Short version: merge Findings 3+4, cut Figure 2, fold the Recommendations section into the Conclusion.
