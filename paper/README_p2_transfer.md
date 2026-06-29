# p2_transfer.tex

**Title:** Hate Helps Sarcasm, but Not the Reverse: Asymmetric Cross-Task Transfer in Small Multimodal Models  
**Venue:** MUWS 2026 @ ACM Multimedia, Rio de Janeiro, Nov 10–14, 2026  
**Framing:** Cross-task transfer / low-resource adaptation  
**Status:** Complete draft. Solid result; thin alone (2×2 matrix) — works best combined with p3 or as part of the combined paper.

---

## Thesis

LoRA adapters trained on hateful-meme detection transfer positively to sarcasm detection, but sarcasm adapters do not transfer back to hate. The asymmetry is consistent and explainable: hate fine-tuning instils a general image–text grounding behaviour that benefits any task requiring joint vision-language reasoning.

## Key result — full transfer matrix (SmolVLM-500M, fp16, n=500)

| Adapter trained on | Hate AUROC | Sarcasm AUROC |
|---|---|---|
| none (zero-shot) | 0.581 (pred_pos=0.93) | 0.600 (pred_pos=0.98) |
| Hateful Memes | **0.619** (0.36) | **0.668** (0.61) |
| MMSD2.0 sarcasm | 0.594 (0.65) | **0.881** (0.29) |

- Hate adapter raises sarcasm from 0.600 → **0.668** (above both the sarcasm zero-shot floor *and* the hate in-domain score of 0.619)
- Sarcasm adapter moves hate from 0.581 → 0.594 (within noise)
- Both adapters de-bias the model (`pred_pos` moves toward base rate) — but only hate cross-transfers meaningfully

**Practical implication:** when labelled data for a new content category is scarce, a hate-detection adapter is a better bootstrap than a sarcasm one or zero-shot.

## How to compile

1. Upload `p2_transfer.tex` and `refs.bib` to Overleaf
2. Menu → Compiler → **pdfLaTeX**
3. Compile twice

## Figures

- **Figure 1** (`\label{fig:bars}`) — AUROC grouped by source adapter, highlighting hate→sarcasm cross-transfer — **placeholder** `\fbox{}`

Can be generated as a grouped bar chart from `results/leaderboard.jsonl`.

## Tables

- **Table 1** (`\label{tab:matrix}`) — full 3×2 transfer matrix with pred_pos — real numbers ✓

## Caveat

The matrix is 2×2 tasks on a single backbone and single seed. The asymmetry is consistent but should be confirmed with more seeds or a third task (e.g. HarMeme) to be fully convincing as a standalone paper.
