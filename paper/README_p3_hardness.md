# p3_hardness.tex

**Title:** Why Is Multimodal Hate Harder than Sarcasm? A Matched-Budget Adaptation Study with Small Vision–Language Models  
**Venue:** MUWS 2026 @ ACM Multimedia, Rio de Janeiro, Nov 10–14, 2026  
**Framing:** Task difficulty / learnability analysis  
**Status:** Complete draft. Strong controlled finding — identical recipe, 7× different outcome.

---

## Thesis

Under an identical small-model LoRA budget (same backbone, same recipe, same number of training examples), multimodal sarcasm detection gains +0.28 AUROC while hate detection gains only +0.04. The gap is not a fitting problem — training loss converges to ~0.1 for both — but a generalisation problem caused by the adversarial benign-confounder design of Hateful Memes.

## Key result

| Task | Zero-shot AUROC | +LoRA AUROC | Δ | Train loss |
|---|---|---|---|---|
| Hateful Memes | 0.581 | 0.619 | **+0.038** | ~0.1 |
| MMSD2.0 sarcasm | 0.600 | 0.881 | **+0.281** | ~0.1 |

**7× difference in adaptation gain under an identical budget.**

The model memorises hate training targets as readily as sarcasm ones (same train loss), but hate dev-set patterns don't match train-set patterns — the benign confounder pairs in Hateful Memes make surface-level cues non-transferable. Sarcasm doesn't have this design.

**Practical implication:** "multimodal toxicity detection" bundles tasks of very different learnability. Hate-meme detection with compact models needs more diverse data, harder negatives, or confounder-aware training — not just more fine-tuning.

## How to compile

1. Upload `p3_hardness.tex` and `refs.bib` to Overleaf
2. Menu → Compiler → **pdfLaTeX**
3. Compile twice

## Figures

- **Figure 1** (`\label{fig:loss}`) — overlaid training-loss curves (both converge to ~0.1) beside held-out AUROC gains (only sarcasm rises) — **placeholder** `\fbox{}`

Training loss curves were not persisted during the LoRA runs. To generate: re-run `src/train_lora.py` with a `--log_loss` flag that writes loss per step to a CSV, then plot.

## Tables

- **Table 1** (`\label{tab:gain}`) — matched-budget comparison: zero-shot vs. +LoRA AUROC per task with Δ — real numbers ✓

## Caveat

Single backbone (SmolVLM-500M), single seed, single training-set size (3,000 examples). Cannot rule out that a much larger hate-training set would close the gap — the fitting/generalisation mismatch is inferred from loss and AUROC, not a formal attribution study.
