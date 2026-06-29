# p1_collapse.tex

**Title:** Broken Scorers and Misleading Metrics: Label-Prior Collapse in Zero-Shot Small Vision–Language Models  
**Venue:** MUWS 2026 @ ACM Multimedia, Rio de Janeiro, Nov 10–14, 2026  
**Framing:** Measurement / responsible evaluation  
**Status:** Complete draft. Strongest standalone paper — the finding is clean and self-contained.

---

## Thesis

Zero-shot small VLMs (≤2B) do not behave as classifiers on multimodal harmful-content tasks — they emit a near-constant label regardless of input. Standard thresholded metrics (F1, accuracy) conceal this completely. AUROC and a prediction-distribution diagnostic are the minimum honest reporting standard.

## Key result

- SmolVLM-256M: answers "yes" for **98%** of all inputs (`pred_pos_rate = 0.982`)
- SmolVLM-500M: answers "yes" for **93–98%** of all inputs
- Gemma-4-E2B: answers "no" for **100%** of all inputs
- On a positive-skewed n=50 slice: apparent **F1 = 0.89** for SmolVLM-256M
- Same model on balanced n=500: **AUROC = 0.50**, accuracy below majority baseline

LoRA fine-tuning fixes the collapse: `pred_pos_rate` moves from ~0.95 to the true base rate (0.36 hate, 0.29 sarcasm) — but this is the *first-order* effect, not genuine class separation.

## How to compile

1. Upload `p1_collapse.tex` and `refs.bib` to Overleaf
2. Menu → Compiler → **pdfLaTeX**
3. Compile twice

## Figures

- **Figure 1** (`\label{fig:hist}`) — P(yes) score histograms per model showing mass collapsed near 0 or 1 + reliability diagram — **placeholder** `\fbox{}`

To generate: add a `--dump_scores` flag to `src/eval.py` to persist per-example P(yes) scores, then plot histograms from those.

## Tables

- **Table 1** (`\label{tab:collapse}`) — pred_pos_rate + AUROC per model, zero-shot and +LoRA — real numbers ✓

## Why this is the strongest standalone

The finding requires no extra experiments beyond zero-shot eval. The F1 artifact example is concrete and reproducible. The reporting recommendation (AUROC + pred_pos) is actionable. Workshop audiences respond well to "here is a metric failure mode you should stop doing."
