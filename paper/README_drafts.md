# MUWS 2026 — four paper drafts (copy/paste into Overleaf)

All numbers in these drafts are the **real measured results** from our runs
(`results/leaderboard.jsonl`). Figures are placeholders (`\fbox{...}`) with TODO comments
pointing at what to plot. Each `.tex` is a standalone `acmart` paper.

## How to use in Overleaf
1. New Project → Blank Project.
2. Upload (or paste) the chosen `pN_*.tex` **and** `refs.bib`.
3. Menu → Compiler: **pdfLaTeX**. Compile (Overleaf auto-runs BibTeX).
4. For camera-ready, remove the `nonacm` option and fill the ACM rights block.

## The four framings (decoupled from the original "efficiency" thesis)
| file | thesis | readiness |
|---|---|---|
| `p1_collapse.tex` | **Small zero-shot VLMs collapse to a constant label; thresholded metrics hide it; LoRA mainly de-biases.** Measurement/responsible-evaluation paper. | **Strongest** — experiments essentially done; needs Fig.1 (score histograms + reliability) + prose. |
| `p2_transfer.tex` | Hate→sarcasm transfer is positive; sarcasm→hate is not. Asymmetric cross-task transfer. | Solid; thin alone (2×2). |
| `p3_hardness.tex` | Matched-budget LoRA: sarcasm +0.28 AUROC vs hate +0.04 (~7×). Fitting≠generalisation; HM confounders. | Solid; wants qualitative error examples. |
| `p4_efficiency.tex` | Bigger≠better: SmolVLM-500M Pareto-dominates Gemma-4-E2B (7× less memory); LoRA, not scale, moves the frontier. | Leaner efficiency story; no quantization needed. |

## Shared caveat (in every draft's Limitations)
2 datasets, ≤3 models, single seed, n=500 balanced slices (CI ≈ ±0.03–0.04). These are
focused **workshop short-paper** scope, not a flagship study.

## To turn placeholders into real figures
Score histograms / reliability / Pareto scatter can all be generated from
`results/leaderboard.jsonl` (extend `scripts/make_figures.py`). Per-example scores are not
yet persisted — add a flag to `eval.py` to dump them if you want reliability diagrams.
