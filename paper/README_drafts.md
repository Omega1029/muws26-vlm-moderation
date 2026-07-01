# MUWS 2026 — four paper drafts (copy/paste into Overleaf)

All numbers in these drafts are the **real measured results** from our runs
(`results/leaderboard.jsonl`). Figures are placeholders (`\fbox{...}`) with TODO comments
pointing at what to plot. Each `.tex` is a standalone `acmart` paper.

## How to use in Overleaf
1. New Project → Blank Project.
2. Upload (or paste) the chosen `pN_*.tex` **and** `refs.bib`.
3. Menu → Compiler: **pdfLaTeX**. Compile (Overleaf auto-runs BibTeX).
4. For camera-ready, remove the `nonacm` option and fill the ACM rights block.

## Consolidated slate (two papers)
The four single-finding framings (`p1_collapse`, `p2_transfer`, `p3_hardness`, `p4_efficiency`) were
merged into `muws2026_combined.tex`, which already unifies all four findings (scale / collapse /
adaptation / transfer) into one analysis paper. Two manuscripts remain:

| file | target venue | thesis |
|---|---|---|
| `muws2026_combined.tex` | **AAAI 2027** | Scale, Collapse, Adaptation, Transfer — the unified four-finding analysis of small VLMs on multimodal hate/sarcasm. |
| `muws2026_cyber.tex` | **ICLR 2027** | Same experiments reframed as on-device offline harmful-content scanning (endpoint-security angle). |

The four findings, for reference: **(collapse)** small zero-shot VLMs collapse to a constant label
and thresholded metrics hide it; **(transfer)** hate→sarcasm transfer is positive, sarcasm→hate is
not; **(hardness)** matched-budget LoRA gives sarcasm +0.28 AUROC vs hate +0.04 (~7×); **(efficiency)**
SmolVLM-500M Pareto-dominates Gemma-4-E2B at ~7× less memory.

## Shared caveat (in every draft's Limitations)
2 datasets, ≤3 models, single seed, n=500 balanced slices (CI ≈ ±0.03–0.04). These are
focused **workshop short-paper** scope, not a flagship study.

## To turn placeholders into real figures
Score histograms / reliability / Pareto scatter can all be generated from
`results/leaderboard.jsonl` (extend `scripts/make_figures.py`). Per-example scores are not
yet persisted — add a flag to `eval.py` to dump them if you want reliability diagrams.
