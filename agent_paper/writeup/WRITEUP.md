# Resolve Rate Is Not Enough: Diagnosing and Fixing Silent Failure in Local Gemma 4 Coding Agents

> Kaggle Writeup draft. Paste into the Writeup editor on the competition page. Replace every `TODO`.
> Paper type: **empirical** (plus a resource: the open diagnostics harness).
> Map each section to the five judging criteria once you've copied them from the Evaluation tab.

## TL;DR
TODO (3 sentences): Small local coding agents fail silently. Resolve rate can't tell a model that tries
and misses apart from one that loops, emits malformed calls, or submits without testing. We measure these
failure modes across the Gemma 4 family, show they are common and that resolve rate hides them, and find
that cheap scaffold and LoRA fixes remove the collapse first. Resolve-rate gains follow.

## 1. Motivation
- Frontier coding agents need cloud models; many developers can't use them (cost, privacy, air-gapped work).
- Small local models (Gemma 4 E2B/E4B/26B-A4B/31B-QAT) are now close, but they fail at multi-turn SWE work.
- Resolve rate alone gives no guidance on *what to fix*. TODO: cite an example where two configs tie on resolve rate but fail differently.
- Prior evidence of this pattern: our MUWS26 study of small VLMs, where headline metrics hid label collapse.

## 2. Contributions
1. A trajectory-diagnostics suite (8 metrics) plus an open, dependency-free harness that mirrors the competition's tools.
2. An empirical map of failure modes across 4 Gemma 4 sizes × 2 tool protocols × N tasks × 3 seeds.
3. Evidence that the diagnostics predict / explain resolve rate better than model size (TODO: confirm).
4. A cost-ranked set of interventions (scaffold rules, trajectory LoRA) with before/after diagnostics.

## 3. Setup
- **Models:** TODO exact IDs, quantization, serving stack (vLLM version), hardware.
- **Tasks:** TODO SWE-bench subset (n, selection criteria), plus toy tasks for protocol checks.
- **Agent:** tools and caps (run_command 300 s, read_file 150 lines/10k chars, edit/write, submit), max steps, temperature, seeds.
- **Verification:** agent diff applied to a fresh checkout; hidden tests added; pass/fail.

## 4. RQ1: How do small agents fail?
TODO table: `scripts/summarize.py` output per model × protocol.
TODO figure: stacked bar per model: resolved / wrong-but-verified / blind submit / loop-out / malformed / empty.

## 5. RQ2: What the metric hides
TODO: pairs of configs with similar resolve rate and different diagnostic profiles.
TODO: logistic regression of `resolved` on diagnostics vs. on model size (report AUROC for each).

## 6. RQ3: Cheap fixes, ranked by cost
| Intervention | Cost | Δ valid_call | Δ repeat_action | Δ false_submit | Δ resolve |
|---|---|---|---|---|---|
| text-JSON vs native tools | 0 | TODO | | | |
| verify-before-submit rule | 0 | | | | |
| loop breaker (reject repeated action) | 0 | | | | |
| trajectory-SFT LoRA (r=16) | ~1 GPU-h | | | | |

## 7. RQ4: Budget
TODO Pareto figure: resolve rate vs. peak VRAM (log-x), marker size = mean tokens.

## 8. Limitations
TODO: task subset size, public-repo contamination risk, single scaffold, sandbox differences from the official evaluator.

## 9. Reproducibility
Code: TODO public repo / Kaggle notebook link. Every number comes from `results/runs.jsonl` via `scripts/summarize.py`.
Trajectories released in `results/trajectories/`.

## References
TODO: SWE-bench (Jimenez et al. 2024), SWE-agent (Yang et al. 2024), Gemma 4 tech report, LoRA (Hu et al. 2022), MUWS26 (ours).
