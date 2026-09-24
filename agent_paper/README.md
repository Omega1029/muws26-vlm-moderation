# Gemma 4 Developer Agent — Paper Track starter kit

Competition: <https://www.kaggle.com/competitions/gemma-4-developer-agent-paper>
Deliverable: a **Kaggle Writeup** (markdown, on Kaggle) documenting *unpublished* research that
advances agentic software engineering with small/local models.
**Deadline: 2026-11-12, 23:59 UTC.** Drafts that are not submitted by then are not judged.

## Rules (from the competition page, 2026-09-24)

- **Deliverable:** Kaggle Writeup, **≤ 3,000 words**, original and unpublished (non-archival, so it can go to a conference later).
  It must include: **title + subtitle, abstract, introduction, methods & experiments, related work + citations**.
- **Optional:** a public notebook (Project Links) and/or an arXiv-style PDF (Public Project Link). No login or paywall.
- **Deadline:** 2026-11-12, 23:59 UTC. Press **Submit**. Saved drafts are not judged.
- **Scoring:** five criteria, each 0–5, equal weight, averaged. Ties go to whoever entered first, **so join now**.

| Criterion | What judges ask | How this paper answers it |
|---|---|---|
| **Novelty** | new insights; important properties of existing methods | resolve rate hides agent collapse; the diagnostic profile explains failures that model size doesn't |
| **Quality** | generalizes beyond the competition? | harness-agnostic metrics; report on more than one task source (SWE-bench subset + toy/other) and more than one model size |
| **Relevance** | impact on SWE + agentic learning | tells practitioners *what to fix* in local agents, and which fixes are cheapest |
| **Verifiability** | enough detail on method + data | public notebook, released trajectories, every number from `runs.jsonl` via `summarize.py` |
| **Clarity** | clear writing | one headline figure per RQ; stay well under 3,000 words |

**Prizes ($35k):** Overall Best Paper $15k · **Best New Resource** $10k (dataset/tool/software) ·
**Best New Application** $10k (new use-cases, especially of **code-graphs / the provided embeddings**).
Listed topics: PEFT/RL for SWE agents, code comprehension (code graphs, parsing, embeddings),
tasks & benchmarks, graph reasoning. Many of the judges are graph-ML researchers.

**Strategic implication:** the diagnostics harness is a natural **Resource** entry. To also be competitive
on topic fit, add the code-graph dimension: implement `get_code_neighbors` / `search_similar_code`-style
tools (AST call graph + embeddings) and measure whether graph tools *reduce collapse* (fewer blind reads
and loops, earlier first edit). That makes RQ3 a graph-reasoning result, not only a prompting one.

Main competition context (from a third-party repo's notes, **not verified**):
`gemma-4-31b-it-qat-w4a16-ct` served by vLLM on 4×L4, up to 8 LoRA adapters (rank ≤ 128),
~120 hidden SWE tasks from private repos, 12 h total budget. Score = % of patches that make the hidden tests pass.
Tools: `run_command` (300 s), `read_file` (150 lines / 10k chars), `edit_file`, `write_file`, `get_status`,
`submit_patch`, plus two code-graph tools (`get_code_neighbors`, `search_similar_code`).
Check the main competition's Data tab for the official tool specs and the provided embeddings.

## Recommended angle: bring the MUWS26 method to coding agents

The MUWS26 paper in this repo found that **small models collapse to degenerate behaviour and the headline
metric hides it**, and that **a cheap LoRA adapter fixes the collapse first, then improves quality**.
The same story, applied to local coding agents, fits this track closely and is ours to tell:

**Working title:** *Resolve Rate Is Not Enough: Diagnosing and Fixing Silent Failure in Local Gemma 4 Coding Agents*

| MUWS26 finding | Agent-paper analogue | Measured by |
|---|---|---|
| Zero-shot scanners emit a near-constant label | Small agents collapse: loop the same action, emit malformed calls, submit without testing | `dominant_tool_share`, `repeat_action_rate`, `valid_call_rate`, `no_call_rate` |
| F1/accuracy hide the collapse | Resolve rate hides *why* agents fail; two agents with equal resolve rates can fail in very different ways | diagnostics table next to `resolve_rate` |
| AUROC + pred_pos_rate as reporting standard | Propose a **minimal trajectory-diagnostics reporting standard** | `swe_agent/diagnostics.py` |
| LoRA de-biases first, then improves | Scaffold fixes (tool protocol, verify-before-submit) and a small LoRA on good trajectories fix *format/collapse* first; resolve rate follows | before/after on the same diagnostics |
| Memory budget, not params, governs viability | Resolve rate per GB and per token across E2B → E4B → 26B-A4B → 31B-QAT | `mean_tokens`, VRAM, wall time |

This counts as **empirical**, and the diagnostics harness plus the toy/SWE task converter add a
**resource** contribution too.

### Research questions
1. **RQ1: Collapse.** How often do Gemma 4 models of different sizes fail through *degenerate* behaviour (loops, malformed calls, blind submits) versus making a real attempt that is wrong?
2. **RQ2: Hidden by the metric.** Do configs with similar resolve rates differ a lot on the diagnostics? Does the diagnostic profile predict resolve rate better than model size?
3. **RQ3: Cheap fixes.** Which interventions reduce collapse per unit of cost: native vs. text tool protocol, a verify-before-submit rule, step budget, a trajectory-SFT LoRA?
4. **RQ4: Budget.** Where is the Pareto front of resolve rate against VRAM, tokens, and wall-clock for local deployment?

## Experiment plan (≈7 weeks to Nov 12)

| Week | Dates | Work | Output |
|---|---|---|---|
| 1 | Sep 24–30 | Read rules/criteria. Serve Gemma 4 E4B locally (vLLM/Ollama). Run toy tasks. Add a SWE-bench-Lite/Verified subset (~50 instances) to the task format | harness validated on real model |
| 2 | Oct 1–7 | **RQ1** sweep: E2B, E4B, 26B-A4B, 31B-QAT × native/text tools × 50 tasks × 3 seeds (temp 0.7) | `results/runs.jsonl`, first diagnostics table |
| 3 | Oct 8–14 | **RQ2** analysis: failure taxonomy (hand-label ~100 trajectories), correlate diagnostics with resolve | taxonomy figure |
| 4 | Oct 15–21 | **RQ3** scaffold interventions (verify-before-submit, loop breaker) **+ code-graph tools** (call-graph neighbours, embedding search) | ablation table |
| 5 | Oct 22–28 | **RQ3** LoRA: SFT on successful trajectories from the 31B, applied to E4B (reuse `../src/train_lora.py` patterns) | before/after table |
| 6 | Oct 29–Nov 4 | **RQ4** Pareto figure; write the full draft | draft Writeup |
| 7 | Nov 5–11 | Polish, reproducibility section, public notebook + repo link. **Submit by Nov 11** (one day buffer) | submitted Writeup |

Compute: E2B/E4B fit on one 16–24 GB GPU (Kaggle free T4×2 / P100 or Colab). 31B-QAT needs ~4×L4 or one A100-80GB (the MUWS26 A100 box works).

## Harness (`swe_agent/`)

Dependency-free Python (stdlib only). Talks to any OpenAI-compatible endpoint.

```
swe_agent/llm.py          OpenAICompatLLM (vLLM / Ollama / llama.cpp) + ScriptedLLM for tests
swe_agent/tools.py        competition-shaped tools with the same caps; path-escape guard
swe_agent/agent.py        agent loop; native tool_calls AND text-JSON fallback; every step logged
swe_agent/diagnostics.py  per-episode + aggregate diagnostics (the paper's core measurement)
swe_agent/tasks.py        task JSONL, fresh checkouts, hidden-test verification (applies diff to a clean copy)
scripts/run_agent.py      resumable runner → results/runs.jsonl + per-episode trajectories
scripts/summarize.py      one markdown/CSV row per experiment tag
tasks/toy.jsonl           3 planted-bug tasks with hidden tests (for smoke-testing a model/server)
tests/test_smoke.py       end-to-end tests with a scripted LLM (no GPU)
```

**Safety:** `run_command` runs a local subprocess with **no sandbox**. Toy tasks are harmless. For real
repos, run inside a container or VM.

### Quickstart

```bash
cd agent_paper
python -m unittest discover -s tests -v          # harness sanity, no GPU

# serve a Gemma 4 model. Check the vLLM docs for the current Gemma 4 tool-call parser name
vllm serve google/gemma-4-E4B-it --enable-auto-tool-choice --tool-call-parser <gemma4-parser>
#   or: ollama pull gemma4:e4b && ollama serve   (base_url http://localhost:11434/v1)

python scripts/run_agent.py --tasks tasks/toy.jsonl --model google/gemma-4-E4B-it --tag e4b-native
python scripts/run_agent.py --tasks tasks/toy.jsonl --model google/gemma-4-E4B-it --tag e4b-text --no_native_tools
python scripts/summarize.py
```

Model IDs above are placeholders. Use the exact names from the Hugging Face / Kaggle Gemma 4 model pages.

### Adding SWE-bench tasks (week 1 TODO)
For each instance: clone `repo` at `base_commit` into `tasks/swebench/<instance_id>/`, apply the
environment setup, and write a JSONL row. Set `hidden_tests` to the files touched by `test_patch`
(extract them after applying it to a scratch copy), and `test_command` to the `FAIL_TO_PASS` tests.
Start with ~50 pure-Python instances that have light dependencies.

## Diagnostics reference

| Metric | Meaning | Failure it exposes |
|---|---|---|
| `resolve_rate` | hidden tests pass on a clean checkout + patch | headline |
| `false_submit_rate` | submitted, but wrong | confident failure |
| `verified_before_submit_rate` | ran a command after the last edit, then submitted | blind submits |
| `empty_patch_rate` | no diff at all | never acted |
| `valid_call_rate` | tool calls that parsed and executed | format collapse |
| `text_fallback_rate` / `no_call_rate` | ignored native tools / emitted no call | protocol mismatch |
| `dominant_tool_share` | share of steps using the single most-used tool | action collapse (≈ pred_pos_rate) |
| `repeat_action_rate` | share of (tool, args) repeats | loops |
