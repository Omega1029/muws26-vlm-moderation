# CLAUDE.md

This repo has two research projects. **The active one is `agent_paper/`**, the Kaggle
*Gemma 4 Developer Agent Paper Track* entry (deadline **2026-11-12, 23:59 UTC**).
The MUWS26 VLM-moderation paper (`src/`, `scripts/`, `paper/`) was submitted. Treat it as reference
material and prior work, and don't change it unless asked.

## Active project: agent_paper/

**Goal:** a ≤ 3,000-word Kaggle Writeup, *"Resolve Rate Is Not Enough: Diagnosing and fixing silent
failure in local Gemma 4 coding agents"*, plus a public notebook. Plan, rules, rubric and timeline are in
`agent_paper/README.md`. The draft is in `agent_paper/writeup/WRITEUP.md`. Read both before starting a task.

Judged on Novelty, Quality (generalization), Relevance, Verifiability and Clarity, each 0–5, equal weight.
Prize tracks: Best Paper, **Best New Resource** (our harness fits) and Best New Application
(code-graphs / provided embeddings). Judges favour code-graph work, so the planned code-graph-tools
ablation (RQ3) is a priority.

### Layout
```
agent_paper/swe_agent/llm.py          OpenAI-compatible client (vLLM/Ollama) + ScriptedLLM
agent_paper/swe_agent/tools.py        competition-shaped tools and caps (TOOL_SCHEMAS, Workspace)
agent_paper/swe_agent/agent.py        agent loop: native tool_calls + text-JSON fallback, per-step log
agent_paper/swe_agent/diagnostics.py  episode_metrics() / aggregate(): the paper's core measurement
agent_paper/swe_agent/tasks.py        task JSONL, fresh_checkout(), verify() with hidden tests
agent_paper/scripts/run_agent.py      resumable runner → results/runs.jsonl + results/trajectories/<tag>/
agent_paper/scripts/summarize.py      one table row per --tag
agent_paper/tasks/toy.jsonl           3 planted-bug tasks (protocol smoke test)
agent_paper/tests/test_smoke.py       scripted-LLM end-to-end tests
```

### Commands (run from `agent_paper/`)
```bash
python -m unittest discover -s tests -v                       # must pass before every commit
python scripts/run_agent.py --tasks tasks/toy.jsonl --model <id> --base_url http://localhost:8000/v1 --tag <cell>
python scripts/run_agent.py ... --no_native_tools --tag <cell>-text   # tool-protocol ablation
python scripts/run_agent.py ... --temperature 0.7 --seeds 3           # variance / CIs
python scripts/summarize.py [--csv out.csv]
```
Serving a model locally (check exact model IDs and the vLLM Gemma 4 tool-call parser name in current docs;
don't guess):
```bash
vllm serve <gemma-4 model id> --enable-auto-tool-choice --tool-call-parser <parser>   # :8000/v1
ollama pull <gemma4 tag> && ollama serve                                              # :11434/v1
```
Before a long sweep, run `nvidia-smi` and check which model fits. E2B/E4B fit on a single consumer GPU;
31B-QAT needs ~4×L4 or an A100-80GB.

### Conventions
- The harness is **stdlib only**. Don't add pip dependencies to `swe_agent/` without asking. Analysis and plotting
  scripts may use matplotlib/pandas/scikit-learn.
- Match the existing style: module docstring stating purpose + usage, `from __future__ import annotations`,
  compact functions, comments only where the *why* isn't obvious.
- New agent behaviours (loop breaker, verify-before-submit, code-graph tools) go behind CLI flags so every
  change is an ablation cell with its own `--tag`. Never change the default behaviour silently, because that breaks
  comparability with earlier runs.
- A new diagnostic goes in `diagnostics.py` with a unit test in `tests/test_smoke.py`.
- Keep tool caps matching the competition (`READ_MAX_LINES=150`, `READ_MAX_CHARS=10_000`, `CMD_TIMEOUT_S=300`).

### Experiments and results integrity
- `agent_paper/results/` is gitignored. Results live on the machine. Back up `runs.jsonl` and trajectories
  before deleting anything, and never overwrite or hand-edit `runs.jsonl`. It is append-only, keyed by
  (tag, instance_id, seed).
- Every number in the Writeup must come from `summarize.py` output or a committed analysis script. Never type
  numbers in by hand, and never report a result that wasn't actually run. If a sweep crashed or was partial, say so.
- Tag names encode the cell: `<model>-<protocol>-<intervention>-t<temp>`, e.g. `e4b-native-graph-t0.7`.
- Report resolve rate **together with** the diagnostics, with n and seeds. That pairing is the paper's thesis.

### Safety
`run_command` executes **unsandboxed** on this machine. Toy tasks are fine. For SWE-bench or any real repo, run
inside Docker or a VM, and never point the agent at a directory outside a throwaway checkout. Don't run an
agent sweep in the repo root.

### Writeup rules
- Hard limit **3,000 words** (check with `wc -w writeup/WRITEUP.md` minus the instruction block).
- Required sections: title + subtitle, abstract, introduction, methods & experiments, related work + citations.
- Must be original and unpublished. MUWS26 is cited as prior work; don't reuse its text.
- Public notebook (Kaggle) with no login wall. It goes in Project Links.

## Reference project: MUWS26 (VLM moderation)
Needs two venvs (transformers 4.51 for SmolVLM, ≥5 for Gemma 4) and torch 2.4.1 cu121. See `README.md`.
`src/train_lora.py` is the template for the planned trajectory-SFT LoRA (answer-token-only loss, prompt masked with -100).

## Git
- Work on a feature branch. Commit with clear messages. Don't commit results, checkpoints, model weights or `.env`.
- Run the unit tests before committing.
