# CodeReviewEnv v3 — Environment Specification

> One-page formal specification of the OpenEnv environment that backs this submission.
> Maps directly to the criteria in OpenEnv Hackathon Self-Serve Guide §4 *(environment as
> first-class artifact)* and §8 *(prevention against reward hacking)*.

---

## 1. Substrate

| Property | Value |
|---|---|
| Framework | [OpenEnv](https://github.com/meta-pytorch/OpenEnv) `0.2.3` + [FastMCP](https://github.com/jlowin/fastmcp) |
| Episode dataset | 150 CVE-investigation episodes, 92 with at least one bug (61%), 8 difficulty bands |
| Code snippets indexed | 2,869 across all episodes |
| Tools exposed | 4 MCP tools (see §3) |
| Server | FastAPI + Uvicorn, deployed as a HuggingFace Space |
| Source file | [`environment.py`](code_review_env/server/environment.py) (~440 lines) |

The env follows the canonical OpenEnv contract: `reset()` → `step(action)` → `state()` → `reward`.

---

## 2. Observation space

Every observation returned by `reset()` or `step()` is a single string with
deterministic structure:

```
[Investigation step k]
<tool output OR system message>
[Budget: B investigation points remaining | Flags: f/F]
```

- **Investigation step k**: monotonically increasing integer, anti-loop signal.
- **Tool output**: structured text from one of the 4 tools.
- **Budget line**: explicit budget telemetry. The agent always knows what it has left,
  so the policy is never punished for running out without warning.

The first observation also includes the CVE description, repository name, file list, and
risk features (churn, complexity, TODO count, recency, language). This is the entire
context the agent operates from — there is no hidden state.

---

## 3. Action space

The agent calls one of 4 MCP tools per step:

| Tool | Args | Effect | Cost |
|---|---|---|---:|
| `read_file(filename)` | filename | Returns full file contents (truncated to 4 KB) and risk summary | 1 invest. point |
| `search_pattern(pattern, filename?)` | regex, optional filename | Returns matching lines across files | 1 invest. point |
| `flag_vulnerability(filename, reason)` | filename, justification | Marks a file as bug-bearing in the agent's report | counts toward flag budget `F` |
| `submit_report()` | — | Terminates the episode with the current flag set | terminates |

Action validation (in `_step_impl`):

- Unknown tool name → terminating error response (no soft-fail), `reward=0`, `done=True`.
- Invalid filename → returns "file not found" with current state preserved (no point cost).
- Malformed JSON args → tool error message, `reward=0` for the step but episode continues.

---

## 4. Budgets and termination conditions

**Two independent budgets keep behaviour bounded:**

1. **Flag budget `F`** = `min(num_files, max(2·bugs+3, 5))` per episode.
   Prevents the trivial "flag everything" attack — flagging every file gives positives
   for free but also caps the F1-numerator. F1 normalisation makes this strictly
   dominated by selective flagging.

2. **Investigation budget `B`** = `2 · num_files`. Each `read_file` / `search_pattern` call
   costs 1 point. `flag_vulnerability` and `submit_report` are free. Once `B` reaches 0,
   further investigation tools return `"WARNING: Investigation budget exhausted. Submit
   your report."` and the agent must submit.

3. **Episode terminates on:**
   - Explicit `submit_report()` call.
   - Investigation budget exhausted **AND** any further non-`submit` action attempted.
   - `step_count` exceeds `len(files) · 3` (hard cap to prevent infinite loops if the
     model emits malformed tool JSON forever).

All three termination paths produce a final reward computation, never a silent crash.

---

## 5. Reward decomposition

The composite reward (this is the single function the GRPO trainer optimises):

```
R(τ) = 0.40 · F1_flagging(τ)               [outcome]
     + 0.10 · format_compliance(τ)         [structure]
     + 0.10 · valid_json(τ)                [structure]
     + 0.15 · action_diversity(τ)          [process]
     + 0.10 · efficiency(τ)                [process]
     + 0.15 · thinking_allocation(τ)       [process — Qwen3 thinking budget]
     + 0.30 · metacognitive(τ)             [meta — calibration, difficulty, coupling]
```

(The 0.30 metacog weight runs in parallel as a separate reward function passed to
`GRPOTrainer`, so the trainer sees two independent reward callables, not one weighted
sum. This is the §7 multi-independent-reward design verbatim.)

Formal definitions of `metacognitive` are in [`PAPER.md`](PAPER.md) §4.3.

---

## 6. Anti-reward-hacking mechanisms

> *"Do not optimize a reward you have not tried to break yourself first."* — OpenEnv FAQ Q57

Every defensive mechanism is enumerated here so a judge can verify by file lookup:

| Attack vector | Defence | Where in code |
|---|---|---|
| **Flag-everything** to drive recall up | F1 normalisation; flag budget `F` | [`environment.py`](code_review_env/server/environment.py) |
| **Flag-nothing** to avoid penalty | `format_compliance` requires non-empty flag list when bugs present | reward fn in `train_grpo.py` |
| **Infinite tool-call loop** to delay termination | Hard step cap = `3·num_files`; investigation budget | [`environment.py`](code_review_env/server/environment.py) |
| **Predict-long-everywhere** to maximise calibration | `difficulty_awareness` component penalises uncorrelated predictions | [`metacognitive_reward.py`](metacognitive_reward.py) |
| **Predict without thinking** to game the budget tag | `coupling` component requires actual `<think>` length to match prediction band | [`metacognitive_reward.py`](metacognitive_reward.py) |
| **Random-text payload** to satisfy format check | `valid_json` + `action_diversity` require real tool calls; F1 measures real outcomes | reward fn in `train_grpo.py` |
| **Edit timer / abuse globals** | Tool calls run in the FastMCP server process; agent cannot mutate session state, snippet store, or label dict | OpenEnv server boundary |
| **Feature-threshold shortcut** (NEW) | ~20% of safe files have inflated risk features ("deceptive traps") — looks like high-churn, high-complexity bug candidates, but are actually safe. Forces real reasoning, not feature hacking | [`environment.py`](code_review_env/server/environment.py) — `reset()` |

All eight attack vectors are empirically tested by [`scripts/red_team.py`](scripts/red_team.py)
and documented in [`SAFEGUARDS.md`](SAFEGUARDS.md). Every attempted attack scores
**below** the honest metacognitive policy. Closest gap: −22 %.

### Cost observability

Every tool response includes a running cost counter:

```
[Budget: 12 investigation points remaining | Flags: 2/5 | Thinking cost: 847 chars]
```

This makes the agent's resource consumption **observable in the observation space** — the agent can see how much reasoning it has already spent and adapt. Standard tool-use environments hide this; we expose it so the policy can make real-time strategic decisions about where to invest its remaining budget.

---

## 7. Reproducing the environment locally

```bash
git clone https://github.com/subwaycookiecrunch/Meta-project
cd Meta-project
pip install -r requirements.txt
python -m uvicorn server.environment:app --host 0.0.0.0 --port 7860
```

Then in a separate shell:

```bash
python demo.py            # runs untrained-baseline + smart-investigator policies
python scripts/red_team.py  # rebuilds data/red_team_results.json
python transfer_eval.py    # rebuilds grpo_output/transfer_results.png
```

The hosted Space at <https://huggingface.co/spaces/lucid987654/code-review-env-v3>
runs the same code and the same env build.

---

## 8. Why this design satisfies the OpenEnv guide criteria

| Guide criterion | This env |
|---|---|
| §1 *Task verifiable* | F1 against gold bug labels in the episode dataset |
| §1 *Success probability > 0* | Untrained baseline already achieves 28 % F1 on the transfer set |
| §4 *reset / step / state / reward* | Canonical OpenEnv contract, all four implemented |
| §4 *abuse prevention* | 7 attack vectors enumerated above with explicit defences |
| §6 *curriculum-ready* | 8 difficulty bands; subset by `cvss` to construct curriculum |
| §7 *multiple independent rewards* | 6 components (4 in main fn + 2 standalone callables) |
| §8 *resist reward hacking* | Empirical red-team; SAFEGUARDS.md formal writeup |
| §9 *process-aware feedback* | Metacog reward scores the prediction-vs-action coupling, not just final F1 |
| §13 *deployment* | Live Space + Docker container + local Uvicorn paths all working |
| §15 *inspect generations* | Live Trace Inspector tab streams every reward call |
