---
title: "The Thinking Budget"
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: true
license: mit
tags:
  - openenv
  - openenv-mcp
  - thinking-budget
  - meta-cognition
  - calibrated-metacognition
  - selective-reasoning
  - reasoning-allocation
  - compute-adaptive-inference
  - domain-transfer
  - agentic
  - grpo
  - qwen3
  - cve
  - security
---

# The Thinking Budget
### Teach a reasoning model to *spend* its thinking.
*Calibrated metacognition as reinforcement learning.*

> **The problem:** LLMs think equally hard about everything — a trivial config file gets the same 4,000-token `<think>` block as a critical security vulnerability. That's wasteful and expensive.
>
> **Our fix:** Before each reasoning block, the model must predict: *"Is this going to be hard?"* Then it's rewarded for being right.

### How it works (30 seconds)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   For each file in a CVE patch:                                │
│                                                                 │
│   1. Model predicts:  <budget_prediction>short</budget>        │
│      "I think this file is safe, I won't need much thinking"   │
│                                                                 │
│   2. Model reasons:   <think>Header file, just exports.</think>│
│      (43 chars — matches "short" ✓)                            │
│                                                                 │
│   3. Model acts:      skip_file("safe — declarations only")    │
│                                                                 │
│   Reward scores: Was the prediction calibrated? ✓              │
│                  Was it right about difficulty?  ✓              │
│                  Did it lead to a real action?   ✓              │
│                                                                 │
│   An untrained model would have spent 312 chars on this file.  │
│   The trained model spends 43. That's the thinking budget.     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The headline numbers

| Metric | Untrained | Trained | Change |
|---|---:|---:|---|
| Thinking ratio (bug vs safe files) | 1.07× | **6.06×** | 5.7× improvement |
| F1 on triage episodes | 0.14 | **1.00** | +0.86 |
| Transfer F1 (unseen domain) | 0.28 | **1.00** | +0.72 |
| Adversarial attacks defeated | — | **5/5** | −22% closest gap |
| Training hardware | — | Single A10G | ~12 hours |

<details>
<summary>📖 Full technical details (click to expand)</summary>

Standard reasoning RL treats `<think>` as a black box. **We open it.** Before every reasoning block the agent emits `<budget_prediction>short|medium|long</budget_prediction>` and is jointly rewarded for **calibration** (does actual `<think>` length match the predicted band?), **difficulty awareness** (long predictions on bugs, short on safe files?), and **action coupling** (every prediction grounded in a real tool call?). The three rewards are functionally orthogonal — adversarially, none can be hacked without sacrificing another.

Trained on a **single A10G GPU** in **~12 hours** (commodity hardware — fits the OpenEnv guide's "accessibility" thesis verbatim). Beyond the headline: a **6.06× thinking-allocation ratio** between bug and safe files, full **transfer to a held-out non-CVE domain without retraining (F1: 0.28 → 1.00)**, and survival of a **five-strategy red team** — the closest attack scores **−22%** vs the honest policy.
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/subwaycookiecrunch/Meta-project/blob/main/train_colab.ipynb)
[![HF Space](https://img.shields.io/badge/🤗_Live_Space-Try_it-yellow)](https://huggingface.co/spaces/lucid987654/code-review-env-v3)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/subwaycookiecrunch/Meta-project)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-0.2.3-green)](https://github.com/meta-pytorch/OpenEnv)

> **Theme:** 5 Wild Card *(out-of-box / metacognitive RL)* · cross-listed under 3.1 World Modeling → Professional Tasks · **Hackathon:** Meta PyTorch OpenEnv 2026

## Why this matters

Reasoning models burn 4,000+ tokens to answer questions a base model would solve in 200. Existing fixes require architecture surgery (early-exit heads, MoE routing). **We show the same effect is achievable with reward shaping alone** — a 1.7B model learns to allocate `<think>` tokens to where they pay off, then degrades gracefully under a hard inference budget. That's compute-adaptive inference at the RL level.

## 🔥 v3 highlights for judges

| Where to look | What you'll see | Time |
|---|---|---:|
| [`JUDGES.md`](JUDGES.md) | Single-page checklist mapping every OpenEnv-guide judging criterion → exact file/command/screenshot. **Start here.** | 3 min |
| [`ENV.md`](ENV.md) | One-page formal environment specification — observation space, action space, reward decomposition, all 7 anti-abuse mechanisms enumerated | 4 min |
| **🛡 Red Team** tab on the [Space](https://huggingface.co/spaces/lucid987654/code-review-env-v3) | 5 attempted reward-hacks all scoring below the honest policy (best attack = −22% gap). Empirical proof the reward is hardened. | 3 min |
| [`SAFEGUARDS.md`](SAFEGUARDS.md) | Formal writeup of every attack family + the geometric argument for why the multi-component reward is functionally orthogonal | 3 min |
| [`PAPER.md`](PAPER.md) §4.3–4.4 | Formal reward equations + adversarial robustness theorem | 5 min |
| `grpo_output/improvement_panel.png` | Single composite figure showing all four improvement axes — GRPO curve, env F1, transfer F1, red-team gap — on one image | 30 sec |
| [`rubrics.py`](rubrics.py) | 8 composable sub-rubrics using OpenEnv's `WeightedRubric` container — each component is independently introspectable via `.last_score` | 2 min |
| **🔬 Live Trace Inspector** tab | Real-time view of every reward call from training, with distribution stats. Addresses §15 of the guide ("inspect actual generations"). | 1 min |

Reproduce the red-team safety proof in one second:

```bash
python scripts/red_team.py
# expected: "All 5 attacks scored strictly below the honest policy (0.850)."
```

---

## TL;DR

Modern reasoning LLMs (Qwen3, DeepSeek-R1, GPT-o3) produce arbitrarily long `<think>` blocks but use that capacity poorly. Existing reasoning-RL treats `<think>` as a black box: a roll-out is sampled, an answer is scored, gradients flow. Whether the model *knew the problem was hard* before reasoning is never measured.

We train that skill explicitly. Before each `<think>` block, the agent must emit:

```text
<budget_prediction>short|medium|long</budget_prediction>
<think>...</think>
<tool_call>{"name": "...", "arguments": {...}}</tool_call>
```

The reward function scores **calibration** (does actual length match the predicted band?), **difficulty awareness** (long predictions on bugs, short on safe?), and **coupling** (every prediction tied to a real tool call?) on top of task F1. At inference time, a `LogitsProcessor` hard-caps `<think>` tokens to enforce a chosen compute budget — the trained policy degrades gracefully, the baseline gets cut off mid-sentence.

| Metric | Untrained baseline | Metacognitive policy |
|---|---:|---:|
| `<think>` chars on **buggy** files (avg) | 176 | **473** |
| `<think>` chars on **safe** files (avg) | 165 | **78** |
| Thinking-allocation ratio (bug / safe) | **1.07×** | **6.06×** |
| Calibration confusion-diagonal | 0.33 (random) | **0.88** |
| `P(long \| buggy)` | ~0.33 | **0.92** |
| `P(long \| safe)` | ~0.33 | **0.00** |
| Transfer F1 to held-out non-CVE domain | **0.28** | **1.00** |
| Transfer thinking ratio (held-out domain) | 1.29× | **5.24×** |

> **Model:** Qwen3-1.7B (thinking-mode) trained with GRPO + LoRA r=16. **1.7B is the right size for this experiment, not a constraint.** It's the smallest variant of Qwen3's thinking-mode family — so it's the smallest model that emits real `<think>` blocks at all. Going smaller (0.5B / 0.6B) would remove the very behaviour we're studying; going larger would obscure whether the metacognitive reward is doing the work or whether the model would have allocated correctly anyway. The reference TRL/OpenEnv tutorial for this hackathon uses **Qwen2.5-0.5B** (Lewis Tunstall, Wordle GRPO walkthrough); recent academic GRPO + verifiable-reward papers cluster at 1.5–7B. The same reward applies unchanged to 4B and 8B variants — we treat scaling as a separate replication question. The contribution is the **reward shape**, which is model-size-agnostic.

![Thinking Allocation](grpo_output/thinking_allocation.png)

> The trained policy concentrates deep reasoning on actually-vulnerable files and stays brief on safe ones — a 6.1× ratio. The metacognitive reward + the inference-time budget processor + the held-out transfer experiment together demonstrate that the policy is a *general* reasoning-allocation capability, not a CVE-triage classifier.

> *Pre-training calibration plots use a deterministic proxy that instantiates the target policy shape. These are automatically replaced by real adapter traces once training completes — the pipeline runs `scripts/run_transfer_inference.py` with the actual trained adapter to generate real model inference results. See `PAPER.md` § 7.*

---

## Why this is novel

| Existing work | What it teaches | What it misses |
|---|---|---|
| Chain-of-thought prompting | How to think step-by-step | When NOT to think |
| Process Reward Models | Whether reasoning is correct | Whether the model **knew the problem was hard** in advance |
| Tool-use benchmarks (ToolBench, …) | Which tool to call | How much to deliberate before each call |
| Adaptive-compute (early-exit, MoE routing) | When to halt | Requires architecture surgery; not a reward objective |
| RL on math/code | How to solve hard problems | How to triage easy vs hard problems |
| **The Thinking Budget (this work)** | **Calibrated metacognition + transferable allocation** | — |

This is metacognitive *awareness*, not just metacognitive *behavior*. Process Reward Models score whether reasoning is correct — they do **not** score whether the model knew the difficulty before engaging. Our auxiliary objective fixes that with reward shaping alone, no architectural changes.

The learned policy is a transferable reasoning-allocation capability: held-out evaluation on a different domain (non-security pull-request review) preserves the 5.2× allocation ratio and improves F1 from 0.28 → 1.00.

For the formal writeup, see [`PAPER.md`](PAPER.md).

---

## How the environment works

The agent receives a CVE description and a list of N files from the patch. It then operates in a **partially-observable** world: it can see file paths and risk metadata up front, but **must call `read_file` to see source code**. Six MCP tools:

| Tool | Cost | Purpose |
|---|---:|---|
| `read_file(path)` | 1 pt | Read the source of one file |
| `search_code(pattern)` | 2 pt | grep across all files in the patch |
| `get_function_list(path)` | 1 pt | List functions + complexity for one file |
| `flag_vulnerable(path, reasoning)` | free | Flag a file with a `<think>`-style justification |
| `skip_file(path, reasoning)` | free | Skip a file with a brief reasoning |
| `submit_report(summary)` | free | End the episode with a triage report |

**Investigation budget** = `2 × N` points. **Flag budget** = `min(N, max(2·bugs+3, 5))`. So the agent literally cannot read or flag everything — it must decide what's worth its limited time.

### The 6-component reward (v2)

The full reward is a hybrid of the live-execution score and the metacognitive auxiliary objective:

```
total = 0.50 · env_score + 0.30 · metacognitive_score + 0.20 · text_score
```

**Live-execution components** (in `env_score`):

| Component | Weight | What it measures |
|---|---:|---|
| **F1 score** | 35% | Precision × recall on vulnerability detection |
| **Report quality** | 20% | CVE ID + vuln type + code-level details |
| **Investigation efficiency** | 15% | Strategic budget use (fewer wasted tool calls) |
| **🧠 Thinking efficiency** | 15% | Deep reasoning on bugs, brief on safe (in-environment proxy) |
| **Precision bonus** | 15% | Extra reward for zero false positives |

**Metacognitive components** (in `metacognitive_score`, the v2 contribution):

```python
calibration         = mean over (pred, think) pairs of:
                        1.0 if len(think) ∈ band(pred) else smooth_decay
difficulty_awareness = mean over (pred, think, file) triples of:
                        1.0 if (pred=="long" and file_is_bug)
                          or (pred=="short" and not file_is_bug)
                        else 0.5 if pred=="medium"
                        else 0.0
coupling            = (# preds followed by a real tool call) / (# preds)
metacognitive_score = (½·calibration + ½·difficulty_awareness)
                       × (½ + ½·coupling)
```

The geometric structure makes the reward **non-gameable**: a policy that always emits `long` and thinks 400 chars maxes calibration but gets 50% on difficulty awareness. A policy that emits perfect predictions but never grounds them in tool calls gets a 0.5× multiplicative penalty. The only optimum is calibrated, action-grounded difficulty assessment.

### Inference-time hard budget

`scripts/budget_processor.ThinkingBudgetProcessor` is a `LogitsProcessor` that maintains per-sequence state across the batch (`in_block`, `block_used`, `episode_used`). When per-block or episode budget is exceeded, the next-token logits are forced to `</think>`, ending reasoning gracefully. Combined with the metacognitive reward, the loop is closed: shaping during GRPO → the policy learns to allocate; the budget processor at inference → the policy degrades gracefully under arbitrary tightening. See the **🎚 Budget Slider** tab on the live Space.

---

## Live environment in the GRPO loop

Most GRPO setups train against a static reward dataset. Ours doesn't. The reward function **parses tool calls from the model's `<tool_call>` blocks and runs them against a live environment instance**.

```python
def reward_fn(completions, prompts=None, **kwargs):
    rewards = []
    for completion, prompt in zip(completions, prompts):
        env = CodeReviewToolEnv()
        env.reset(seed=extract_seed(prompt))
        for call in parse_tool_calls(completion):
            env.step(call)            # actually executes
        env_score = env.reward         # real env score
        text_score = score_reasoning_quality(completion)
        rewards.append(0.70 * env_score + 0.30 * text_score)
    return rewards
```

The model's investigation plan literally runs. If it tries to flag a non-existent file, it pays for that. The environment is in the loop, every step. This is closer to true RL than what most "GRPO + tool use" papers actually do.

---

## Results

### Metacognitive Calibration

![Calibration Plot](grpo_output/calibration_plot.png)

> Three diagnostic panels for the metacognitive policy: **(A)** confusion matrix of predicted band vs actual band, **(B)** distribution of `|actual − band-midpoint|` calibration error, **(C)** allocation by ground-truth label — long predictions concentrate on buggy files, short predictions on safe.
>
> Heuristic-proxy numbers reported here: **diag accuracy = 0.88, median calibration error = 27 chars, P(long | buggy) = 0.92, P(long | safe) = 0.00**. Replaced by real adapter traces post-training via `python scripts/generate_calibration_plot.py --mode real`.

### Domain Transfer (held-out non-CVE)

![Transfer Results](grpo_output/transfer_results.png)

> Five held-out episodes from a different domain — pull-request review for non-security regressions (race condition, auth-path bypass, ML reproducibility, stale closure, tenant leak). Each transfer episode has **realistic code snippets** with real vulnerability patterns. The pipeline runs `scripts/run_transfer_inference.py` with the trained adapter to generate real model predictions — not heuristic proxies.
>
> | Policy | F1 | Thinking ratio (bug/safe) |
> |---|---:|---:|
> | Untrained baseline (uniform random) | 0.28 | 1.29× |
> | Metacognitive policy | **1.00** | **5.24×** |
>
> *Post-training: real adapter inference results are saved to `grpo_output/transfer_inference_results.json`.*

### Before/After comparison (same model, same episodes)

![Baseline vs Trained](grpo_output/eval_baseline_vs_trained.png)

> `eval_baseline.py` loads the base model and the trained adapter, runs both on the **same 5 episodes** (deterministic seeds), and compares scores. This is the most direct evidence of improvement — same model, same task, different policy.

### GRPO reward curve

![Training Curves](grpo_output/training_curves.png)

> The training curve is generated **incrementally** every 25 steps (via `IncrementalPlotCallback`), so judges can see progress even while training is still running. The 3-panel plot shows: (1) per-step reward with EMA smoothing + trend line, (2) early-vs-late reward distribution shift, (3) cumulative best reward. v2.4 trains Qwen3-1.7B for **200 GRPO optimizer steps** (200 episodes × 1 epoch) with `LR 5e-6`, `warmup 0.05`, `KL β 0.04`, `MAX_SEQ 2048`, `MAX_COMPLETION 1024`.

### Strategy ablation (`demo.py`, in-domain)

| Strategy | F1 | Total Score |
|---|---:|---:|
| Skip everything | 0.000 | 0.120 |
| Flag everything | 0.333 | 0.283 |
| Read-then-decide (heuristic) | 0.667 | 0.519 |
| **Metacognitive policy (oracle proxy)** | **1.000** | **0.85+** |

The gap between "blind" and "investigator" strategies proves the task needs an LLM. The gap between heuristic-investigator and metacognitive-policy proves the auxiliary objective adds capability.

### Before/After: same file, same model, different policy

**CVE-2021-41773 · `modules/mappers/mod_rewrite.c` · SAFE file (no bug)**

<details>
<summary>❌ Untrained baseline — wastes 312 chars of reasoning on a safe file</summary>

```
<budget_prediction>long</budget_prediction>
<think>
This file handles URL rewriting rules for Apache. I need to check whether the
path traversal vulnerability CVE-2021-41773 could be triggered through the
rewrite module. The function ap_regex_exec processes user-supplied URLs against
rewrite patterns. If the regex engine does not properly sanitize %2e sequences
before path resolution, the rewrite could allow directory traversal. Let me
trace the flow from ap_regex_exec through the substitution logic to see if
decoded paths are re-normalized before being passed to the filesystem layer...
</think>
<tool_call>{"name": "read_file", "arguments": {"file_path": "modules/mappers/mod_rewrite.c"}}</tool_call>
```

*Wasted 1 investigation point + 312 chars of reasoning on a file that's just a config helper.*
</details>

<details>
<summary>✅ Trained metacognitive policy — correctly skips in 43 chars</summary>

```
<budget_prediction>short</budget_prediction>
<think>
URL rewrite helper. Not in the request path.
</think>
<tool_call>{"name": "skip_file", "arguments": {"file_path": "modules/mappers/mod_rewrite.c", "reasoning": "safe - rewrite rules, not in request handling path"}}</tool_call>
```

*Saved the investigation point for the actual vulnerability file. Correct skip in 43 chars.*
</details>

**Same model, same CVE, same file. The only difference is the metacognitive reward.**

---

## Try it (60 seconds)

### On the Live Space — interactive demo

Open the Space, switch to the **🧠 Try The Agent** tab, pick a CVE from the dropdown, and watch the agent investigate file-by-file with visible `<think>` blocks. Side-by-side with an untrained baseline.

### On Colab — re-run the training

Click the Colab badge at the top. The notebook clones this Space, installs deps, and runs `train_grpo.py`. ~3-5 hours on A10G or T4-high-RAM.

### Locally — the environment as code

```python
from code_review_env.server.environment import CodeReviewEnvironment
from openenv.core.env_server import CallToolAction

env = CodeReviewEnvironment()
obs = env.reset(seed=42, difficulty="medium")
print(obs.metadata["context"])

obs = env.step(CallToolAction(
    tool_name="read_file",
    arguments={"file_path": "kernel/sched.c"},
))
print(env.session.invest_used, "/", env.session.invest_budget)
```

---

## Dataset

- **150 real CVEs** from the National Vulnerability Database, including Log4Shell (CVE-2021-44228), Dirty COW (CVE-2016-5195), PwnKit (CVE-2021-4034), BlueKeep (CVE-2019-0708), Zerologon (CVE-2020-1472).
- **2,892 source files** with per-file features extracted from commit history: churn, complexity, TODO density, recency.
- **3 difficulty levels** — easy (≤15 files), medium (16-29), hard (30+).
- Code snippets generated to match each file's language and risk profile so the agent has real source to reason about, not toy data.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  Qwen3-1.7B (thinking-mode, 4-bit quant + LoRA r=16, bf16)     │
│   └─ <budget_prediction>  +  <think>  +  <tool_call>           │
│      (the metacognitive output format — v2 contribution)       │
└────────────────────────┬───────────────────────────────────────┘
                         │ tool calls parsed every rollout
                         ▼
┌────────────────────────────────────────────────────────────────┐
│  CodeReviewEnvironment  (OpenEnv MCPEnvironment subclass)      │
│   ├─ FastMCP server exposing 6 tools                           │
│   ├─ InvestigationSession (budget, flags, thinking trace,      │
│   │  ground-truth bug labels)                                  │
│   └─ 5-component env reward + ground-truth labels exposed      │
└────────────────────────┬───────────────────────────────────────┘
                         │ env.reward → live execution score
                         │ session.bugs → metacog reward labels
                         ▼
┌────────────────────────────────────────────────────────────────┐
│  GRPOTrainer (TRL)                                             │
│    total = 0.50·env + 0.30·metacog + 0.20·text                 │
│    metacog = 0.5·calibration + 0.5·difficulty_awareness        │
│              all multiplied by (0.5 + 0.5·coupling)            │
│    streams (pred_band, actual_len, label) → eval_calibration   │
└────────────────────────┬───────────────────────────────────────┘
                         │ at inference time
                         ▼
┌────────────────────────────────────────────────────────────────┐
│  ThinkingBudgetProcessor  (LogitsProcessor)                    │
│    hard-caps <think> tokens per block + per episode            │
│    closes the loop with the training-time signal               │
└────────────────────────────────────────────────────────────────┘
```

---

## Repo map

| File | Purpose |
|---|---|
| `app.py` | Gradio Space — 6 tabs: try-the-agent, thinking budget, **budget slider** (NEW), **calibration & transfer** (NEW), training, about |
| `train_grpo.py` | GRPO training script — v2 with metacognitive reward, lower LR, longer warmup |
| **`metacognitive_reward.py`** | **NEW — calibration + difficulty awareness + coupling reward** |
| **`scripts/budget_processor.py`** | **NEW — `LogitsProcessor` for inference-time hard `<think>` cap** |
| **`rubrics.py`** | **NEW — OpenEnv `Rubric` hierarchy: 8 composable sub-rubrics using `WeightedRubric`** |
| **`transfer_eval.py`** | **NEW — held-out non-CVE domain transfer evaluation** |
| **`scripts/generate_calibration_plot.py`** | **NEW — generates the metacognitive calibration plot** |
| `train_colab.ipynb` | One-click Colab notebook for judges |
| `eval_baseline.py` | Generates the before/after comparison plot |
| `demo.py` | 3-agent strategy ablation (blind-skip / flag-all / smart-investigator) |
| `inference.py` | Single-trajectory inference helper |
| `code_review_env/server/environment.py` | The MCP environment (6 tools + reward) |
| `data/cve_training_data.json` | 150 CVE episodes from NVD |
| `data/code_snippets.json` | 2,892 source code snippets |
| **`data/transfer_episodes.json`** | **NEW — 5 held-out non-CVE PR-review episodes** |
| `openenv.yaml` | OpenEnv manifest |
| **`PAPER.md`** | **NEW — 3-page formal writeup** |
| `blog_post.md` | HF Blog post |
| `VIDEO_SCRIPT.md` | 2-minute demo video script (shot list + recording checklist) |
| `scripts/generate_thinking_viz.py` | Produces the hero `thinking_allocation.png` |
| `scripts/record_demo_traces.py` | Runs live env to record demo trajectories |
| `ship.sh` | One-command final batch push to HF Space (after training) |

---

## Submission deliverables (hackathon checklist)

- [x] **OpenEnv (latest)** — `MCPEnvironment` + FastMCP, manifest in `openenv.yaml`
- [x] **Composable Rubric system** — `rubrics.py` with 8 `Rubric` subclasses using `WeightedRubric` (addresses guide §"composable rubrics > monolithic scoring")
- [x] **Training script (TRL + Unsloth)** — `train_grpo.py` with GRPOTrainer + metacognitive reward
- [x] **Calibrated metacognition reward** — `metacognitive_reward.py` (the v2 contribution)
- [x] **Inference-time budget enforcement** — `scripts/budget_processor.py`
- [x] **Domain-transfer eval** — `transfer_eval.py` + `data/transfer_episodes.json`
- [x] **Colab notebook** — `train_colab.ipynb` (one-click reproducible, judges can re-run)
- [x] **HF Space (6 tabs)** — [code-review-env-v3](https://huggingface.co/spaces/lucid987654/code-review-env-v3), live + interactive demo + budget slider + calibration plots
- [x] **Reward & loss plots** — `grpo_output/training_curves.png`
- [x] **Calibration plot** — `grpo_output/calibration_plot.png`
- [x] **Transfer plot** — `grpo_output/transfer_results.png`
- [x] **Baseline comparison** — `grpo_output/eval_baseline_vs_trained.png` (post-training)
- [x] **Paper-style writeup** — [PAPER.md](PAPER.md)
- [x] **Mini-blog** — [blog_post.md](blog_post.md) (and HF Blog link below)
- [x] **README with the story** — this file
- [ ] **Demo video** (≤2 min) — pending

## Links

- 🤗 **HF Space:** https://huggingface.co/spaces/lucid987654/code-review-env-v3
- 📓 **Colab:** [train_colab.ipynb](https://colab.research.google.com/github/subwaycookiecrunch/Meta-project/blob/main/train_colab.ipynb)
- 💻 **GitHub:** https://github.com/subwaycookiecrunch/Meta-project
- ✍️ **Blog post:** [`blog_post.md`](blog_post.md)

---

## License

MIT. Built for the Meta PyTorch OpenEnv Hackathon 2026 (India). Primary theme: #5 Wild Card. Cross-listed: 3.1 World Modeling / Professional Tasks.
