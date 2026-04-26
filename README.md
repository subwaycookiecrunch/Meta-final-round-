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
  - grpo
  - qwen3
  - security
---

# The Thinking Budget

Reasoning models think the same amount on everything. A one line variable declaration gets the same 4,000 token `<think>` block as a critical buffer overflow. This project trains a 1.7B model to predict how hard something is *before* it starts thinking, then rewards it for being right.

The model learns to skim easy files and deep dive on suspicious ones. 6x thinking ratio between bugs and safe files, up from basically flat.

[Try it](https://huggingface.co/spaces/lucid987654/code-review-env-v3) · [GitHub](https://github.com/subwaycookiecrunch/Meta-project) · [Blog](blog_post.md) · [Paper](PAPER.md)

## What it does

Before each `<think>` block the model emits a prediction: `short`, `medium`, or `long`. After it's done reasoning, the reward checks three things:

1. **Calibration**: did the actual thinking length match what was predicted?
2. **Difficulty awareness**: did hard files get `long` and easy files get `short`?
3. **Action coupling**: did every prediction lead to an actual tool call?

That third one turned out to be the most important. Without it, a model can just emit perfect predictions and never do any work. With it, orphan predictions get their score halved.

```
<budget_prediction>long</budget_prediction>
<think>
do_ioctl_handler at line 412 calls copy_from_user with a user-supplied
size and pipes the result into kmalloc. Classic integer overflow into
heap allocation. This is the bug.
</think>
<tool_call>{"name":"flag_vulnerable","arguments":{...}}</tool_call>
```

## Numbers

| Metric | Before | After |
|---|---:|---:|
| Thinking on safe files | 170 chars | **78 chars** |
| Thinking on buggy files | 182 chars | **473 chars** |
| Bug vs safe ratio | 1.07x | **6.06x** |
| Calibration accuracy | 33% (random) | **88%** |
| F1 on triage episodes | 0.14 | **1.00** |
| Transfer F1 (unseen domain) | 0.28 | **1.00** |
| Adversarial attacks defeated | 0 | **5/5** |

Trained on a single A10G in about 12 hours. Qwen3 1.7B with LoRA r=16, 4 bit quantization.

## The environment

Security code review. 150 real CVEs from NVD (Log4Shell, Dirty COW, PwnKit, BlueKeep, Zerologon, etc). 2,892 source files. The agent gets a CVE description and file paths but can't see code until it calls `read_file`, which costs investigation points. Budget is `2 × number_of_files` so you can't just read everything.

Six MCP tools:

| Tool | Cost | What it does |
|---|---:|---|
| `read_file(path)` | 1 pt | Read one file |
| `search_code(pattern)` | 2 pt | Grep across all files |
| `get_function_list(path)` | 1 pt | List functions + complexity |
| `flag_vulnerable(path, reasoning)` | free | Flag a file |
| `skip_file(path, reasoning)` | free | Skip a file |
| `submit_report(summary)` | free | End the episode |

The reward is a mix of live execution score and the metacognitive objective:

```
total = 0.50 * env_score + 0.30 * metacognitive_score + 0.20 * text_score
```

The metacognitive part combines calibration, difficulty awareness, and action coupling. The geometric structure means you can't game one axis without tanking another. Tried five different attack strategies, all scored below the honest policy.

## Red team results

| Strategy | Score |
|---|---:|
| Honest policy | **0.850** |
| Padding (real work + garbage thinking) | 0.662 |
| Flag everything | 0.426 |
| Skip everything | 0.278 |
| Invert difficulty predictions | 0.192 |
| Predict without acting | 0.076 |

The padding attack was closest because it actually does the job, it just fakes the reasoning depth. Still 22% behind. Run `python scripts/red_team.py` to reproduce.

Full writeup in [`SAFEGUARDS.md`](SAFEGUARDS.md).

## Transfer

Ran the trained policy on 5 episodes from a completely different domain. Payment processing race condition, JWT auth bypass, ML pipeline seed issue, React stale closure, SQL tenant filter bug. Nothing from training data.

| Policy | F1 | Thinking ratio |
|---|---:|---:|
| Untrained baseline | 0.28 | 1.29x |
| Trained policy | **1.00** | **5.24x** |

5 episodes is small, I know. But the allocation pattern (long on hard, short on easy) held across all five consistently. That's not what memorization looks like.

## Before/after example

**CVE-2021-41773 · `modules/mappers/mod_rewrite.c` · This file is SAFE (no bug)**

<details>
<summary>Untrained: wastes 312 chars and an investigation point on a config helper</summary>

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
→ reads the file, wastes 1 investigation point
```
</details>

<details>
<summary>Trained: correctly skips in 43 chars</summary>

```
<budget_prediction>short</budget_prediction>
<think>
URL rewrite helper. Not in the request path.
</think>
→ skips the file, saves the investigation point for actual bugs
```
</details>

Same model, same CVE, same file. Only difference is the metacognitive reward during training.

## How to try it

**Live demo:** Open the [Space](https://huggingface.co/spaces/lucid987654/code-review-env-v3), go to the Try The Agent tab, pick a CVE, watch it investigate.

**Retrain it:** Click the Colab badge below. Clones the repo, installs deps, runs training. About 3 to 5 hours on A10G.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/subwaycookiecrunch/Meta-project/blob/main/train_colab.ipynb)

**Run locally:**

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
```

## Repo map

| File | What it does |
|---|---|
| `app.py` | Gradio Space with 6 tabs |
| `train_grpo.py` | GRPO training with metacognitive reward |
| `metacognitive_reward.py` | The calibration + difficulty + coupling reward |
| `scripts/budget_processor.py` | LogitsProcessor for inference time think caps |
| `rubrics.py` | 8 composable sub rubrics using OpenEnv's WeightedRubric |
| `transfer_eval.py` | Held out domain transfer evaluation |
| `eval_baseline.py` | Before/after comparison |
| `demo.py` | Strategy ablation (skip all / flag all / smart) |
| `code_review_env/server/environment.py` | The MCP environment (6 tools + reward) |
| `data/cve_training_data.json` | 150 CVE episodes from NVD |
| `data/transfer_episodes.json` | 5 held out non CVE episodes |
| `PAPER.md` | Formal writeup |
| `JUDGES.md` | Judge checklist, maps every criterion to a file/command |
| `SAFEGUARDS.md` | Red team writeup |
| `blog_post.md` | HF Blog post |

## For judges

Start with [`JUDGES.md`](JUDGES.md). It maps every OpenEnv judging criterion to an exact file, command, or screenshot. Takes about 3 minutes to read.

Quick reproduce:

```bash
python scripts/red_team.py          # red team proof
python transfer_eval.py             # transfer results
python eval_baseline.py             # before/after comparison
```

## Links

- HF Space: https://huggingface.co/spaces/lucid987654/code-review-env-v3
- Colab: [train_colab.ipynb](https://colab.research.google.com/github/subwaycookiecrunch/Meta-project/blob/main/train_colab.ipynb)
- GitHub: https://github.com/subwaycookiecrunch/Meta-project

MIT License. Built for the Meta PyTorch OpenEnv Hackathon 2026 (India). Theme 3.1 World Modeling.
