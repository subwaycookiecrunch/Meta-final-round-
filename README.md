---
title: CodeReviewEnv v3
emoji: 🔐
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
  - security
  - cve
  - grpo
  - qwen3
  - thinking-budget
  - agentic
---

# CodeReviewEnv v3 — Agentic Security Investigation

> **Theme:** 3.1 World Modeling → Professional Tasks
> **Model:** Qwen3-8B (4-bit) | **Training:** GRPO with live env execution | **Framework:** OpenEnv + FastMCP

> *"Our agent doesn't just reason — it knows **WHEN** to reason."*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/subwaycookiecrunch/Meta-project/blob/main/train_colab.ipynb) &nbsp; [![HF Space](https://img.shields.io/badge/🤗_Space-CodeReviewEnv_v3-yellow)](https://huggingface.co/spaces/lucid987654/code-review-env-v3) &nbsp; [![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/subwaycookiecrunch/Meta-project) &nbsp; [![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

---

## What it is

An MCP-based RL environment that trains an LLM to investigate CVE vulnerabilities the way a real security engineer does. Given a CVE description and a list of files from a code patch, the agent uses 6 tools to read code, search for patterns, and decide which files contain the bug — then writes a triage report.

The **innovation**: the agent learns a **thinking budget**. It reasons deeply (`<think>` blocks > 100 chars) on suspicious files and stays brief on safe files. The reward function explicitly tracks this, so wasted "deep thinking" on clean code is penalized just like a missed bug.

## The 6 tools

| Tool | Cost | Description |
|---|---|---|
| `read_file` | 1 pt | Read source code of a file in the patch |
| `search_code` | 2 pt | Search for a text pattern across all files |
| `get_function_list` | 1 pt | List functions with complexity indicators |
| `flag_vulnerable` | free | Flag a file with detailed reasoning |
| `skip_file` | free | Mark a file as safe with brief reasoning |
| `submit_report` | free | Submit final triage report (ends episode) |

Investigation budget is `2 × num_files`. Flag budget is `min(num_files, max(2·bugs+3, 5))`.

## Reward function (5 components)

| Component | Weight | What it measures |
|---|---|---|
| **F1 score** | 35% | Precision × recall on vulnerability detection |
| **Report quality** | 20% | CVE ID, vuln type keywords, code-level details |
| **Investigation efficiency** | 15% | Strategic budget use (fewer wasted steps) |
| **Thinking efficiency** | 15% | Deep reasoning on bugs, brief on safe files |
| **Precision bonus** | 15% | Extra reward for zero-false-positive flagging |

The 5-component reward is what prevents the **"always skip"** reward hacking exploit that killed v1.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Qwen3-8B (4-bit quant + LoRA r=16)                          │
│   ├─ <think> block: deep reasoning on suspicious files       │
│   └─ <tool_call> blocks: 6 MCP tools                         │
└──────────────────┬───────────────────────────────────────────┘
                   │ tool calls parsed from generation
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  CodeReviewEnvironment (MCPEnvironment subclass)             │
│   ├─ 150 real CVEs from NVD                                  │
│   ├─ 2,892 code files with churn/complexity/recency feats    │
│   └─ FastMCP server exposes 6 tools                          │
└──────────────────┬───────────────────────────────────────────┘
                   │ live execution returns env score
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  GRPOTrainer (TRL)                                           │
│   reward = 0.70 * env_score + 0.30 * text_score              │
└──────────────────────────────────────────────────────────────┘
```

The reward function **executes the parsed tool calls against the live environment** — this is not a static dataset. The model's investigation plan actually runs, and the real environment score becomes the reward.

## Training results

![Training Curves](grpo_output/training_curves.png)

![Baseline vs Trained](grpo_output/eval_baseline_vs_trained.png)

Run `python eval_baseline.py` after training to regenerate the before/after comparison.

## Quick start

### Run the environment locally

```bash
git clone https://huggingface.co/spaces/lucid987654/code-review-env-v3
cd code-review-env-v3
pip install -r requirements.txt
python demo.py    # 3-agent comparison: blind-skip, flag-all, smart-investigator
```

### Train it yourself (Colab)

Click the badge at the top of this README, or open `train_colab.ipynb` directly. It clones this Space, installs deps, runs `train_grpo.py`, and renders the curves. ~3-5 hours on A10G.

### Use it as an OpenEnv RL env

```python
from code_review_env.server.environment import CodeReviewEnvironment
from openenv.core.env_server import CallToolAction

env = CodeReviewEnvironment()
obs = env.reset(seed=42, difficulty='easy')
print(obs.metadata['context'])

obs = env.step(CallToolAction(
    tool_name='read_file',
    arguments={'file_path': 'kernel/sched.c'}
))
```

## File map

| File | Purpose |
|---|---|
| `app.py` | Gradio dashboard (auto-refreshing training logs + plot) |
| `train_grpo.py` | GRPO training script with bf16 dtype-safety hook |
| `train_colab.ipynb` | One-click Colab notebook for judges |
| `eval_baseline.py` | Before/after comparison script |
| `demo.py` | 3-agent baseline comparison (blind-skip vs flag-all vs investigator) |
| `inference.py` | LLM inference baseline against the environment |
| `code_review_env/server/environment.py` | The MCP environment (6 tools, reward fn) |
| `data/cve_training_data.json` | 150 CVE episodes from NVD |
| `data/code_snippets.json` | 2,892 source files |
| `openenv.yaml` | OpenEnv manifest |

## Engineering notes (the bugs we hunted)

### bf16 dtype mismatch in `lm_head`

Qwen3 + 4-bit quantization with `bnb_4bit_compute_dtype=bfloat16` left `lm_head` in float32 because `prepare_model_for_kbit_training` explicitly upcasts non-int8 params for stability ([huggingface/peft#816](https://github.com/huggingface/peft/issues/816)). The fp32 `lm_head` matmul-ed against bf16 hidden states throws `RuntimeError: expected scalar type Float but found BFloat16`.

**Fix:** install a `register_forward_pre_hook(with_kwargs=True)` on `lm_head` that casts the input dtype to match `weight.dtype` on every forward call. Bulletproof — works whether `lm_head` ends up fp32 or bf16.

### Reward hacking: "always skip"

v1 gave +0.8 for correct skips. The model exploited this by never flagging anything, scoring high while finding zero bugs. Fixed in v3 with asymmetric penalties: missing a real bug costs −1.0, correct skips earn only +0.3.

### TRL `warnings_issued` compatibility

GRPOTrainer expects `model.warnings_issued` but Qwen3 doesn't have it. One-line workaround: `model.warnings_issued = {}`.

## Submission deliverables

| Required item | Location |
|---|---|
| OpenEnv RL environment | `code_review_env/server/environment.py` |
| OpenEnv manifest | `openenv.yaml` |
| Live HF Space | https://huggingface.co/spaces/lucid987654/code-review-env-v3 |
| GitHub repo | https://github.com/subwaycookiecrunch/Meta-project |
| Colab training notebook | `train_colab.ipynb` |
| Reward improvement curves | `grpo_output/training_curves.png` |
| Before/after eval | `grpo_output/eval_baseline_vs_trained.png` + `eval_baseline.py` |
| Blog post | `blog_post.md` |
| Demo agents (3 baselines) | `demo.py` |

## License

MIT. Built for the Meta PyTorch OpenEnv Hackathon 2026, Theme 3.1.
