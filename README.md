---
title: CodeReviewEnv v3
emoji: 🔐
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: false
license: mit
---

# CodeReviewEnv v3 — Agentic Security Investigation

> **Theme:** 3.1 World Modeling → Professional Tasks  
> **Model:** Qwen3-8B | **Training:** GRPO | **Framework:** OpenEnv + FastMCP

## 🔐 What Is This?

An MCP-based environment that trains LLMs to investigate CVE vulnerabilities like a real security engineer. The agent uses 6 tools to read code, search for patterns, flag vulnerabilities, and write triage reports.

**The Innovation:** Our agent learns a "Thinking Budget" — it reasons deeply on suspicious code and briefly on safe files, just like a human expert.

## 🎯 The Pitch

> *"Our agent doesn't just reason — it knows WHEN to reason."*

## 🏗️ Architecture

| Component | Details |
|---|---|
| **Environment** | 6 MCP tools: `read_file`, `search_code`, `get_function_list`, `flag_vulnerable`, `skip_file`, `submit_report` |
| **Dataset** | 150 real-world CVEs (Log4Shell, Dirty COW, PwnKit, BlueKeep, Zerologon) with 2,892 code files |
| **Model** | Qwen3-8B with thinking mode (`<think>` blocks for deep reasoning) |
| **Training** | GRPO with **live environment execution** — tool calls are parsed and run against the real environment |
| **Reward** | 5-component: F1 (35%) + Report Quality (20%) + Efficiency (15%) + Thinking Budget (15%) + Precision (15%) |

## 📊 Training Results

![Training Curves](grpo_output/training_curves.png)

## 🚀 Quick Start

```bash
pip install openenv-core fastmcp
python demo.py
```

## 📁 Files

| File | Description |
|---|---|
| `app.py` | HuggingFace Space (Gradio UI) |
| `train_grpo.py` | GRPO training script |
| `train_colab.ipynb` | Colab notebook for judges |
| `demo.py` | 3-agent comparison demo |
| `server/environment.py` | MCP environment with 6 tools |
| `data/cve_training_data.json` | 150 CVE episodes |
| `data/code_snippets.json` | 2,892 code files |

## 📹 Demo Video

[Watch the 2-minute demo →](TODO)

## 📝 Blog Post

[Read on HuggingFace →](TODO)

## License

MIT
