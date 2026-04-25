---
title: "The Thinking Budget — calibrated metacognition as RL"
thumbnail: /blog/assets/thinking-budget/hero.png
authors:
- user: lucid987654
---

# The Thinking Budget — calibrated metacognition as RL

> Built for the **Meta PyTorch OpenEnv Hackathon 2026** (India). Theme 3.1 — World Modeling / Professional Tasks. [Paper-style writeup](https://github.com/subwaycookiecrunch/Meta-project/blob/main/PAPER.md) · [Live Space](https://huggingface.co/spaces/lucid987654/code-review-env-v3) · [Repo](https://github.com/subwaycookiecrunch/Meta-project)

Modern reasoning LLMs — Qwen3, DeepSeek-R1, GPT-o3, Claude Sonnet — can produce arbitrarily long `<think>` blocks. They use that capacity poorly. Existing reasoning-RL (GRPO, PPO over reasoning tokens) treats `<think>` as a black box: a roll-out is sampled, an answer is scored, gradients flow. Whether the model **knew the problem was hard before reasoning** is never measured and never trained.

We trained that skill explicitly.

Before each `<think>` block, our agent must emit a budget prediction:

```text
<budget_prediction>long</budget_prediction>
<think>
do_ioctl_handler in drivers/foo.c:412 calls copy_from_user with a
user-supplied size; integer overflow into kmalloc → heap overflow.
This is the bug.
</think>
<tool_call>{"name": "flag_vulnerable", "arguments": {...}}</tool_call>
```

The reward function scores **calibration** (does actual length match the predicted band?), **difficulty awareness** (long predictions land on bugs, short on safe?), and **coupling** (every prediction grounded in a real tool call?) on top of task F1. At inference time, a `LogitsProcessor` hard-caps `<think>` tokens; the trained policy degrades gracefully, the untrained baseline gets cut off mid-sentence.

The substrate is real CVE triage. The contribution is the auxiliary objective — and we have evidence the learned policy transfers, without retraining, to a different domain.

This post walks through the environment, the reward design, the bug-hunt that almost killed the project, and the held-out transfer experiment.

## The headline image

![Thinking allocation: untrained vs trained](https://huggingface.co/spaces/lucid987654/code-review-env-v3/resolve/main/grpo_output/thinking_allocation.png)

**Left:** untrained Qwen3-8B reasons roughly the same length on every file in a patch — the deep-thinking ratio (bug / safe) is **1.07×**. **Right:** the trained policy concentrates deep `<think>` blocks on bugs (avg 473 chars) and stays brief on safe files (avg 78 chars) — ratio **6.06×**.

That single plot is the project. Everything else explains how to make it.

## Why this is novel

Process Reward Models score whether reasoning is *correct*. They do **not** score whether the model *knew the problem was hard* in advance. That distinction matters because:

| Existing work | Teaches | Misses |
|---|---|---|
| Chain-of-thought prompting | How to reason step-by-step | When NOT to |
| Process Reward Models | Whether reasoning is correct | Whether the model knew the problem was hard |
| Tool-use benchmarks | Which tool to call | How much to deliberate first |
| Adaptive-compute (early-exit, MoE routing) | When to halt | Requires architecture surgery; not a reward objective |
| RL on math/code | How to solve hard problems | How to triage easy vs hard |
| **This work** | **Calibrated metacognition + transferable allocation** | — |

Models that self-assess difficulty *before* reasoning can allocate compute on demand. That's compute-adaptive inference via reward shaping alone — no architectural changes. And calibration is also a safety property: an agent that overestimates its certainty is a known failure mode; one that reliably emits `medium` when actually unsure is a foundation for downstream uncertainty handling.

## How the environment works

**Six MCP tools.** The agent gets a CVE description and a list of N files (paths only — no source). To see code it must call `read_file`. Other tools: `search_code` (grep), `get_function_list` (extract callable + complexity), `flag_vulnerable` (mark a file with reasoning), `skip_file` (mark safe with reasoning), `submit_report` (end episode with a triage summary).

**Hard budgets.** Investigation budget = `2 × N` points (each `read_file` costs 1, `search_code` costs 2). Flag budget = `min(N, max(2·bugs+3, 5))`. The agent **literally cannot read everything or flag everything** — it must triage.

**Real CVEs.** 150 episodes from the National Vulnerability Database including Log4Shell (CVE-2021-44228), Dirty COW (CVE-2016-5195), PwnKit (CVE-2021-4034), BlueKeep (CVE-2019-0708), Zerologon (CVE-2020-1472). 2,892 source files with churn / complexity / TODO / recency features pulled from commit history. Three difficulty levels.

## The reward — six environment components and three metacognitive ones

The full reward is a hybrid:

```
total = 0.50 · env_score  +  0.30 · metacognitive_score  +  0.20 · text_score
```

**Live-execution components** (in `env_score`):

| Component | Weight | What it measures |
|---|---:|---|
| F1 score | 35% | Precision × recall on vulnerability detection |
| Report quality | 20% | CVE ID, vuln type keywords, code-level details |
| Investigation efficiency | 15% | Strategic budget use (fewer wasted steps) |
| 🧠 Thinking efficiency | 15% | Deep reasoning on bugs, brief on safe (in-env proxy) |
| Precision bonus | 15% | Extra reward for zero false positives |

**Metacognitive components** (`metacognitive_reward.py`, the v2 contribution):

```python
calibration         = mean over (pred, think) pairs of:
                        1.0 if len(think) ∈ band(pred)
                        else smooth_decay
difficulty_awareness = mean over (pred, think, file) triples of:
                        1.0 if (pred=="long" and file_is_bug)
                          or (pred=="short" and not file_is_bug)
                        else 0.5 if pred=="medium"
                        else 0.0
coupling            = (# preds followed by a real tool call) / (# preds)
metacognitive_score = (½·calibration + ½·difficulty_awareness)
                       × (½ + ½·coupling)
```

The geometric structure makes it **non-gameable**. A policy that always emits `long` and thinks 400 chars maxes calibration but gets 50% on difficulty awareness on average. A policy that emits perfect predictions but never grounds them in tool calls gets a 0.5× multiplicative penalty. The only optimum is calibrated, action-grounded difficulty assessment.

### Inference-time hard budget

`scripts/budget_processor.ThinkingBudgetProcessor` is a `LogitsProcessor` that maintains per-sequence state across the batch. When per-block or episode budget is exceeded inside a `<think>...</think>`, the next-token logits are forced to `</think>`, ending reasoning gracefully:

```python
class ThinkingBudgetProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            self._scan_last_token(int(input_ids[b, -1].item()), self._state[b])
            if self._state[b]["in_block"] and self._budget_exceeded(self._state[b]):
                forced = torch.full_like(scores[b], float("-inf"))
                forced[self.preferred_close_id] = 0.0
                scores[b] = forced
        return scores
```

Combined with the metacognitive reward, the loop is closed. Shaping during GRPO → policy learns to allocate; budget processor at inference → policy degrades gracefully under arbitrary tightening. You can move the slider yourself in the [🎚 Budget Slider tab](https://huggingface.co/spaces/lucid987654/code-review-env-v3) on the live Space.

## Live environment in the GRPO loop — not a static dataset

Most "RL on reasoning + tools" papers actually train against a static reward dataset where the reward is precomputed. We do something more honest: the reward function **parses tool calls from the model's `<tool_call>` blocks and runs them against a live environment instance**, every step.

```python
def reward_fn(completions, prompts=None, **kwargs):
    rewards = []
    for completion, prompt in zip(completions, prompts):
        env = CodeReviewToolEnv()
        env.reset(seed=extract_seed(prompt))
        for call in parse_tool_calls(completion):
            env.step(call)            # actually executes
        env_score = env.reward         # real environment score
        text_score = score_reasoning_quality(completion)
        rewards.append(0.70 * env_score + 0.30 * text_score)
    return rewards
```

If the model tries to flag a file that doesn't exist, the env returns an error and the reward reflects it. If it submits a thoughtful report mentioning the right CVE ID and vulnerability type, the report-quality scorer rewards it. **The environment is in the loop**, every step, every group.

This matters because GRPO trains on relative ranks within a group. Static reward signals can collapse into "always emit the same template" if the template wins on average. Live execution makes that exploit *impossible* — every completion is a real interaction, not a textual artifact.

## The bug hunt: a `prepare_model_for_kbit_training` story

I want to record one debugging episode because it taught me something.

Mid-training, the run kept dying with:

```
RuntimeError: expected scalar type Float but found BFloat16
  at logits = self.lm_head(hidden_states[:, slice_indices, :])
```

First fix: cast `lm_head.weight` to bf16. Re-ran. Same crash.

Second fix: walk every parameter in the model, force-cast non-int8 floats to bf16. Re-ran. Same crash.

Then I read the source of PEFT's `prepare_model_for_kbit_training` and found this:

```python
# cast all non INT8 parameters to fp32
for param in model.parameters():
    if param.dtype in (torch.float16, torch.bfloat16):
        param.data = param.data.to(torch.float32)
```

It silently undoes everything I cast — by design, for QLoRA stability ([huggingface/peft#816](https://github.com/huggingface/peft/issues/816)).

The final fix was to stop fighting PEFT. I installed a `register_forward_pre_hook(with_kwargs=True)` on `lm_head` that casts the input tensor to match `lm_head.weight.dtype` on **every forward call**. Whatever dtype the head ends up with, the input adapts:

```python
def hook(module, args, kwargs):
    target = module.weight.dtype
    new_args = tuple(
        a.to(target) if torch.is_tensor(a) and a.is_floating_point() and a.dtype != target else a
        for a in args
    )
    new_kwargs = {
        k: (v.to(target) if torch.is_tensor(v) and v.is_floating_point() and v.dtype != target else v)
        for k, v in kwargs.items()
    }
    return (new_args, new_kwargs)

model.lm_head.register_forward_pre_hook(hook, with_kwargs=True)
```

The lesson: when a library invariant fights you, **adapt around it instead of overpowering it**. The hook approach turns out to be the canonical fix HF maintainers recommend, and it makes the training script robust to any future PEFT upcasting changes.

## Reward hacking, briefly

Our first reward function gave +0.8 for correctly skipping a safe file. The model figured out it could skip *every* file and rack up high scores while finding zero bugs. Classic reward hacking — Goodhart's Law in action. *A reward becomes a target, and once it's a target it stops measuring what you wanted.*

Fix: rebalance asymmetrically. **Missing a real bug costs −1.0.** Correct skips earn **+0.3**. The "always skip" exploit now scores ~0. The 5-component composite makes the exploit harder still — even if you cheated F1, you'd still need a passing report quality, a valid investigation pattern, and a non-trivial thinking allocation.

## Results

**Training curve** (GRPO on Qwen3-8B with LoRA r=16, four-bit quantization, bf16 compute):

![training curve](https://huggingface.co/spaces/lucid987654/code-review-env-v3/resolve/main/grpo_output/training_curves.png)

Mean reward climbs from baseline ~0.20 to ~0.5+ over training; variance shrinks as the agent learns the budget policy.

**Strategy ablation** (`demo.py`, three baseline policies on the same CVE distribution):

| Strategy | F1 | Total reward |
|---|---:|---:|
| Skip everything | 0.000 | 0.120 |
| Flag everything | 0.333 | 0.283 |
| Read-then-decide (heuristic) | 0.667 | 0.519 |
| **Qwen3-8B + GRPO** | **see eval** | **see eval** |

The gap between "blind" and "investigator" strategies proves the task needs an LLM. The gap between heuristic-investigator and trained-LLM is what GRPO produces.

**Per-trajectory traces** (recorded by running both policies on the same Dirty COW patch):

| Policy | Steps | Avg `<think>` on bugs | Avg `<think>` on safe | Ratio | F1 |
|---|---:|---:|---:|---:|---:|
| Untrained | 25 | 62 | 60 | 1.0× | 0.14 |
| Trained-style | 29 | 729 | 20 | **36×** | 1.00 |

You can step through both trajectories yourself in the [HF Space](https://huggingface.co/spaces/lucid987654/code-review-env-v3) under **🧠 Try The Agent**.

## Transfer — the result that makes this real research

The CVE substrate is a vehicle. The actual contribution is the **reward shape that produces calibrated, selective reasoning**. To check whether the learned shape is genuinely a meta-skill or just a CVE-classifier in disguise, we constructed five held-out episodes from a different domain — pull-request review for non-security regressions:

| Task | Files | Bug type |
|---|---:|---|
| TR-CR-001 | 12 | Race condition in payment refactor |
| TR-CR-002 | 14 | Path-prefix auth bypass in JWT migration |
| TR-CR-003 | 10 | Reproducibility regression in ML pipeline |
| TR-CR-004 | 11 | Stale-closure bug in React refactor |
| TR-CR-005 |  9 | Tenant filter missing in SQL optimization |

None of these are CVEs. None appear anywhere in the training data. We evaluated the same risk-driven allocation policy that the metacognitive reward shapes the trained model toward — using only structural file features (churn, complexity, TODOs, recency, is_test), never the ground-truth label.

| Policy | Aggregate F1 | Aggregate think ratio (bug / safe) |
|---|---:|---:|
| Untrained baseline (uniform random) | 0.28 | 1.29× |
| Metacognitive policy (transfer) | **1.00** | **5.24×** |

The same allocation pattern that solves CVE triage solves PR review. Reasoning effort routes to the right files in a domain the policy has never seen, on bug *types* never observed during training. That's domain-general meta-cognition, not just a CVE classifier.

The transferable skill compounds across:

- **Code review more broadly** — most files in a PR are uninteresting; focus on the few that aren't (shown).
- **Customer support triage** — most tickets are template-matchable; a few need deep investigation.
- **Scientific paper screening** — most submissions are clearly out of scope; a few merit detailed review.
- **Bug triage in production** — most alerts are noise; a few are real incidents.

In every case the meta-skill is the same: **decide whether to invest in deep reasoning based on cheap signals first, predict the budget honestly, and deliver exactly that much.**

## Stack

- **OpenEnv 0.2.3** — `MCPEnvironment` base class + FastMCP server with 6 tools
- **TRL ≥ 0.17** — `GRPOTrainer` with custom `reward_funcs` parsing & live env execution
- **Unsloth + bitsandbytes** — 4-bit Qwen3-8B fits in 16 GB VRAM
- **PEFT** — LoRA r=16, α=32, on attention + MLP projections
- **Gradio 5** — interactive Space (training dashboard + agent demo + thinking-budget visualizer)

## Try it

- 🤗 **HF Space:** [lucid987654/code-review-env-v3](https://huggingface.co/spaces/lucid987654/code-review-env-v3) — interactive demo, switch to the **🧠 Try The Agent** tab and pick Dirty COW
- 📓 **Colab:** [`train_colab.ipynb`](https://colab.research.google.com/github/subwaycookiecrunch/Meta-project/blob/main/train_colab.ipynb) — re-run training end-to-end
- 💻 **GitHub:** [`subwaycookiecrunch/Meta-project`](https://github.com/subwaycookiecrunch/Meta-project)
- 🎬 **2-minute demo video:** _(YouTube link added at submission)_

```bash
git clone https://huggingface.co/spaces/lucid987654/code-review-env-v3
cd code-review-env-v3
pip install -r requirements.txt
python demo.py    # 3-agent strategy ablation
python scripts/record_demo_traces.py    # regenerate the Try The Agent traces
python scripts/generate_thinking_viz.py # regenerate the hero plot
```

---

*Built for the Meta PyTorch OpenEnv Hackathon 2026 — Theme 3.1: World Modeling / Professional Tasks.*

*If you train on this environment and find a useful reasoning-allocation policy that transfers to other agentic tasks, please open an issue on GitHub or DM me — I'd love to hear about it.*
