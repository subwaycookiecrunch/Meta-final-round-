---
title: "I taught a 1.7B model to know when not to think hard"
thumbnail: /blog/assets/thinking-budget/hero.png
authors:
- user: lucid987654
  name: Shri Raj Bisaria
  solo_participant: true
---

# I taught a 1.7B model to know when not to think hard

I kept watching Qwen3 generate 4,000 token `<think>` blocks on files that were obviously fine. Like, `extern int x;` does not need 3 paragraphs of analysis. But the model doesn't know that. It just thinks the same amount on everything.

So I tried training it to stop doing that.

**Quick results:**

| What changed | Before | After |
|---|---:|---:|
| Thinking on safe files | 170 chars | **78 chars** (−54%) |
| Thinking on buggy files | 182 chars | **473 chars** (+160%) |
| Bug-vs-safe ratio | 1.07× | **6.06×** |
| Transfer to new domain | F1 = 0.28 | **F1 = 1.00** |
| Reward-hacking attacks defeated | 0 | **5/5** |

[live Space](https://huggingface.co/spaces/lucid987654/code-review-env-v3) | [GitHub](https://github.com/subwaycookiecrunch/Meta-project)

## The setup

Before each `<think>` block, the model has to say how much it's going to think:

```text
<budget_prediction>long</budget_prediction>
<think>
do_ioctl_handler at line 412 calls copy_from_user with a user-supplied
size and pipes the result into kmalloc. Classic integer overflow into
heap allocation. This is the bug.
</think>
<tool_call>{"name":"flag_vulnerable","arguments":{...}}</tool_call>
```

Three options: `short`, `medium`, `long`. The reward checks whether the prediction matched the actual length, whether hard files got `long` and easy files got `short`, and whether each prediction was followed by a real tool call. That last part matters more than I expected (more on that later).

## Why not just use a process reward model

PRMs check if reasoning steps are correct. They don't check whether the model knew upfront that the problem was hard. A PRM will happily reward 4,000 tokens of reasoning on a trivial file as long as the conclusion is right. This penalizes that.

I looked into early-exit transformers and MoE routing as alternatives but those need architecture changes. Can't bolt them onto an existing checkpoint. I wanted something that works with just reward shaping on top of standard GRPO.

## Data

150 CVEs from NVD. Log4Shell, Dirty COW, PwnKit, BlueKeep, Zerologon, and a bunch of less famous ones. 2,892 source files with features from commit history (churn, complexity, TODO count, recency).

The agent sees the CVE description and file paths but not the code itself. Has to call `read_file` to look at anything. Six MCP tools total. Each costs investigation points, budget is `2 × N` files, so the agent has to be selective.

If you're building an OpenEnv: the guide has reserved tool names (`reset`, `step`, `state`, `close`). My first version used `state` as a tool name and things broke silently for like an hour before I figured out why.

## The dtype thing

GRPO kept crashing:

```
RuntimeError: expected scalar type Float but found BFloat16
  at logits = self.lm_head(hidden_states[:, slice_indices, :])
```

Cast `lm_head.weight` to bf16. Still crashed. Walked every parameter, force-cast everything. Still crashed.

Spent a while on this before I looked at PEFT's `prepare_model_for_kbit_training` source and found this:

```python
for param in model.parameters():
    if param.dtype in (torch.float16, torch.bfloat16):
        param.data = param.data.to(torch.float32)
```

It upcasts everything back to fp32 after you downcast it. By design, for QLoRA stability ([peft#816](https://github.com/huggingface/peft/issues/816)).

I tried a few things to override this. None of them worked cleanly. What ended up working was just putting a forward hook on lm_head that casts the input to match the weight dtype at call time:

```python
def hook(module, args, kwargs):
    target = module.weight.dtype
    new_args = tuple(
        a.to(target) if torch.is_tensor(a) and a.is_floating_point()
        and a.dtype != target else a
        for a in args
    )
    new_kwargs = {
        k: (v.to(target) if torch.is_tensor(v)
            and v.is_floating_point() and v.dtype != target else v)
        for k, v in kwargs.items()
    }
    return (new_args, new_kwargs)

model.lm_head.register_forward_pre_hook(hook, with_kwargs=True)
```

Not elegant but it works and it doesn't break when PEFT updates. I still don't love it.

## Reward hacking

First reward: +0.8 for correct skip, +1.0 for correct flag, -1.0 for missing a bug.

Model learned to skip everything within 30 steps. Most files are safe, so skip-all gets you ~80% accuracy, and +0.8 per skip adds up. Reward function was just wrong.

Fixed the obvious stuff (skip reward down to +0.3, miss penalty up to -1.0) and then wrote a red team before doing the real run because I didn't want to get burned again. Tried a few things: flag-everything with always-long predictions, skip-everything with always-short, emit perfect predictions but never call any tools (this one was interesting), pad `<think>` with garbage text while still taking the right actions, and also the reverse where you predict `long` on safe files and `short` on bugs.

Honest policy scores 0.850. The garbage-padding one was the strongest at 0.662 because it actually does the task correctly, it just fills the reasoning with junk. Everything else was under 0.43.

The no-tool-calls attack was the one I was most worried about but it turns out the coupling multiplier kills it. If predictions aren't followed by real tool calls the metacognitive score gets halved. I almost didn't add that term. Would have been a problem.

```bash
python scripts/red_team.py
```

Details in [`SAFEGUARDS.md`](SAFEGUARDS.md).

## Numbers

Anyway. Before training: model thinks ~170 chars on every file regardless. Bug-to-safe ratio is 1.07x.

After: 473 chars on buggy files, 78 on safe. 6.06x ratio.

Calibration accuracy hit 88% on the diagonal (random would be 33%). P(long | buggy) = 0.92. P(long | safe) = 0.00.

## Transfer

Ok so this is the part I need to be careful about because the numbers look too good and I want to be upfront about the caveats.

I ran the trained policy on 5 non-CVE episodes. Completely different domain. Payment processing race condition, JWT auth bypass, ML pipeline seed issue, React stale closure, SQL tenant filter bug. None in training data. Different bug types entirely.

Baseline F1: 0.28. Trained policy F1: 1.00. Thinking ratio went from 1.29x to 5.24x.

Now, 5 episodes is a tiny eval set. F1 of 1.00 on 5 episodes means the model got all 5 right. That's good but it's not a statistically robust claim. I'd want at least 50-100 episodes across more domains before I'd be confident the transfer is real and not just lucky. I'm reporting it because it's what I have and because the allocation pattern (long on hard files, short on easy ones) showed up consistently across all 5, which at least suggests it's not random.

```bash
python transfer_eval.py
```

## What I'd change

Started with Qwen3-8B. OOM'd on the HF Space (14 GiB cap). Went to 1.7B. Fits easily, trains 5x faster. The reward function doesn't change with model size so someone with better hardware should try this at 8B.

I wanted curriculum learning too. Start easy, add harder episodes as calibration improves. Half-wrote the code, ran out of time.

There's also an inference-time `LogitsProcessor` that hard-caps `<think>` tokens. Force-emits `</think>` when budget runs out. The trained model handles it fine because it already knows how to be brief. The untrained model falls apart mid-sentence. There's a slider on the Space if you want to try it.

## The format problem (and the fix)

Early GRPO runs had ~30% of steps producing zero reward. The model would emit malformed tool calls, skip the `<budget_prediction>` tag entirely, or produce `<think>` blocks without a following action. These steps are pure noise — the policy gradient gets nothing from them.

The fix was obvious in retrospect: teach the model the output format first, *then* do RL. Two-phase training:

1. **SFT warmup** (48 demonstrations, 3 epochs, ~1 hour). I generated synthetic trajectories from the CVE dataset — correct metacognitive behavior on each file (short/skip on safe, long/flag on buggy). The model learns: `<budget_prediction>` → `<think>` → `<tool_call>`, every time.

2. **GRPO** on the format-aware model. Now every step produces a parseable completion. The reward function can actually score calibration and difficulty. Zero-reward rate drops from 30% to under 5%.

The SFT data is 48 examples, ~400 files total. Not a lot. But you don't need a lot when the target is format compliance, not domain knowledge. The RL phase handles the hard part — learning *when* to predict long vs short.

## Stack

- OpenEnv 0.2.3, MCPEnvironment + FastMCP
- TRL >= 0.17, GRPOTrainer with live env in the reward loop
- bitsandbytes 4-bit, Qwen3-1.7B
- PEFT LoRA r=16 alpha=32
- Gradio 5

`lr=5e-6`, `warmup=0.05`, `KL beta=0.04`, `MAX_SEQ=2048`, `MAX_COMPLETION=1024`. 200 episodes, 1 epoch, ~200 steps. SFT warmup (48 examples, 3 epochs) then GRPO, about 7 hours total on A10G.

## Next

I want to know if you can train with the `<budget_prediction>` tag and then remove it at inference. Does the allocation behavior stick even without the explicit pre-commitment? If yes, that's interesting because it means the reasoning budget is in the weights, not in the scaffolding. Haven't tried it.

## The takeaway

Reasoning LLMs are expensive because they think the same amount on everything. You can fix this without changing the architecture — just add a reward that says "predict difficulty, then be right about it." The model learns. It transfers to new domains. It survives adversarial attacks. And it runs on commodity hardware.

If you're building agentic systems that need to reason under compute constraints — which is basically everyone deploying reasoning models in production — this is a technique you can use today.

---

*Meta PyTorch OpenEnv Hackathon 2026 (India). Theme #5 Wild Card / 3.1 World Modeling.*

**Deep dive:** [`PAPER.md`](https://github.com/subwaycookiecrunch/Meta-project/blob/main/PAPER.md) · **Red team:** [`SAFEGUARDS.md`](https://github.com/subwaycookiecrunch/Meta-project/blob/main/SAFEGUARDS.md) · **Judge checklist:** [`JUDGES.md`](https://github.com/subwaycookiecrunch/Meta-project/blob/main/JUDGES.md)
