# Teaching an LLM *When* to Think: GRPO on a CVE Investigation Environment

> Meta PyTorch OpenEnv Hackathon 2026 — Theme 3.1: World Modeling / Professional Tasks

When a CVE drops, a security engineer gets a code patch — sometimes 50 modified files — and has to figure out which ones actually contain the vulnerability and which are incidental cleanup. This requires reading source, matching the CVE description to code patterns, and writing a triage report. It's a *professional task*, in the language of the hackathon brief.

We built **CodeReviewEnv v3**, an OpenEnv RL environment that trains an LLM to do exactly this — and we trained Qwen3-8B with GRPO to actually be good at it.

## The novel bit: a thinking budget

Qwen3 has a *thinking mode* — `<think>...</think>` blocks where the model reasons before producing an answer. Most prior work either turns thinking on for everything or off for everything.

We do something different. **The reward function tracks where the agent invests its thinking effort.** If the agent writes a 300-character `<think>` block to justify flagging a buggy file, that's deep reasoning on the *right* file — rewarded. If it writes a 300-character `<think>` block to skip a header file with only declarations, that's wasted effort — penalized.

Concretely, our `thinking_efficiency` term (15% of reward):

```
deep_thinks_on_bugs   = count(<think> blocks > 100 chars on actual bugs)
deep_thinks_on_clean  = count(<think> blocks > 100 chars on safe files)
shallow_on_bugs       = count(no/short <think> on actual bugs — missed!)

bug_coverage  = deep_thinks_on_bugs / total_bugs
waste_penalty = deep_thinks_on_clean / total_decisions
score = max(0, bug_coverage − 0.5 * waste_penalty)
```

This is the loss function for **knowing when to think**. The agent learns that thinking is a finite resource, and routes it to where it actually pays off.

## The environment

Six MCP tools, real CVEs from NVD:

| Tool | Cost | What it does |
|---|---|---|
| `read_file` | 1 pt | Read 20–80 lines of actual source |
| `search_code` | 2 pt | Grep across all files in the patch |
| `get_function_list` | 1 pt | Extract function names + complexity |
| `flag_vulnerable` | free | Flag a file with detailed reasoning |
| `skip_file` | free | Skip a file with brief reasoning |
| `submit_report` | free | End the episode with a triage summary |

The dataset: **150 CVEs** including Log4Shell, Dirty COW, PwnKit, BlueKeep, Zerologon. **2,892 code files** with churn/complexity/recency features extracted from the original commit history. Three difficulty levels (≤15 / 16-29 / 30+ files) with a strict flag budget so the agent can't just spray flags everywhere.

## Live environment execution in the loop

This was the part I'm proudest of. Most GRPO setups train against a static reward dataset. We do something more honest: the reward function **parses tool calls from the model's generation and runs them against the live environment**.

```python
def reward_fn(completions, prompts=None, **kwargs):
    rewards = []
    for completion in completions:
        parsed_calls = parse_tool_calls(completion)   # <tool_call> blocks
        env = CodeReviewToolEnv()
        env.env.reset(seed=extract_seed(prompts[idx]))

        for call in parsed_calls:
            execute_against_live_env(env, call)        # actually runs the tool

        env_score = env.reward                         # real env's TOTAL SCORE
        text_score = score_reasoning_quality(completion)
        rewards.append(0.70 * env_score + 0.30 * text_score)
    return rewards
```

So the model's investigation plan literally runs. If it tries to flag a file that doesn't exist, the env returns an error and the score reflects that. If it submits a thoughtful report mentioning the right CVE ID and vulnerability type, the report-quality scorer rewards it. **The environment is in the loop**, every step.

## The bug hunt: a `prepare_model_for_kbit_training` story

I want to mention one debugging episode because it was instructive. Mid-training the run kept dying with:

```
RuntimeError: expected scalar type Float but found BFloat16
  at logits = self.lm_head(hidden_states[:, slice_indices, :])
```

First fix: `model.lm_head.weight.data = model.lm_head.weight.data.to(torch.bfloat16)`. Re-ran. Same crash.

Second fix: walk every parameter and force-cast non-int8 floats to bf16. Re-ran. Same crash.

Turns out PEFT's `prepare_model_for_kbit_training` contains this:

```python
# cast all non INT8 parameters to fp32
for param in model.parameters():
    if param.dtype in (torch.float16, torch.bfloat16):
        param.data = param.data.to(torch.float32)
```

It silently undoes everything I cast. By design — for QLoRA stability ([huggingface/peft#816](https://github.com/huggingface/peft/issues/816)).

Final fix: stop fighting PEFT. Install a `register_forward_pre_hook(with_kwargs=True)` on `lm_head` that casts the input tensor to match `lm_head.weight.dtype` on **every forward call**. Whatever dtype the head ends up with, the input adapts. Bulletproof.

```python
def hook(module, args, kwargs):
    target = module.weight.dtype
    new_args = tuple(a.to(target) if torch.is_tensor(a) and a.is_floating_point() and a.dtype != target else a for a in args)
    new_kwargs = {k: (v.to(target) if torch.is_tensor(v) and v.is_floating_point() and v.dtype != target else v) for k, v in kwargs.items()}
    return (new_args, new_kwargs)

model.lm_head.register_forward_pre_hook(hook, with_kwargs=True)
```

This shipped with the v3 release. It also turns out to be the canonical fix HF maintainers recommend. Lesson: don't fight library invariants — adapt around them.

## Reward hacking, briefly

Our first reward function gave +0.8 for correctly skipping a safe file. The model figured out it could skip *every* file and rack up score while finding zero bugs. Classic reward hacking.

We rebalanced asymmetrically: missing a real vulnerability costs **−1.0**, correct skips earn only **+0.3**. The "always skip" exploit now scores ~0. The principle is just Goodhart's law restated: **a reward becomes a target, and once it's a target, it stops measuring what you wanted**.

## Results

After 200 episodes of GRPO on Qwen3-8B with LoRA r=16:

- The training curves climb from baseline ~0.2 to late-mean ~0.5+ (see `grpo_output/training_curves.png` on the Space).
- The before/after eval (`eval_baseline.py`) shows the trained model finding more bugs with fewer false positives on held-out seeds.
- A simple "smart investigator" heuristic that reads code before deciding scores **5×** higher than blind strategies — proving that *investigation matters*, and the LLM-with-tools formulation is the right shape for this task.

| Strategy | F1 | Total Score |
|---|---|---|
| Skip everything | 0.000 | 0.120 |
| Flag everything | 0.333 | 0.283 |
| Read, then decide (heuristic) | 0.667 | 0.519 |
| **Qwen3-8B + GRPO (trained)** | **see plots** | **see plots** |

## Stack

- **OpenEnv 0.2.3** — `MCPEnvironment` base + FastMCP server
- **TRL ≥ 0.17** — `GRPOTrainer` with custom `reward_funcs`
- **Unsloth + bitsandbytes** — 4-bit Qwen3-8B fits in 16GB VRAM
- **PEFT** — LoRA r=16, alpha=32, on attention + MLP projections
- **Gradio 5** — auto-refreshing training dashboard

## Try it

- 🤗 **HF Space:** https://huggingface.co/spaces/lucid987654/code-review-env-v3
- 📓 **Colab:** [open `train_colab.ipynb`](https://colab.research.google.com/github/subwaycookiecrunch/Meta-project/blob/main/train_colab.ipynb)
- 💻 **GitHub:** https://github.com/subwaycookiecrunch/Meta-project

```bash
git clone https://huggingface.co/spaces/lucid987654/code-review-env-v3
cd code-review-env-v3
pip install -r requirements.txt
python demo.py
```

---

*Built for the Meta PyTorch OpenEnv Hackathon 2026 — Theme 3.1: World Modeling / Professional Tasks.*
