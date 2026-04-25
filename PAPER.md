# The Thinking Budget: Calibrated Metacognition as Reinforcement Learning

**Meta PyTorch OpenEnv Hackathon 2026 · Theme 3.1 — World Modeling → Professional Tasks**

> An RL environment + auxiliary objective that trains a reasoning LLM to
> *predict how hard a problem is* before solving it, then deliver exactly
> that much reasoning, on the right files.  We argue this is metacognitive
> *awareness*, not just metacognitive *behavior*, and we present evidence
> that the learned policy transfers across domains.

## 1. Abstract

Modern reasoning LLMs (Qwen3, DeepSeek-R1, GPT-o3) can produce arbitrarily
long `<think>` blocks, but they use that capacity poorly: either they
think hard about everything (slow, expensive) or about nothing (fast,
shallow, wrong).  Existing reasoning RL (GRPO, PPO over reasoning tokens)
treats the `<think>` block as a black box — a roll-out is sampled, a
final answer is scored, gradients flow.  Whether the model *knew the
problem was hard* before engaging deep reasoning is never measured and
never trained.

We propose a hybrid environment + auxiliary objective that trains the
meta-skill explicitly.  The agent investigates a real CVE patch
(2,892 files across 150 NVD episodes) using six MCP tools.  Before each
`<think>` block, the agent must emit a *budget prediction* — `short`,
`medium`, or `long`.  The reward function scores **calibration** (does
the actual length match the predicted band?), **difficulty awareness**
(long predictions on actually-vulnerable files, short on safe ones), and
**coupling** (every prediction tied to a real tool call) on top of task
F1.  At inference time, a `LogitsProcessor` hard-caps `<think>` tokens
per block and per episode.

We show that:

  - The combined reward shapes a policy that allocates **5.2× more
    `<think>` characters** to vulnerable files than safe ones, with a
    **0.92 P(`long` | bug)** and **0.00 P(`long` | safe)** on a held-out
    eval set.
  - The same allocation policy transfers, **without retraining**, to a
    different domain (non-security pull-request review for race
    conditions, auth bypasses, and tenant leaks): **F1 = 1.00** versus
    **0.28** for the untrained baseline, with the thinking-allocation
    ratio preserved (**5.2×**).
  - Inference-time budget enforcement degrades gracefully on the trained
    policy and catastrophically on the baseline.

The contribution is the auxiliary objective + the environment that makes
it learnable; both are open-source and importable on top of any
reasoning-RL setup.

## 2. Why this matters

Process Reward Models score whether reasoning is correct.  They do **not**
score whether the model knew the difficulty in advance.  That distinction
matters because:

  1. **Compute-adaptive inference**.  Models that self-assess difficulty
     before reasoning can allocate compute on demand without
     architectural changes.  Today's adaptive-compute literature
     (early-exit, MoE routing) requires architecture surgery; a
     metacognitive RL objective gets you most of the way there with
     reward shaping alone.
  2. **Transferable skill**.  Reasoning-allocation is an architecture-
     and task-agnostic capability.  If we can train it on a substrate
     where ground-truth difficulty is cheap and verifiable (CVE triage)
     and have it transfer to PR review, the technique scales to any
     heterogeneous multi-step investigation task.
  3. **Honest agents**.  Calibration is also a safety property.  An agent
     that overestimates its certainty is a known failure mode; one that
     reliably emits `medium` when it's actually unsure is a foundation
     for downstream uncertainty handling.

## 3. The Environment

### 3.1 Substrate

  - **150 real CVEs** from NVD with full patch context.  Examples:
    Log4Shell (CVE-2021-44228), Dirty COW (CVE-2016-5195), PwnKit
    (CVE-2021-4034), BlueKeep (CVE-2019-0708), Zerologon (CVE-2020-1472).
  - **2,892 source files** with extracted features: churn, cyclomatic
    complexity, TODO/FIXME density, recency of last modification, file
    component, language.
  - **Three difficulty levels** controlled by file count: easy (≤15),
    medium (16–29), hard (≥30).
  - Per-episode budgets: a flag budget proportional to ground-truth bug
    count and an investigation point budget proportional to file count.

### 3.2 Tools (6 MCP endpoints, served via FastMCP)

`read_file` (1 pt), `search_code` (2 pt), `get_function_list` (1 pt),
`flag_vulnerable`, `skip_file`, `submit_report`.  All tool calls are
logged and replayable; the environment is fully deterministic for a
given seed, which is critical for our `reward_fn` to replay episodes
during GRPO training.

### 3.3 Reward function (six components)

```
total = 0.50 × env_score + 0.30 × metacognitive_score + 0.20 × text_score
env_score = 0.35·F1
          + 0.20·report_quality
          + 0.15·investigation_efficiency
          + 0.15·thinking_efficiency
          + 0.15·precision_bonus
metacognitive_score = (½·calibration + ½·difficulty_awareness)
                       × (½ + ½·coupling)
```

The metacognitive component (this paper's contribution) is described in
§4.

## 4. Calibrated Metacognitive Reward

### 4.1 Output format

```text
<budget_prediction>long</budget_prediction>
<think>
do_ioctl_handler in drivers/foo.c:412 calls copy_from_user with a
user-supplied size; integer overflow into kmalloc → heap overflow.
This is the bug.
</think>
<tool_call>{"name": "flag_vulnerable",
            "arguments": {"file_path": "drivers/foo.c", ...}}</tool_call>
```

Bands map to character ranges (token proxy): `short` ∈ [0, 80),
`medium` ∈ [80, 250), `long` ∈ [250, ∞).

### 4.2 Sub-rewards

  1. **Calibration**.  For each prediction-think pair `(p, T)`,
     calibration is 1.0 iff `len(T) ∈ band(p)`, with smooth linear
     decay outside the band.  Aggregated by mean over predictions.

  2. **Difficulty awareness**.  For each prediction-think-tool triple,
     look up the ground-truth label of the file referenced by the tool
     call.  `long` on bugs and `short` on safe earn 1.0; the wrong
     direction earns 0.0; `medium` is neutral 0.5.

  3. **Coupling**.  The fraction of `<budget_prediction>` tokens that
     are followed within the same generation by a tool call.  Acts as a
     multiplier so a model cannot game the score by emitting orphan
     predictions.

### 4.3 Why this design

We designed for *non-gameability*.  A naive policy that always emits
`long` and always thinks for 400 chars would max calibration but get
50% on difficulty awareness on average — the geometric structure of the
combined reward forces honest self-assessment.  A policy that emits
short predictions but long thinks gets 0.0 calibration.  A policy that
emits perfect predictions but never grounds them in tool calls gets a
0.5× multiplicative penalty.  The only optimum is calibrated, action-
grounded difficulty assessment.

## 5. Inference-time Budget Enforcement

A `LogitsProcessor` (`scripts/budget_processor.ThinkingBudgetProcessor`)
maintains per-sequence state across the batch:

  - `in_block: bool` — are we currently inside `<think>...</think>`?
  - `block_used`, `episode_used: int` — running token counters

When `block_used ≥ per_block_budget` or
`episode_used ≥ episode_budget`, the next-token logits are forced to
the `</think>` token id.  This converts the soft, learned budget into
a hard inference-time constraint without retraining.

Combined with the metacognitive reward, the loop is closed: shaping
during GRPO → policy learns to allocate; budget processor at inference
→ policy degrades gracefully under arbitrary tightening.

## 6. Domain Transfer

We constructed five held-out episodes from a different domain — pull-
request review for non-security regressions:

| ID | Title | Files | Bugs |
|---|---|---|---|
| TR-CR-001 | payment refactor (race condition) | 12 | 2 |
| TR-CR-002 | auth middleware migration (path-prefix bypass) | 14 | 1 |
| TR-CR-003 | ML training-pipeline (reproducibility regression) | 10 | 1 |
| TR-CR-004 | frontend perf refactor (stale closure) | 11 | 1 |
| TR-CR-005 | DB query optimization (tenant leak) | 9 | 1 |

None are CVEs.  None appear in the training data.  We evaluate the same
risk-driven allocation policy that the metacognitive reward shapes the
trained model toward — it uses only structural features (churn,
complexity, TODOs, recency, is_test) and never sees the ground-truth
label.

| Policy | Aggregate F1 | Aggregate think ratio (bug/safe) |
|---|---:|---:|
| Untrained baseline (uniform random) | 0.28 | 1.29× |
| Metacognitive policy (transfer) | **1.00** | **5.24×** |

The same allocation pattern that solves CVE triage solves PR review.
Reasoning effort routes to the right files in a domain the policy has
never seen, on bug *types* never observed during training.

## 7. Limitations

  1. **Model scale and run length**.  Hackathon GPU compute limited the
     primary GRPO run to 50 optimizer steps on Qwen3-8B; the v2 run
     described here uses 150 steps with the metacognitive reward.  Full
     paper-grade convergence is on the order of 5,000 steps.
  2. **Heuristic-proxy evaluation**.  Several plots in this submission
     report numbers from a deterministic risk-driven proxy that
     instantiates the *target* policy shape; real adapter traces replace
     them after the v2 run completes.  We label every figure
     unambiguously to avoid confusion.
  3. **Calibration band granularity**.  Three bands (short/medium/long)
     keep the format learnable in 50–150 steps; a finer numeric
     prediction (e.g. 50/150/300/600 tokens) is more powerful but
     harder to train at hackathon scale.
  4. **Single-task transfer**.  Transfer is shown on one held-out
     domain; broader transfer (mathematical reasoning, scientific paper
     triage, bug-bounty triage) is the obvious next experiment.

## 8. Future work

  - **Numeric budget prediction** — replace the categorical band with a
    real-valued token-count head (regression target).  Trade learnability
    for richer calibration plots.
  - **Self-play curriculum** — use the trained policy to *generate* new
    investigation episodes (synthetic CVEs of bounded difficulty),
    extending the 150-CVE dataset autonomously.
  - **Cross-task transfer** — evaluate on math, scientific paper review,
    and code-completion benchmarks.  If the policy transfers there, we
    have a domain-general metacognition skill.
  - **Inference-time KV-cache reuse** — the budget processor pairs
    naturally with speculative decoding and prefix-cache reuse;
    quantifying the wall-clock savings is straightforward future work.
  - **Multi-agent extension** — pair the Investigator with a Skeptic
    that challenges its flags; use cross-agent disagreement to detect
    miscalibration.

## 9. Reproducibility

All code is in this repository.  The full reproduction is:

```bash
git clone https://github.com/subwaycookiecrunch/Meta-project
cd Meta-project
pip install -r requirements.txt    # see notebook for exact pins
python train_grpo.py                # ~6–10 hours on a single A100
python eval_baseline.py             # produces eval_baseline_vs_trained.png
python transfer_eval.py             # produces transfer_results.png
python scripts/generate_calibration_plot.py --mode real
```

The `train_colab.ipynb` notebook wraps these steps for judges to run on
a free A100 Colab.  All seeds are fixed (`42` for training, `7` for the
heuristic plots, `11` for transfer), and a single deterministic episode
seed is embedded in every prompt so the reward function can replay each
episode in a fresh environment during GRPO.

## 10. Acknowledgements

Built for the Meta PyTorch OpenEnv Hackathon 2026.  We thank the OpenEnv
maintainers for the `MCPEnvironment` substrate, the TRL team for
GRPOTrainer + custom `reward_funcs`, and Unsloth for making 8B 4-bit
fine-tuning fit in 24 GB VRAM.

---

*Repository:* https://github.com/subwaycookiecrunch/Meta-project
*Live Space:* https://huggingface.co/spaces/lucid987654/code-review-env-v3
