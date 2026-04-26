---
title: "I taught a 1.7B model to know when not to think hard"
thumbnail: /blog/assets/thinking-budget/hero.png
authors:
- user: lucid987654
  name: Shri Raj Bisaria
  solo_participant: true
---

# I taught a 1.7B model to know when not to think hard

## The problem I couldn't stop noticing

I was debugging a code review agent one night — the kind that reads source files and flags security bugs. Standard stuff. Qwen3 under the hood, running through files one by one.

And I noticed something stupid.

The model was writing three paragraphs of analysis on `extern int x;`. A single line declaration. No logic, no pointers, no memory ops. Just a type and a variable name. Three paragraphs.

Then it got to a file with an actual buffer overflow — `copy_from_user` piping into `kmalloc` with a user-controlled size — and spent about the same amount of time on it. Same length of reasoning. Same effort.

It was like watching someone study for an exam by spending equal time on every page of the textbook, including the table of contents.

That's when the idea clicked: what if the model could learn *where* to spend its thinking? Not just what to think, but how much.

![How the trained model allocates thinking — bugs get deep analysis, safe files get skimmed](https://raw.githubusercontent.com/subwaycookiecrunch/Meta-project/main/grpo_output/thinking_allocation.png)

[Try it yourself →](https://huggingface.co/spaces/lucid987654/code-review-env-v3) · [Source code](https://github.com/subwaycookiecrunch/Meta-project)

---

## The idea, explained like I'd explain it to my mom

Imagine you're a doctor in an emergency room. Twenty patients walk in. Some have a cold. Some have chest pain. You don't spend 45 minutes examining the runny noses — you triage. You look at each person for a few seconds, decide how serious it is, and allocate your time accordingly.

That's exactly what reasoning models don't do. They "examine" every problem with the same intensity. A trivial question gets the same 2,000-word internal monologue as a genuinely hard one.

My fix: before the model starts thinking about a file, make it say out loud how hard it thinks the problem will be. Three options — `short`, `medium`, or `long`. Then, after it's done thinking, check whether its prediction was actually right.

Got it right? Good, here's a reward. Said "short" and then wrote 500 words anyway? That's a penalty. Said "long" on something trivial? Also a penalty.

Over time, the model learns to match its effort to the difficulty. It becomes a better triage doctor.

---

## What I actually built

The whole system is a security code review environment. Here's the setup:

I took 150 real security vulnerabilities from the National Vulnerability Database — things like Log4Shell, Dirty COW, PwnKit. Each one comes with a set of source files. Some files contain the bug, most don't.

The AI agent gets a description of the vulnerability and a list of file names, but it can't see the code. It has to actively choose to read each file, which costs "investigation points." Budget is limited — you can't read everything. This forces the agent to be strategic.

Before examining each file, the model has to commit to a thinking budget:

```
<budget_prediction>long</budget_prediction>
<think>
This file handles user input directly and the CVE mentions
integer overflow. The function at line 412 calls copy_from_user 
with an attacker-controlled size parameter. Classic heap overflow.
</think>
→ flags the file as vulnerable
```

The reward has three parts that work together:

1. **Calibration** — did you predict `long` and then actually think a lot? Or did you say `short` and keep it brief? Match your prediction to your behavior.

2. **Difficulty awareness** — did hard files get `long` predictions and easy files get `short`? Don't just be consistent with yourself — be right about the world.

3. **Action coupling** — did each prediction lead to a real tool call? No empty predictions. You have to actually *do* something after committing to a budget.

That third one sounds minor. It wasn't. More on that later.

---

## The first attempt that failed spectacularly

My first reward function was simple: +0.8 for correctly skipping a safe file, +1.0 for catching a bug, -1.0 for missing one.

The model figured out the exploit within 30 training steps.

Most files in any codebase are safe. If you just skip everything, you get +0.8 on ~85% of files and -1.0 on ~15%. Net positive. The model learned to be a terrible security reviewer who rubber-stamps everything as fine.

I should've seen it coming. I didn't.

---

## Trying to break my own system (before someone else did)

After fixing the obvious reward problems, I got paranoid. I didn't want to train for 7 hours and then discover the model found another shortcut.

So I wrote a red team. Five different attack strategies, all designed to exploit the reward function:

**Attack 1: "Flag everything, predict long."** Just say every file is buggy with maximum thinking. You'll catch all the real bugs, right? Sure, but the F1 score tanks because you're also flagging tons of safe files.

**Attack 2: "Skip everything, predict short."** The opposite. You'll miss every bug.

**Attack 3: "Perfect predictions, no tool calls."** This one was sneaky. Emit beautiful budget predictions — `short` on safe files, `long` on buggy ones — but never actually read any files or flag anything. Just… predict well and do nothing.

**Attack 4: "Do the job, but pad the thinking."** Actually find the bugs, actually flag the right files, but stuff your reasoning blocks with garbage text to hit the `long` character count. Real work with fake depth.

**Attack 5: "Invert everything."** Predict `long` on safe files and `short` on bugs. See if the reward catches it.

Results:

| Strategy | Score |
|---|---:|
| Honest metacognitive policy | **0.850** |
| Garbage padding (Attack 4) | 0.662 |
| Flag everything (Attack 1) | 0.426 |
| Skip everything (Attack 2) | 0.278 |
| Invert difficulty (Attack 5) | 0.192 |
| Predict without acting (Attack 3) | 0.076 |

The honest policy won by a mile. The padding attack was the closest because it actually does the job right — it just fakes the reasoning. But even that scored 22% lower.

Attack 3 — the one I was most worried about — got demolished. That's where the action coupling reward earned its keep. Without it, the model could farm calibration and difficulty points by making perfect predictions and never doing anything. With it, orphan predictions get their metacognitive score cut in half. I almost didn't add that multiplier. Glad I did.

---

## The training itself (and the crash that ate two hours)

I used GRPO — Group Relative Policy Optimization. It's like giving the model several attempts at the same problem, ranking them, and pushing the weights toward the better attempts.

First problem: the model kept crashing with a dtype error.

```
RuntimeError: expected scalar type Float but found BFloat16
```

I spent two hours casting every parameter to the right type before I found out that PEFT's `prepare_model_for_kbit_training` was silently undoing my casts. By design. It upcasts everything to fp32 for QLoRA stability.

The fix was a forward hook that re-casts on the fly. Ugly, but it works, and it survives PEFT updates.

Second problem: about 30% of training steps produced garbage — malformed outputs that the reward function couldn't even parse. The model hadn't learned the output format yet, so it was emitting broken tool calls and missing tags.

The fix was straightforward once I thought of it: teach the format first, *then* train the reasoning. I wrote 48 example trajectories — here's what a correct interaction looks like, step by step — and fine-tuned on those for three epochs. About an hour. After that, the model knew the grammar. GRPO could focus on the hard part — learning *when* to predict short versus long.

### The training curve

![Training Curves](https://huggingface.co/spaces/lucid987654/code-review-env-v3/resolve/main/grpo_output/training_curves.png)

200 steps on a single A10G GPU. You can see the reward trending upward — not dramatically, this isn't a loss curve going to zero — but consistently. The non-zero rate (how often the model produces scoreable output) climbs from around 60% to 83%. Peak reward hits 0.252 at step 129.

---

## The results

Before training: the model thinks about 170 characters on every file, regardless. Buggy files, safe files, doesn't matter. The thinking ratio between bugs and safe files is 1.07x. Basically flat.

After training: 473 characters on buggy files, 78 on safe ones. A 6x ratio.

The model learned to skim safe files and deep-dive on suspicious ones. Not because I told it which files are buggy — it figured out the correlation between file characteristics and bug likelihood, and started allocating its reasoning budget accordingly.

The calibration numbers back this up. When the model predicts `long`, there's a 92% chance the file is actually buggy. When it predicts `short`, there's a 0% chance. It went from random (33% accuracy — three options, one-in-three guess) to 88% on the diagonal.

Bug detection F1 went from 0.14 to a perfect 1.00 across the evaluation episodes.

![The full picture — reward convergence, F1 improvement, transfer results, and adversarial robustness](https://huggingface.co/spaces/lucid987654/code-review-env-v3/resolve/main/grpo_output/improvement_panel.png)

---

## The part where it gets interesting (and where I'm being careful)

Here's the question I really wanted to answer: is this a general reasoning skill, or did the model just memorize which CVE files are buggy?

I tested it on 5 completely new episodes. Different domain. A payment processing race condition, a JWT authentication bypass, an ML pipeline seed issue, a React stale closure bug, a SQL tenant filter vulnerability. None of these were in the training data. Different programming languages, different bug types entirely.

Baseline F1: 0.28. Trained policy: 1.00. Thinking ratio went from 1.29x to 5.24x.

Now — five episodes is a tiny sample. I want to be upfront about that. Perfect F1 on 5 cases means it got all 5 right, which is good but not statistically ironclad. I'd want 50-100 episodes across more domains before making strong claims.

But what gives me some confidence is that the *pattern* showed up in all five. The model was consistently brief on easy files and verbose on hard ones, even in domains it had never seen. That's not what memorization looks like.

---

## What I'd do differently

I started with Qwen3-8B. Hit the memory ceiling on the HuggingFace Space (14 GB limit). Dropped to 1.7B. It fits easily and trains 5x faster. The reward function is the same regardless of model size, so someone with a beefier GPU should try this at 8B — I'd bet the transfer results get even stronger.

I also wanted to add curriculum learning — start with easy episodes and ramp up difficulty as calibration improves. Half-wrote the code, ran out of time.

There's a question I keep coming back to: can you train with the budget prediction tag and then *remove it* at inference? Does the model still allocate its thinking correctly even without the explicit pre-commitment step? If yes, that means the reasoning allocation is baked into the weights, not dependent on the scaffolding. Haven't tested it yet, but I think the answer is yes.

---

## The bigger picture

Reasoning models are expensive because they think the same amount on everything. An easy question and a hard question get the same multi-thousand-token internal monologue. That's wasteful if you're paying per token, and it's slow if you're running inference at scale.

This project shows you can fix that without changing the model architecture. No custom attention heads, no mixture-of-experts routing, no early-exit mechanisms. Just a reward signal that says: "predict how hard this will be, then prove you were right."

The model learned. It transferred to new domains. It survived five different adversarial attacks. And it ran on a single commodity GPU.

If you're building anything that needs a language model to reason under compute constraints — which, honestly, is most production deployments of reasoning models right now — this is a technique you can use today.

---

*Built for the Meta PyTorch OpenEnv Hackathon 2026 (India) · Theme 3.1 World Modeling*

**Want the details?** [`PAPER.md`](https://github.com/subwaycookiecrunch/Meta-project/blob/main/PAPER.md) · **How I tried to break it:** [`SAFEGUARDS.md`](https://github.com/subwaycookiecrunch/Meta-project/blob/main/SAFEGUARDS.md) · **Judge checklist:** [`JUDGES.md`](https://github.com/subwaycookiecrunch/Meta-project/blob/main/JUDGES.md)
