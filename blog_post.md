---
title: "I taught a 1.7B model to know when not to think hard"
thumbnail: /blog/assets/thinking-budget/hero.png
authors:
- user: lucid987654
  name: Shri Raj Bisaria
  solo_participant: true
---

# I taught a 1.7B model to know when not to think hard

So I was debugging this code review agent late one night. Qwen3 under the hood, scanning source files for security bugs. And I noticed it was writing three full paragraphs of analysis on `extern int x;`. Just — a type and a variable name. Three paragraphs.

Then it hits a file with a real buffer overflow — `copy_from_user` piping into `kmalloc` with a user-controlled size — and spends roughly the same effort on it. Same amount of reasoning. No difference.

It's like watching someone study for an exam by spending equal time on every page of the textbook. Including the index.

What if the model could learn *where* to think hard, not just *what* to think?

![Before vs After — untrained model thinks the same on everything, trained model focuses on bugs](https://raw.githubusercontent.com/subwaycookiecrunch/Meta-project/main/grpo_output/before_after_thinking.png)

[Try it yourself →](https://huggingface.co/spaces/lucid987654/code-review-env-v3) · [Source code](https://github.com/subwaycookiecrunch/Meta-project)

---

## The basic idea

OK so think of an ER doctor. Twenty patients walk in. Some have a cold, some have chest pain. You obviously don't spend 45 minutes on the runny noses. You triage — quick look, decide how serious, then allocate your time.

Reasoning models don't do this at all. Everything gets the same 2,000-word internal monologue regardless of difficulty.

So here's what I did: before the model starts reasoning about a file, I make it commit out loud to how hard it thinks this will be. `short`, `medium`, or `long`. After it's done, I check — did the prediction match? Did hard stuff get `long` and easy stuff get `short`?

Get it right, reward. Say `short` then ramble for 500 words, penalty. Predict `long` on something trivial, also penalty. Over time it learns to match effort to difficulty.

---

## What the environment looks like

It's a security code review setup. 150 real CVEs from the National Vulnerability Database — Log4Shell, Dirty COW, PwnKit, etc. Each comes with source files. Some contain the actual bug, most are clean.

The agent gets the CVE description and file names but can't see any code. Has to call `read_file` to look at anything, and each call costs investigation points. Budget is `2 × number_of_files`, so you can't just read everything. You have to pick.

Before each file the model commits to a thinking budget:

```
<budget_prediction>long</budget_prediction>
<think>
This file handles user input directly and the CVE mentions
integer overflow. The function at line 412 calls copy_from_user
with an attacker-controlled size parameter. Classic heap overflow.
</think>
→ flags the file as vulnerable
```

The reward checks three things. First, calibration — did your prediction match your actual reasoning length. Second, difficulty awareness — did you correctly predict `long` on hard files and `short` on easy ones. Third, and this is the one I almost didn't add, action coupling — every prediction has to be followed by an actual tool call. No predicting without doing.

That third one turned out to be critical. I'll get to why.

(Side note if you're building an OpenEnv: the guide reserves tool names like `reset`, `step`, `state`, `close`. I used `state` as a tool name in my first version and stuff broke silently for an hour. Fun times.)

---

## My first reward function was terrible

+0.8 for correctly skipping a safe file, +1.0 for catching a bug, -1.0 for missing one.

The model figured out the loophole in 30 steps. Most files are safe, right? So just skip everything. You collect +0.8 on 85% of files and eat -1.0 on the other 15%. Net positive. Congrats, you've trained a security reviewer that rubber-stamps everything.

Yeah I should've seen that coming.

---

## Breaking my own reward (the red team)

After fixing the obvious stuff (bumped the miss penalty, lowered the skip reward), I got kind of paranoid. Didn't want to burn 7 hours of GPU time only to discover another loophole.

So I sat down and tried to break it myself. Five attacks:

The "flag everything" attack — just call every file buggy with max thinking. Catches all real bugs but F1 craters because you're also flagging all the safe ones.

The "skip everything" attack — opposite problem. Miss every bug.

The "predict perfectly but do nothing" attack — this was the one I worried about. Generate beautiful budget predictions, `short` on safe files, `long` on buggy ones, but never actually read any files or flag anything. Just… predict well and sit there.

The padding attack — actually find the bugs and flag the right files, but stuff your `<think>` blocks with random garbage to inflate the character count. Real work with fake depth.

And the inverter — predict `long` on safe files, `short` on buggy ones. Just see if the reward catches it.

| Strategy | Score |
|---|---:|
| Honest policy | **0.850** |
| Padding attack | 0.662 |
| Flag everything | 0.426 |
| Skip everything | 0.278 |
| Inverter | 0.192 |
| Predict without acting | 0.076 |

Honest policy wins easily. Padding was closest because it's actually doing the job — just faking the reasoning depth. Still 22% behind though.

The predict-without-acting one — the one I was worried about — got obliterated. And that's entirely because of the action coupling term. Without it, a model could game calibration and difficulty scores while doing zero actual work. With it, orphan predictions get their score halved. I almost left that multiplier out because it felt like overengineering. Really glad I didn't.

Full details in [`SAFEGUARDS.md`](SAFEGUARDS.md) and you can run `python scripts/red_team.py` to reproduce.

---

## Training (and the bug that ate my afternoon)

Used GRPO — you give the model a bunch of attempts at the same problem, rank them, push the weights toward the better ones.

It kept crashing:

```
RuntimeError: expected scalar type Float but found BFloat16
```

I spent *hours* on this. Cast every parameter to bf16 manually. Still crashed. Force-cast everything. Still crashed. Walked the entire model parameter-by-parameter. Nothing worked.

Then I looked at what PEFT's `prepare_model_for_kbit_training` actually does under the hood and found it silently upcasts everything back to fp32 after you downcast. It's intentional — QLoRA stability. Documented in [peft#816](https://github.com/huggingface/peft/issues/816) but not exactly obvious when you're debugging at 3am.

Ended up sticking a forward hook on lm_head that casts inputs on the fly to match the weight dtype. It's ugly and I'm not proud of it but it works and it doesn't break when PEFT updates.

The other problem was that ~30% of training steps were producing total garbage. Malformed tool calls, missing tags, broken JSON. The model didn't know the output format yet so it was just flailing.

Fix: teach the format before teaching the reasoning. Wrote 48 example trajectories, fine-tuned on those for 3 epochs (~1 hour), and after that the model understood the grammar. Then GRPO could focus on the actual hard part — when to predict `short` vs `long`.

### Training curve

![Training Curves](https://huggingface.co/spaces/lucid987654/code-review-env-v3/resolve/main/grpo_output/training_curves.png)

200 steps, one A10G. Reward trends up slowly — it's not a dramatic loss curve, but the direction is clear. Non-zero rate (how often the model produces something the reward function can actually score) goes from ~60% early on to 83% by the end. Best reward was 0.252 at step 129.

---

## What happened

Before training the model thinks ~170 characters on every file. Doesn't matter if it's buggy or safe. Bug-to-safe thinking ratio: 1.07x. Dead flat.

After: 473 chars on buggy files, 78 on safe. 6x ratio.

It figured out the correlation between file features and bug probability on its own and started spending its thinking budget accordingly. I never told it which files were buggy — it learned that from the reward signal.

Calibration went from random (33% — three options, coin flip) to 88% on the diagonal. When the model says `long`, there's a 92% chance the file is actually buggy. When it says `short`, 0% chance it's a bug.

F1 went from 0.14 to 1.00 across the evaluation episodes.

![Full improvement summary](https://huggingface.co/spaces/lucid987654/code-review-env-v3/resolve/main/grpo_output/improvement_panel.png)

One thing I didn't do that I should have: compare against naive truncation. Like, what if you just hard-cut `<think>` at 80 tokens for every file? No prediction, no metacognition, just a brute-force cap. The trained model would almost certainly still win because truncation doesn't help you *detect bugs better* — it just makes you think less. But it deserves a row in the table and I don't have it. Noted for future work.

---

## Does it transfer? (kinda, but read the caveat)

This is the part where I want to be really careful because the numbers look suspiciously good.

Took the trained policy, ran it on 5 brand new episodes that weren't in training. Completely different stuff — payment processing race condition, JWT bypass, ML pipeline seed bug, React stale closure, SQL tenant filter issue. Different languages, different bug types, nothing the model had seen before.

Baseline F1: 0.28. Trained: 1.00. Thinking ratio jumped from 1.29x to 5.24x.

Now look — 5 episodes is tiny. F1 of 1.00 on 5 episodes just means it got all 5 right. That's nice but not statistically robust, and I know that. You'd want 50+ episodes across more domains to really say something.

What makes me think it's not just luck is that the allocation *pattern* held across all five. Short on easy files, long on hard files, consistently. That's not what memorization looks like. But I'm not going to oversell a 5-episode eval.

---

## Things I'd do differently (or didn't get to)

Started with Qwen3-8B. OOM'd on the HuggingFace Space immediately — 14 GiB cap. Dropped to 1.7B, which fits fine and trains 5x faster. Same reward function works at any size so someone with a real GPU should try 8B.

Wanted curriculum learning — start easy, ramp up as calibration improves. Got halfway through the code and ran out of hackathon time.

There's also an inference-time `LogitsProcessor` that hard-caps `<think>` tokens. When budget runs out it force-emits `</think>`. The trained model handles it gracefully because it already knows how to be brief. The untrained model just cuts off mid-sentence. There's a slider on the Space if you want to play with it.

One thing I keep thinking about: what happens if you train with the `<budget_prediction>` tag and then *remove it* at inference? Does the model still allocate correctly without the explicit pre-commitment step?

This is actually a bigger question than it sounds. Right now I can't tell you whether the tag *causes* better allocation or just *correlates* with it. Maybe the model learned to reason better in general and the tag is just along for the ride. Maybe the tag is doing all the work and the reasoning improvement is downstream of it. I don't have the ablation to separate these.

But if you remove the tag and the behavior persists — if the model still thinks briefly on easy files and deeply on hard ones without being asked to predict upfront — then the metacognitive calibration is representational. It's in the weights, not in the scaffolding. That's the version of this that's actually interesting as a research direction. Haven't tested it, but it's first on the list.

---

## Why this matters (if you're building with reasoning models)

Every reasoning model right now — o1, Qwen3 thinking mode, DeepSeek-R1, whatever — spends the same amount of compute on easy problems and hard problems. If you're deploying these in production and paying per-token, that's expensive for no reason. If you're running inference at scale, it's slow for no reason.

This shows you can fix it without touching the architecture. No custom layers, no MoE routing, nothing that requires a from-scratch pretrain. Just a reward that asks: can you predict how hard this is going to be? And then were you right?

Trained on one GPU. Transfers to domains it never saw. Survives adversarial attacks. 1.7 billion parameters.

---

*Built for the Meta PyTorch OpenEnv Hackathon 2026 · Theme 3.1 World Modeling*

More depth: [`PAPER.md`](https://github.com/subwaycookiecrunch/Meta-project/blob/main/PAPER.md) · Red team: [`SAFEGUARDS.md`](https://github.com/subwaycookiecrunch/Meta-project/blob/main/SAFEGUARDS.md) · Judge checklist: [`JUDGES.md`](https://github.com/subwaycookiecrunch/Meta-project/blob/main/JUDGES.md)
