# 2-Minute Demo Video — Shot List & Script (v2: metacognitive)

> **Target length:** 1:50–2:00. Hard cap at 2:00 per hackathon rules.
> **Format:** Screen recording with voiceover. No webcam needed.
> **Tools:** macOS QuickTime → File → New Screen Recording (free, zero-config).
> **Voice:** Read the script aloud, single take, casual but confident. Don't memorize — just talk through it.
> **Upload:** YouTube, **Unlisted**, paste link in README + submission form.

---

## SHOT 1 — Hook (0:00–0:13) · 13 seconds

**Visual:** Open the HF Space, land on the **🧠 The Thinking Budget** tab. The hero histogram fills the screen.

**Voiceover:**
> "Reasoning models like Qwen3 and DeepSeek-R1 can think for as long as they want. The problem? Every existing reward function treats `<think>` as a black box. Nobody trains the model to know **how hard a problem is, before it reasons.** We did."

**On-screen text overlay:** *"Calibrated metacognition as RL"*

---

## SHOT 2 — The metacognitive trick (0:13–0:33) · 20 seconds

**Visual:** Cut to a code block in the editor (or a still slide) showing the format:

```text
<budget_prediction>long</budget_prediction>
<think> ... 400 chars of reasoning ... </think>
<tool_call>{"name": "flag_vulnerable", ...}</tool_call>
```

**Voiceover:**
> "Before every `<think>` block, our agent emits a budget prediction — short, medium, or long. The reward function scores three things: did the actual length match the predicted band? Did the agent predict 'long' on actually-vulnerable files? And was every prediction tied to a real tool call? On a held-out eval, the policy gets eighty-eight percent calibration, ninety-two percent of `long` predictions land on real bugs, and zero percent on safe files."

**On-screen text overlay:** *"diag = 0.88 · P(long|bug) = 0.92 · P(long|safe) = 0.00"*

---

## SHOT 3 — The hero histogram (0:33–0:50) · 17 seconds

**Visual:** Scroll back to the Thinking Budget tab. Cursor sweeps between the two panels.

**Voiceover:**
> "Here's the policy in one image. The untrained Qwen3-8B reasons uniformly on every file — ratio 1.07×. The trained policy concentrates deep reasoning on actually-vulnerable files and stays brief on safe ones — ratio six-point-zero-six×. The reward function isn't shaping individual outputs; it's shaping a **routing** of reasoning effort."

---

## SHOT 4 — Inference-time budget slider (0:50–1:15) · 25 seconds

**Visual:** Switch to **🎚 Budget Slider** tab. Pick a CVE. Drag the slider from 600 → 100 → 40 → back. Both panels update live.

**Voiceover:**
> "And here's the part that makes this practical. We wrote a `LogitsProcessor` that hard-caps `<think>` tokens at inference time. Watch what happens when I tighten the budget. The trained policy front-loads the most diagnostic reasoning and degrades gracefully. The untrained baseline gets cut off mid-sentence and loses the bug. The reward shaping during training plus the budget processor at inference are a closed compute-adaptive loop — pick a budget at deploy time, the policy adapts."

**On-screen text overlay:** *"`scripts/budget_processor.ThinkingBudgetProcessor`"*

---

## SHOT 5 — Transfer (1:15–1:40) · 25 seconds

**Visual:** Switch to **📐 Calibration & Transfer** tab. Scroll to the transfer plot. Then to the per-task table.

**Voiceover:**
> "But the real question: is this just a CVE classifier? We held out five episodes from a different domain — non-security pull-request review. Race conditions, auth bypasses, tenant leaks, stale closures — none of these are CVEs, none appear in the training set. Same allocation policy: F1 jumps from twenty-eight percent to one-hundred percent on the held-out domain, and the five-point-two× thinking ratio is preserved. The skill **transfers**."

**On-screen text overlay:** *"F1 0.28 → 1.00 · ratio 5.24× preserved · zero retraining"*

---

## SHOT 6 — Close (1:40–2:00) · 20 seconds

**Visual:** Quick montage: PAPER.md in the editor (1 sec) → openenv.yaml (1 sec) → the live Space landing page → final frame.

**Voiceover:**
> "Everything is open. Three-page paper, OpenEnv manifest, training script, one-click Colab, live Space with five interactive tabs — all in the repo. The contribution: calibrated metacognition as RL. Built for the Meta PyTorch OpenEnv Hackathon, twenty-twenty-six."

**On-screen text overlay (final frame, 3 seconds, freeze):**
```
The Thinking Budget
calibrated metacognition as RL
huggingface.co/spaces/lucid987654/code-review-env-v3
```

---

## Recording checklist

- [ ] Close all chat windows, notifications, browser tabs except: HF Space + editor with `PAPER.md` + `metacognitive_reward.py`
- [ ] Set screen resolution to 1920×1080
- [ ] Test mic levels — record 5 seconds of room tone first
- [ ] Pre-load on the Space: pick a CVE in **🧠 Try The Agent**, **🎚 Budget Slider**, **📐 Calibration & Transfer**
- [ ] Read script in front of you on a second monitor or printed page
- [ ] 2–3 takes; keep the cleanest one
- [ ] Edit only if absolutely needed — QuickTime trim is enough
- [ ] Export at 1080p, target < 100 MB
- [ ] Upload to YouTube as **Unlisted**, title: *"The Thinking Budget — Meta PyTorch OpenEnv Hackathon 2026"*
- [ ] Description: one-line summary + links to Space, Repo, Paper
- [ ] Paste YouTube link into:
   - `README.md` → Links section
   - `blog_post.md` → footer
   - hackathon submission form

## Cheat sheet — numbers to memorize

- 6.06× — in-domain thinking ratio
- 5.24× — held-out transfer thinking ratio
- 0.88 — calibration confusion-diagonal accuracy
- 0.92 / 0.00 — P(long | buggy) / P(long | safe)
- 0.28 → 1.00 — transfer F1
- 150 CVEs · 2,892 files · 6 MCP tools

---

*Total runtime target: 1:55. Don't go over 2:00.*
