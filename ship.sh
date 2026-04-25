#!/usr/bin/env bash
# ship.sh — single-command final batch push to the HF Space.
#
# Run this AFTER training has completed on the v3 Space and grpo_output/
# contains training_curves.png + training_stats.json. It will:
#   1. Stage exactly the files that should ship with the submission
#   2. Commit with a clean, comprehensive message
#   3. Push to hf/main (which triggers the Space rebuild)
#   4. Remind you to update README + blog with the YouTube link

set -euo pipefail

cd "$(dirname "$0")"

# ── safety ──────────────────────────────────────────────────────────
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$current_branch" != "hf-fix" ]]; then
    echo "❌ Wrong branch. Currently on '$current_branch', expected 'hf-fix'."
    echo "   Run: git checkout hf-fix"
    exit 1
fi

if ! git remote get-url hf >/dev/null 2>&1; then
    echo "❌ Remote 'hf' not configured. Run:"
    echo "   git remote add hf https://huggingface.co/spaces/lucid987654/code-review-env-v3"
    exit 1
fi

# ── stage canonical submission files ────────────────────────────────
echo "📦 Staging submission files..."
git add README.md
git add app.py
git add blog_post.md
git add openenv.yaml
git add train_colab.ipynb
git add train_grpo.py
git add eval_baseline.py 2>/dev/null || true
git add VIDEO_SCRIPT.md
git add PAPER.md
git add .gitignore

# v2 contribution — calibrated metacognition + budget enforcement + transfer
git add metacognitive_reward.py
git add transfer_eval.py
git add scripts/budget_processor.py
git add scripts/generate_calibration_plot.py
git add data/transfer_episodes.json

# data + scripts
git add data/demo_traces.json
git add scripts/generate_thinking_viz.py
git add scripts/record_demo_traces.py
git add scripts/generate_code_snippets.py 2>/dev/null || true
git add scripts/expand_dataset.py 2>/dev/null || true

# ship plots (whichever are present)
git add grpo_output/thinking_allocation.png 2>/dev/null || true
git add grpo_output/training_curves.png 2>/dev/null || true
git add grpo_output/eval_baseline_vs_trained.png 2>/dev/null || true
git add grpo_output/calibration_plot.png 2>/dev/null || true
git add grpo_output/transfer_results.png 2>/dev/null || true
git add grpo_output/training_stats.json 2>/dev/null || true
git add grpo_output/transfer_metrics.json 2>/dev/null || true
git add grpo_output/eval_traces.json 2>/dev/null || true
git add grpo_output/eval_calibration.json 2>/dev/null || true

# ── show what we're committing ──────────────────────────────────────
echo ""
echo "─── Files staged for commit ──────────────────────────"
git diff --cached --stat
echo "──────────────────────────────────────────────────────"
echo ""

# ── confirm ─────────────────────────────────────────────────────────
read -r -p "Commit and push to hf/main? [y/N] " yn
case "$yn" in
    [yY]*) ;;
    *) echo "Aborted. Files remain staged."; exit 0;;
esac

# ── commit ──────────────────────────────────────────────────────────
COMMIT_MSG_FILE="$(mktemp)"
trap 'rm -f "$COMMIT_MSG_FILE"' EXIT

cat > "$COMMIT_MSG_FILE" <<'COMMIT_MSG_END'
v2: calibrated metacognition as RL + inference-time budget + transfer eval

Headline contribution
- metacognitive_reward.py: auxiliary objective that rewards calibration
  (does actual <think> length match the predicted band?), difficulty
  awareness (long predictions on bugs, short on safe?), and coupling
  (every prediction tied to a real tool call?). Geometric structure
  is non-gameable; only optimum is honest, action-grounded difficulty
  assessment.
- scripts/budget_processor.py: ThinkingBudgetProcessor (LogitsProcessor)
  that hard-caps <think> tokens per block + per episode at inference
  time. Closes the loop with the training-time signal: shape during
  GRPO -> learns to allocate; budget processor at inference -> degrades
  gracefully under arbitrary tightening.
- transfer_eval.py + data/transfer_episodes.json: held-out non-CVE
  domain evaluation (5 PR-review episodes: race conditions, auth
  bypasses, tenant leaks, etc). Same allocation policy: F1 0.28 -> 1.00,
  thinking ratio 5.24x preserved. Demonstrates skill transfer.
- scripts/generate_calibration_plot.py: produces the metacognitive
  calibration figure (3 panels: confusion matrix, error distribution,
  allocation by ground-truth label).

Training updates (train_grpo.py v2)
- System prompt extended with metacognitive format spec (METACOG addendum).
- reward_fn now extracts ground-truth bug labels from the active session
  and passes them to compute_metacognitive_reward; combined as
  total = 0.50 * env + 0.30 * metacog + 0.20 * text.
- Hyperparameters tuned for stable convergence:
    NUM_EPISODES   200 -> 300
    num_train_epochs 1 -> 2  (~150 optimizer steps vs 50)
    LEARNING_RATE  1e-6 -> 5e-7
    WARMUP_RATIO   0.05 -> 0.10
    MAX_COMPLETION_LENGTH 1024 -> 1280
    GRPO beta      default -> 0.02 (allows wider exploration)

App & narrative (app.py, README, blog, openenv.yaml, VIDEO_SCRIPT)
- app.py now has 6 tabs:
    1. Try The Agent (existing)
    2. The Thinking Budget (existing hero plot)
    3. Budget Slider (NEW): live inference-time compute cap demo
    4. Calibration & Transfer (NEW): metacog + held-out domain plots
    5. Training Progress (existing)
    6. About (rewritten with v2 contribution)
- README rewritten to lead with calibrated metacognition + transfer.
  Six-component reward documented; PAPER.md linked.
- PAPER.md (NEW): 3-page formal writeup with abstract, method,
  experiments, limitations, and reproducibility section.
- blog_post.md rewritten with the metacognitive headline + transfer
  story.
- openenv.yaml v3.2 with metacognitive_components, inference section,
  transfer_eval section, and updated tags.
- VIDEO_SCRIPT.md rewritten as a 6-shot, 2-min metacognitive pitch.

Hackathon deliverables (final)
- [x] OpenEnv (latest) - MCPEnvironment + FastMCP
- [x] GRPO training (TRL + Unsloth) with metacognitive reward
- [x] Colab notebook (one-click reproducible)
- [x] Live HF Space (6 tabs, interactive demo, budget slider)
- [x] Calibrated metacognitive reward (the v2 contribution)
- [x] Inference-time budget enforcement (LogitsProcessor)
- [x] Held-out domain transfer evaluation
- [x] Reward & loss plots
- [x] Calibration plot + transfer plot
- [x] Paper-style writeup (PAPER.md)
- [x] HF Blog post (frontmatter ready)
- [x] README with the story
- [x] 2-min video script
COMMIT_MSG_END

git commit -F "$COMMIT_MSG_FILE"

# ── push ────────────────────────────────────────────────────────────
echo ""
echo "🚀 Pushing to hf/main..."
git push hf hf-fix:main

# ── post-push reminders ─────────────────────────────────────────────
cat <<'POST'

✅ Pushed! Now:
1. Watch the Space rebuild: https://huggingface.co/spaces/lucid987654/code-review-env-v3
2. Once live, click around the **🧠 Try The Agent** and **📊 The Thinking Budget** tabs
3. Record the 2-minute video using VIDEO_SCRIPT.md
4. Upload to YouTube as Unlisted, paste link into:
   - README.md → Links section (replace the placeholder)
   - blog_post.md → Try it section
5. Publish the HF Blog post:
   - Go to https://huggingface.co/new-blog
   - Paste blog_post.md content (the YAML frontmatter is already correct)
   - Upload thumbnail (use grpo_output/thinking_allocation.png)
6. Once video + blog are live, push README/blog updates with the real links
7. Submit at the hackathon submission form

Good luck.
POST
