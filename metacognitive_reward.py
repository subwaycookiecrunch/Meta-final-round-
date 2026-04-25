"""
metacognitive_reward.py
========================
Calibrated metacognition as RL.

THE INNOVATION
--------------
Standard reasoning RL (GRPO, PPO over <think>...</think> tokens) treats the
model's reasoning as a black box: a roll-out is sampled, a final answer is
scored, gradients flow.  Whether the model "knew" the problem was hard
*before* engaging deep reasoning is never measured and never trained.

This module trains the meta-skill explicitly.  Before each reasoning block,
the agent must emit an explicit *budget prediction*:

    <budget_prediction>long</budget_prediction>
    <think> ... reasoning ... </think>
    <tool_call>{"name": "flag_vulnerable", ...}</tool_call>

The reward function rewards:
  1. **Calibration** — does the actual <think> length fall in the predicted band?
  2. **Difficulty awareness** — long predictions land on actually-vulnerable
     files; short predictions land on safe files.
  3. **Coupling** — predictions and actions are made on the same files
     (no orphan predictions).

This turns the reward signal from "did you reason well?" into "did you know
in advance how much reasoning the problem deserved, then deliver exactly
that much, on the right files?".  That is metacognitive *awareness*, not
just metacognitive *behavior*.

The contribution is the auxiliary objective; it is trainable on top of any
existing reasoning RL setup.  This module is self-contained and importable
from the main train_grpo.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ── Predicted budget bands (tokens of <think> content, characters proxy) ──
# We use character-length bands as a coarse proxy for token count.  Three
# bands match the "short / medium / long" categorical the model emits.
BAND_RANGES = {
    "short":  (0,   80),
    "medium": (80,  250),
    "long":   (250, 9999),
}
BAND_TARGETS = {  # midpoints used to define monotonic difficulty signal
    "short":  40,
    "medium": 165,
    "long":   400,
}
BAND_ORDER = {"short": 0, "medium": 1, "long": 2}


# ── Regex patterns ────────────────────────────────────────────────────────
RE_PRED_THINK = re.compile(
    r"<budget_prediction>\s*(short|medium|long)\s*</budget_prediction>"
    r"\s*<think>(.*?)</think>",
    re.IGNORECASE | re.DOTALL,
)

RE_PRED_THINK_THEN_FLAG = re.compile(
    r"<budget_prediction>\s*(short|medium|long)\s*</budget_prediction>"
    r"\s*<think>(.*?)</think>"
    r"\s*<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)

RE_LOOSE_PRED = re.compile(
    r"<budget_prediction>\s*(short|medium|long)\s*</budget_prediction>",
    re.IGNORECASE,
)

RE_TOOL_CALL = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL,
)


# ── Metacognitive reward components ───────────────────────────────────────
@dataclass
class MetacogResult:
    calibration: float          # 0..1 — does actual length match predicted band?
    difficulty_awareness: float  # 0..1 — long preds on bugs, short on safe?
    coupling: float              # 0..1 — fraction of preds followed by a tool call
    n_predictions: int           # how many predictions the model emitted
    raw_score: float             # weighted aggregate (0..1)


def _calibration_score(predicted: str, actual_len: int) -> float:
    """
    1.0 if `actual_len` is inside the predicted band; smooth decay outside.
    """
    lo, hi = BAND_RANGES[predicted]
    if lo <= actual_len < hi:
        return 1.0
    if actual_len < lo:
        return max(0.0, actual_len / max(1, lo))
    overshoot = (actual_len - hi) / max(1, hi)
    return max(0.0, 1.0 - overshoot)


def _difficulty_score(predicted: str, file_is_bug: Optional[bool]) -> float:
    """
    Reward "long on bugs, short on safe".  Medium is neutral.
    """
    if file_is_bug is None:
        return 0.5  # no ground truth — neutral
    if predicted == "long":
        return 1.0 if file_is_bug else 0.0
    if predicted == "short":
        return 1.0 if not file_is_bug else 0.0
    return 0.5  # medium


def _extract_filepath_from_tool_call(json_str: str) -> Optional[str]:
    import json as _json
    try:
        data = _json.loads(json_str)
        args = data.get("arguments") or data.get("function", {}).get("arguments", {})
        if isinstance(args, str):
            args = _json.loads(args)
        return args.get("file_path")
    except Exception:
        return None


def _extract_tool_name(json_str: str) -> Optional[str]:
    import json as _json
    try:
        data = _json.loads(json_str)
        return data.get("name") or data.get("function", {}).get("name")
    except Exception:
        return None


def compute_metacognitive_reward(
    text: str,
    bug_files: Optional[set] = None,
) -> MetacogResult:
    """
    Score a model completion's metacognitive behavior.

    Args:
        text: The model's full completion (post-prompt).
        bug_files: Set of ground-truth vulnerable file paths for this episode,
                   or None if unavailable (e.g. text-only fallback).

    Returns:
        MetacogResult with three sub-scores and a weighted raw_score.
    """
    bug_files = bug_files or set()

    # ── 1. Pair predictions with their following <think> blocks ──────────
    pred_think = RE_PRED_THINK.findall(text)
    n_preds = len(RE_LOOSE_PRED.findall(text))
    n_tool_calls = len(RE_TOOL_CALL.findall(text))

    if not pred_think:
        # The model didn't follow the metacognitive format.  Return zero
        # signal but don't penalize so the GRPO loss can still flow from
        # the live-execution reward; this just means metacognition has
        # not been learned yet.
        return MetacogResult(0.0, 0.0, 0.0, 0, 0.0)

    # ── 2. Calibration ────────────────────────────────────────────────────
    calibration_scores = []
    for pred, think_text in pred_think:
        actual_len = len(think_text.strip())
        calibration_scores.append(_calibration_score(pred.lower(), actual_len))
    calibration = sum(calibration_scores) / len(calibration_scores)

    # ── 3. Difficulty awareness ──────────────────────────────────────────
    # For each prediction-then-flag/skip triple, look up the file label.
    diff_scores = []
    coupled = 0
    for pred, think, tool_json in RE_PRED_THINK_THEN_FLAG.findall(text):
        coupled += 1
        fpath = _extract_filepath_from_tool_call(tool_json)
        tool = _extract_tool_name(tool_json) or ""
        is_bug: Optional[bool]
        if fpath is None:
            is_bug = None
        elif fpath in bug_files:
            is_bug = True
        elif tool in ("flag_vulnerable", "skip_file") and bug_files:
            # the file is in the patch but not in bug_files → safe
            is_bug = False
        else:
            is_bug = None
        diff_scores.append(_difficulty_score(pred.lower(), is_bug))

    difficulty_awareness = sum(diff_scores) / len(diff_scores) if diff_scores else 0.0
    coupling = coupled / max(1, n_preds)

    # ── 4. Aggregate ─────────────────────────────────────────────────────
    # Calibration and difficulty are equal-weighted; coupling is a multiplier
    # so a model that emits predictions but never grounds them in actions
    # cannot game the score.
    raw = (0.5 * calibration + 0.5 * difficulty_awareness) * (0.5 + 0.5 * coupling)

    return MetacogResult(
        calibration=calibration,
        difficulty_awareness=difficulty_awareness,
        coupling=coupling,
        n_predictions=n_preds,
        raw_score=max(0.0, min(1.0, raw)),
    )


# ── System-prompt patch ───────────────────────────────────────────────────
METACOG_SYSTEM_PROMPT_ADDENDUM = """

CRITICAL — Metacognitive Format (REQUIRED):

Before EVERY <think> block, you MUST emit a budget prediction first:

    <budget_prediction>short|medium|long</budget_prediction>
    <think>
    ...your reasoning here...
    </think>
    <tool_call>{"name": "...", "arguments": {...}}</tool_call>

Budget bands:
  - short  : 0–80 characters of reasoning. Use for obviously safe files
             (test files, headers with no logic, boilerplate).
  - medium : 80–250 characters. Use when you need to verify but don't
             see strong red flags.
  - long   : 250+ characters. Use when you suspect the file is vulnerable
             and need to lay out the bug pattern (function name, unsafe
             operation, missing check, exploit path).

You will be SCORED on:
  1. Calibration — does the actual length of your <think> match the band
     you predicted?
  2. Difficulty awareness — do you predict 'long' on actually-vulnerable
     files and 'short' on safe ones?
  3. Coupling — every prediction must be followed by a real tool call
     against a file (no orphan predictions).

The optimal policy predicts BEFORE thinking, thinks the predicted amount,
and predicts longer for bugs.  Be honest about the difficulty.
"""


# ── CLI smoke test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick sanity check
    sample = """
<budget_prediction>long</budget_prediction>
<think>
Looking at copy_from_user without a length check on the user-supplied size
parameter. This is a textbook integer overflow → heap overflow primitive.
The CVE description matches exactly. Function `do_ioctl_handler` in line 412.
</think>
<tool_call>{"name": "flag_vulnerable", "arguments": {"file_path": "drivers/foo.c", "reasoning": "ioctl bug"}}</tool_call>

<budget_prediction>short</budget_prediction>
<think>
Header.
</think>
<tool_call>{"name": "skip_file", "arguments": {"file_path": "include/foo.h", "reasoning": "header"}}</tool_call>
"""
    r = compute_metacognitive_reward(sample, bug_files={"drivers/foo.c"})
    print(f"calibration={r.calibration:.2f}  difficulty_awareness={r.difficulty_awareness:.2f}  "
          f"coupling={r.coupling:.2f}  n={r.n_predictions}  raw={r.raw_score:.3f}")
    assert r.raw_score > 0.7, f"smoke test failed: {r}"
    print("✅ smoke test passed")
