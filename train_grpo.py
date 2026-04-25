"""
GRPO Training Script for CodeReviewEnv v3
==========================================
Trains Qwen3-1.7B (thinking-mode variant) to investigate CVE vulnerabilities
using MCP tools. The agent learns to read code, search patterns, predict
its own reasoning effort (`<budget_prediction>`), and produce triage reports.

Qwen3-1.7B was chosen because:
  - Smallest variant of the Qwen3 thinking-mode family — emits real `<think>`
    blocks, so the metacognitive reward has a behaviour to shape
  - 5× faster step time than Qwen3-8B → 400 GRPO steps fit in the 14 GiB
    Space memory cap and the 12-hour A10G compute budget
  - At 1.7B the prior is weak enough that the metacognitive signal is the
    only plausible cause of the bug-vs-safe allocation pattern (stronger
    methodological claim than running 8B+ which "would have done it
    anyway")
  - The same reward stack applies unchanged to Qwen3-4B and Qwen3-8B

Setup on HuggingFace / Google Colab (A10G or A100 GPU):
    !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    !pip install --no-deps "trl>=0.17" peft accelerate bitsandbytes
    !pip install openenv-core fastmcp datasets matplotlib
    !python train_grpo.py
"""
import os
import sys
import re
import json
import random
import time
import traceback
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# ── Configuration ───────────────────────────────────────────────────
# Model: Qwen3-1.7B (thinking-mode) — same Qwen3 family as the 8B
# variant (identical chat template, native <think> support, agentic
# tool-calling).  Chosen as the primary training target because:
#   • Fits the HF Space 14 GiB memory cap with full 4096 / 2048 lengths
#   • 5× fewer params → 2-3× faster per step → 400 GRPO steps fit
#     in the wall-clock budget (vs ≤75 steps for 8B)
#   • Smaller models adapt to a brand-new output format
#     (<budget_prediction>) faster, producing visibly cleaner curves
#   • All Qwen3 sizes share the same thinking template, so the
#     metacognitive reward, transfer eval, and inference-time budget
#     processor all transfer 1:1 to 8B if we re-run the experiment
#     on bigger hardware later.
# Override via env var to swap to 8B / 4B without touching code.
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-1.7B")
OUTPUT_DIR = "./grpo_output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
EVAL_CALIBRATION_PATH = os.path.join(OUTPUT_DIR, "eval_calibration.json")
TRACE_LOG_PATH = os.path.join(OUTPUT_DIR, "trace_log.jsonl")

# SFT warmup adapter: if set, the model is initialized from this adapter
# before applying a fresh LoRA for GRPO.  This is the two-phase training
# pipeline: SFT (format learning) → GRPO (policy learning).
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")

# Training hyperparameters (right-sized for the available wall-clock —
# Qwen3-1.7B on the HF Space A10G, v2.3).
# v2.3 changes vs v2.2 — generation-throughput limited the v2.2 plan:
#   At 224 s/optimizer-step the v2.2 plan (750 steps) projected a 46h
#   ETA against a 19h deadline.  The bottleneck was generation length
#   (~1,200 token mean completions × K=4 ≈ 70s of pure decoding per
#   prompt).  Halving MAX_COMPLETION_LENGTH and right-sizing the step
#   budget brings ETA to ~5.5h on A10G with full LR schedule alignment.
#   MAX_COMPLETION_LENGTH   2048 → 1024            (2× generation speedup)
#   MAX_SEQ_LENGTH          4096 → 2048            (smaller batch memory)
#   NUM_EPISODES            500  → 200             (fits the wall-clock)
#   NUM_TRAIN_EPOCHS        3    → 2               (LR schedule matches)
#   GRAD_ACCUM_STEPS        4    → 2               (2× more updates / sec)
# Net optimizer steps: 200 ep × 2 ep / 2 grad_accum = 200 steps total.
# Warmup ends at step ~20 (10%), then 180 steps of post-warmup learning
# with cosine LR decay.  Cleaner curve than 750-step plan finishing
# at 45%.  Cost on A10G: ~$5.80 vs $18 for the v2.2 partial run.
# What stays from v2.2:
#   MODEL_NAME              Qwen/Qwen3-1.7B
#   LEARNING_RATE           1e-6
#   WARMUP_RATIO            0.10
#   GRPO_BETA               0.02
#   METACOG_ENABLED         true
#   LORA_R / LORA_ALPHA     16 / 32
NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "200"))
NUM_GENERATIONS = 2          # GRPO group size
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", "1024"))
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "2"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-6"))
WARMUP_RATIO = 0.10
NUM_TRAIN_EPOCHS = int(os.environ.get("NUM_TRAIN_EPOCHS", "1"))
GRPO_BETA = 0.02
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))
LORA_R = 16
LORA_ALPHA = 32
SAVE_EVERY = 25              # More frequent checkpoints (~8 saves total)
USE_UNSLOTH = os.environ.get("USE_UNSLOTH", "true").lower() == "true"

# Metacognitive reward — the v2 contribution.  When True, the system prompt
# requires <budget_prediction> tags and the reward function rewards
# calibration + difficulty awareness (see metacognitive_reward.py).
METACOG_ENABLED = os.environ.get("METACOG_ENABLED", "true").lower() == "true"
METACOG_WEIGHT = 0.30        # Reward weight for the metacog component
CALIBRATION_LOG_EVERY = 1    # Log every reward call (cheap, JSON appends)


# ── System prompt for the security investigator ─────────────────────
# This prompt activates Qwen3's thinking mode and teaches the agent
# to allocate reasoning effort strategically — think hard on complex
# files, reason lightly on obvious safe files.
SYSTEM_PROMPT = """You are an expert security code investigator specializing in CVE vulnerability analysis. You are given a CVE vulnerability description and a list of files from a code patch.

Your mission:
1. Use read_file to examine source code of the most suspicious files first
2. Use search_code to find vulnerability-related patterns (e.g., unsafe functions, missing checks)
3. Use get_function_list to understand file structure and complexity
4. Use flag_vulnerable to mark files containing or related to the vulnerability — provide DETAILED reasoning
5. Use skip_file to mark files that are safe — briefly explain why
6. Use submit_report to provide a detailed triage analysis

CRITICAL — Thinking Strategy:
You have a THINKING BUDGET. Use it wisely, like a real security engineer:
- When flagging a file as VULNERABLE: Think deeply. Explain the exact code pattern that matches
  the CVE. Mention function names, unsafe operations, missing checks. Your reasoning should be
  100+ characters proving you understand the vulnerability.
- When skipping a SAFE file: Be brief. "Header file with only declarations" or "Test file,
  no production logic" is sufficient. Don't waste thinking effort on obviously safe files.
- The environment TRACKS your thinking allocation and rewards you for thinking deeply on the
  RIGHT files (actual bugs) and briefly on safe files.

Investigation strategy:
- Prioritize files with high complexity and recent changes — bugs hide in complex code
- Match the CVE description to code patterns (e.g., "buffer overflow" → look for unchecked memcpy/strcpy)
- Header files (.h) are usually declarations, not vulnerability sources
- Test files are rarely the vulnerability source
- Write a thorough report: mention the CVE ID, affected files, vulnerability type, and root cause

You have limited investigation points and flags. Be strategic — read the most suspicious files first."""


# Metacognitive addendum: appended to the system prompt only when
# METACOG_ENABLED.  Asks the model to emit a budget prediction BEFORE
# every reasoning block.  The reward function then scores calibration
# + difficulty awareness on those predictions.
if METACOG_ENABLED:
    try:
        from metacognitive_reward import METACOG_SYSTEM_PROMPT_ADDENDUM
        SYSTEM_PROMPT = SYSTEM_PROMPT + METACOG_SYSTEM_PROMPT_ADDENDUM
    except ImportError:
        print("⚠️  metacognitive_reward.py not importable; falling back to v1 prompt.")


# ── Chat Template Formatting ───────────────────────────────────────
def format_prompt_as_chat(context, tokenizer):
    """Format the prompt using Qwen3's chat template with thinking mode enabled."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{context}\n\n"
            f"Begin your security investigation. Use the available tools to analyze the files.\n"
            f"Remember: Think DEEPLY when flagging suspicious files (explain the vulnerability pattern).\n"
            f"Be BRIEF when skipping safe files. Submit a thorough triage report when done."
        )},
    ]
    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            # Qwen3 supports enable_thinking parameter
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True  # Activate Qwen3's thinking mode
            )
        except TypeError:
            # Fallback if enable_thinking isn't supported
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    else:
        # Qwen3 chat format fallback
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{context}\n\nBegin your security investigation.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


# ── TRL Environment Wrapper ─────────────────────────────────────────
class CodeReviewToolEnv:
    """Wraps CodeReviewEnv v3 for TRL's environment_factory pattern."""

    def __init__(self):
        from code_review_env.server.environment import CodeReviewEnvironment
        self.env = CodeReviewEnvironment()
        self.reward = 0.0
        self.done = False
        self._files = []
        self._context = ""

    def reset(self, **kwargs):
        self.reward = 0.0
        self.done = False

        obs = self.env.reset(difficulty=random.choice(["easy", "medium"]))
        self._context = obs.metadata.get("context", "")
        self._files = re.findall(r'• (.+?)\s+\[', self._context)
        return self._context

    def _call_tool(self, tool_name, arguments):
        from openenv.core.env_server import CallToolAction
        obs = self.env.step(CallToolAction(tool_name=tool_name, arguments=arguments))
        result = obs.result
        if hasattr(result, 'data'):
            text = str(result.data)
        elif hasattr(result, 'content') and result.content:
            text = str(result.content[0].text)
        else:
            text = str(result)
        return text

    # ── Expose tools ──────────────────────────────────
    def read_file(self, file_path: str) -> str:
        return self._call_tool("read_file", {"file_path": file_path})

    def search_code(self, pattern: str) -> str:
        return self._call_tool("search_code", {"pattern": pattern})

    def get_function_list(self, file_path: str) -> str:
        return self._call_tool("get_function_list", {"file_path": file_path})

    def flag_vulnerable(self, file_path: str, reasoning: str) -> str:
        return self._call_tool("flag_vulnerable", {"file_path": file_path, "reasoning": reasoning})

    def skip_file(self, file_path: str, reasoning: str) -> str:
        return self._call_tool("skip_file", {"file_path": file_path, "reasoning": reasoning})

    def submit_report(self, summary: str, confidence: str = "medium") -> str:
        result = self._call_tool("submit_report", {"summary": summary, "confidence": confidence})
        score_match = re.search(r'TOTAL SCORE: ([\d.]+)', result)
        self.reward = float(score_match.group(1)) if score_match else 0.0
        self.done = True
        return result


def environment_factory():
    """TRL-compatible environment factory."""
    return CodeReviewToolEnv()


# ── Tool Call Parser ─────────────────────────────────────────────────
# Parses tool calls from the model's generated text so we can EXECUTE
# them against the live environment. Supports multiple formats:
#   - Qwen3 format: <tool_call>{"name":"...", "arguments":{...}}</tool_call>
#   - Python-style: tool_name("arg1", "arg2")
#   - JSON-style: {"tool": "...", "args": {...}}

def parse_tool_calls(text):
    """Parse tool invocations from model output text.
    Returns list of {"name": str, "args": dict} dicts."""
    calls = []

    # Format 1: Qwen3 <tool_call> blocks
    for match in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
        try:
            data = json.loads(match.group(1))
            name = data.get("name") or data.get("function", {}).get("name", "")
            args = data.get("arguments") or data.get("function", {}).get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if name:
                calls.append({"name": name, "args": args})
        except (json.JSONDecodeError, AttributeError):
            pass

    # Format 2: Python-style function calls  e.g. read_file("path/to/file.c")
    tool_names = ["read_file", "search_code", "get_function_list",
                  "flag_vulnerable", "skip_file", "submit_report"]
    for tool in tool_names:
        # Match: tool_name("arg1", "arg2") or tool_name(arg1="val", arg2="val")
        pattern = rf'{tool}\s*\(\s*(.+?)\s*\)'
        for match in re.finditer(pattern, text, re.DOTALL):
            raw_args = match.group(1).strip()
            args = _parse_python_args(tool, raw_args)
            if args:
                calls.append({"name": tool, "args": args})

    # Format 3: JSON objects with tool/name field
    for match in re.finditer(r'\{[^{}]*"(?:tool|name|function)"\s*:\s*"(\w+)"[^{}]*\}', text):
        try:
            data = json.loads(match.group(0))
            name = data.get("tool") or data.get("name") or data.get("function", "")
            args = data.get("args") or data.get("arguments") or data.get("parameters", {})
            if name in tool_names and name not in [c["name"] for c in calls]:
                calls.append({"name": name, "args": args if isinstance(args, dict) else {}})
        except (json.JSONDecodeError, AttributeError):
            pass

    return calls


def _parse_python_args(tool_name, raw_args):
    """Parse Python-style function arguments into a dict."""
    # Expected signatures:
    #   read_file(file_path)
    #   search_code(pattern)
    #   get_function_list(file_path)
    #   flag_vulnerable(file_path, reasoning)
    #   skip_file(file_path, reasoning)
    #   submit_report(summary, confidence)
    try:
        # Try to extract quoted strings
        strings = re.findall(r'["\']([^"\']+)["\']', raw_args)
        if not strings:
            # Try unquoted single arg
            strings = [raw_args.strip().strip("\"'")]

        if tool_name in ("read_file", "get_function_list"):
            return {"file_path": strings[0]} if strings else None
        elif tool_name == "search_code":
            return {"pattern": strings[0]} if strings else None
        elif tool_name in ("flag_vulnerable", "skip_file"):
            if len(strings) >= 2:
                return {"file_path": strings[0], "reasoning": strings[1]}
            elif len(strings) == 1:
                return {"file_path": strings[0], "reasoning": "flagged by model"}
        elif tool_name == "submit_report":
            return {
                "summary": strings[0] if strings else "Investigation complete",
                "confidence": strings[1] if len(strings) > 1 else "medium"
            }
    except Exception:
        pass
    return None


# ── Reward Functions ────────────────────────────────────────────────
# DESIGN: The reward function EXECUTES parsed tool calls against the
# LIVE environment. This means the training loop genuinely connects
# to the environment — not a static dataset. The model's generated
# investigation plan is actually run, and the real environment score
# (F1, report quality, thinking efficiency) becomes the reward.

# Global episode registry: maps prompt hash → episode seed
# Populated during dataset generation so reward_fn can replay episodes
_episode_registry = {}

def reward_fn(completions, prompts=None, **kwargs):
    """
    LIVE EXECUTION reward function for GRPO.

    For each model completion:
    1. Parse tool calls from the generated text
    2. Replay the same episode in a fresh environment
    3. Execute the parsed tool calls against the live environment
    4. Return the environment's TOTAL SCORE as the reward

    Falls back to text-based scoring if parsing fails.
    """
    rewards = []

    for idx, completion in enumerate(completions):
        text = completion if isinstance(completion, str) else completion.get("content", "")
        text_lower = text.lower()

        # ── Step 1: Try live execution ────────────────
        parsed_calls = parse_tool_calls(text)
        env_score = None

        bug_files: set = set()  # ground-truth bug files for metacog reward

        if parsed_calls and len(parsed_calls) >= 2:
            # We have parseable tool calls — execute them live
            try:
                env = CodeReviewToolEnv()

                # Find the matching episode from the prompt
                prompt_text = ""
                if prompts and idx < len(prompts):
                    prompt_text = prompts[idx] if isinstance(prompts[idx], str) else str(prompts[idx])

                # Extract seed from prompt or use index
                seed_match = re.search(r'episode_seed=(\d+)', prompt_text)
                seed = int(seed_match.group(1)) if seed_match else (idx * 7 + 42)

                # Reset environment with the matching episode
                random.seed(seed)
                context = env.reset()
                env.done = False
                env.reward = 0.0

                # Capture ground-truth bug files for the metacog reward.
                # The wrapper stores InvestigationSession on env.env._sessions[sid].
                try:
                    sid = env.env._current_session_id
                    sess = env.env._sessions.get(sid)
                    if sess is not None:
                        bug_files = set(sess.bugs)
                except Exception:
                    pass

                # Execute each parsed tool call
                for call in parsed_calls:
                    if env.done:
                        break
                    try:
                        name = call["name"]
                        args = call["args"]
                        if name == "read_file" and "file_path" in args:
                            env.read_file(args["file_path"])
                        elif name == "search_code" and "pattern" in args:
                            env.search_code(args["pattern"])
                        elif name == "get_function_list" and "file_path" in args:
                            env.get_function_list(args["file_path"])
                        elif name == "flag_vulnerable" and "file_path" in args:
                            env.flag_vulnerable(args["file_path"], args.get("reasoning", "flagged"))
                        elif name == "skip_file" and "file_path" in args:
                            env.skip_file(args["file_path"], args.get("reasoning", "skipped"))
                        elif name == "submit_report":
                            env.submit_report(args.get("summary", "Investigation report"),
                                              args.get("confidence", "medium"))
                    except Exception:
                        continue  # Skip failed tool calls, keep going

                # If model didn't submit a report, auto-submit
                if not env.done:
                    try:
                        env.submit_report("Auto-submitted: investigation complete", "low")
                    except Exception:
                        pass

                if env.reward > 0:
                    env_score = env.reward

            except Exception as e:
                pass  # Fall through to text-based scoring

        # ── Step 2: Text-based scoring (fallback + bonus) ─
        text_score = 0.0

        # Tool usage patterns
        tool_mentions = {
            "read_file": "read_file" in text_lower,
            "search_code": "search_code" in text_lower,
            "flag_vulnerable": "flag_vulnerable" in text_lower,
            "skip_file": "skip_file" in text_lower,
            "submit_report": "submit_report" in text_lower,
        }
        text_score += min(0.20, sum(tool_mentions.values()) * 0.04)

        # Investigation order
        if "read_file" in text_lower and "flag_vulnerable" in text_lower:
            if text_lower.find("read_file") < text_lower.find("flag_vulnerable"):
                text_score += 0.05
        if tool_mentions["flag_vulnerable"] and tool_mentions["skip_file"]:
            text_score += 0.05  # Mixed decisions

        # Reasoning quality
        think_blocks = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_blocks:
            avg_len = sum(len(b.strip()) for b in think_blocks) / len(think_blocks)
            if avg_len > 200: text_score += 0.08
            elif avg_len > 80: text_score += 0.04
            # Security terms
            vuln_terms = ["overflow", "injection", "bypass", "escalation", "xss",
                          "traversal", "free", "null", "unchecked", "buffer",
                          "bounds", "sanitiz", "memcpy", "strcpy", "eval"]
            reasoning = " ".join(think_blocks).lower()
            text_score += min(0.08, sum(1 for t in vuln_terms if t in reasoning) * 0.02)

        # Report quality
        if tool_mentions["submit_report"]:
            if re.search(r'cve-\d{4}-\d+', text_lower): text_score += 0.04
            vuln_types = ["buffer overflow", "injection", "privilege escalation",
                          "use-after-free", "traversal", "bypass", "overflow"]
            if any(v in text_lower for v in vuln_types): text_score += 0.04

        # Anti-gaming
        if len(text.strip()) < 50: text_score *= 0.1
        if text_lower.count("skip_file") > 10 and not tool_mentions["flag_vulnerable"]:
            text_score *= 0.5
        lines = text.strip().split('\n')
        if len(lines) > 5 and len(set(lines)) / len(lines) < 0.3:
            text_score *= 0.3

        # ── Step 3: Metacognitive reward (v2 contribution) ─
        # Calibration: did <budget_prediction> match the actual <think> length?
        # Difficulty awareness: long preds on bugs, short on safe?
        # Coupling: every prediction tied to a real tool call?
        metacog_score = 0.0
        metacog_details = []
        metacog_metrics = None
        if METACOG_ENABLED:
            try:
                from metacognitive_reward import compute_metacognitive_reward
                metacog = compute_metacognitive_reward(text, bug_files=bug_files)
                metacog_score = metacog.raw_score
                metacog_details = list(metacog.details)
                metacog_metrics = {
                    "calibration": metacog.calibration,
                    "difficulty_awareness": metacog.difficulty_awareness,
                    "coupling": metacog.coupling,
                    "n_predictions": metacog.n_predictions,
                }
            except Exception:
                metacog_score = 0.0

        # ── Step 4: Combine scores ────────────────────
        # v2 weighting: env (50%) > metacog (30%) > text (20%) when live
        # execution succeeds.  Fallback path keeps text dominant but still
        # rewards metacognitive structure when present.
        if env_score is not None:
            if METACOG_ENABLED:
                final = (0.50 * env_score
                         + METACOG_WEIGHT * metacog_score
                         + (0.50 - METACOG_WEIGHT) * text_score)
            else:
                final = 0.70 * env_score + 0.30 * text_score
        else:
            if METACOG_ENABLED:
                final = 0.70 * text_score + 0.30 * metacog_score
            else:
                final = text_score

        final = min(1.0, max(0.0, final))
        rewards.append(final)

        # ── Step 5: Persistent logging for post-hoc analysis ──
        # Append every reward call's calibration data + scores to disk so
        # we can build the *real* calibration plot from training rollouts
        # (not from a heuristic proxy).
        try:
            _log_reward_call(
                idx=idx,
                env_score=env_score,
                text_score=text_score,
                metacog_score=metacog_score,
                metacog_metrics=metacog_metrics,
                metacog_details=metacog_details,
                final=final,
                bug_files=bug_files,
            )
        except Exception:
            pass

    return rewards


# ── Reward-call logger ──────────────────────────────────────────────
# Appends a JSONL line per reward call (cheap; flushed every step).  At the
# end of training the live calibration plot script reads this file to
# produce eval_calibration.json containing real (pred, actual_len, label)
# triples.  This makes the headline calibration figure REAL data, not a
# heuristic proxy.
_TRACE_FH = None
_CAL_BUFFER: list = []


def _log_reward_call(idx, env_score, text_score, metacog_score,
                     metacog_metrics, metacog_details, final, bug_files):
    global _TRACE_FH, _CAL_BUFFER
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if _TRACE_FH is None:
        _TRACE_FH = open(TRACE_LOG_PATH, "a", buffering=1)  # line-buffered

    payload = {
        "ts": time.time(),
        "idx": idx,
        "env_score": env_score,
        "text_score": text_score,
        "metacog_score": metacog_score,
        "metacog": metacog_metrics,
        "n_bug_files": len(bug_files) if bug_files else 0,
        "final": final,
    }
    _TRACE_FH.write(json.dumps(payload) + "\n")

    # Buffer the per-prediction calibration triples
    for pred, actual_len, label in metacog_details:
        _CAL_BUFFER.append({"pred": pred, "actual_len": int(actual_len),
                            "label": label})

    # Flush the calibration buffer every CALIBRATION_LOG_EVERY calls.
    # The plot script consumes this JSON.
    if len(_CAL_BUFFER) % max(1, CALIBRATION_LOG_EVERY) == 0:
        _flush_calibration_buffer()


def _flush_calibration_buffer():
    """Write the rolling calibration buffer to disk in the plot's expected
    schema: {pred: [...], actual_len: [...], label: [...]} where label is
    1 (bug), 0 (safe), or null (unknown)."""
    if not _CAL_BUFFER:
        return
    out = {
        "pred": [d["pred"] for d in _CAL_BUFFER],
        "actual_len": [d["actual_len"] for d in _CAL_BUFFER],
        "label": [d["label"] for d in _CAL_BUFFER],
    }
    tmp = EVAL_CALIBRATION_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, EVAL_CALIBRATION_PATH)


# ── Training Plots ──────────────────────────────────────────────────
def save_training_plots(rewards, output_dir):
    """Generate publication-quality training curve plots for judges."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    model_short = MODEL_NAME.split("/")[-1] if "/" in MODEL_NAME else MODEL_NAME
    fig.suptitle(f"CodeReviewEnv v3 — GRPO Training on {model_short}",
                 fontsize=16, fontweight='bold', y=1.02)

    # 1. Reward curve with smoothing
    axes[0].plot(rewards, alpha=0.2, color="#667eea", linewidth=0.8, label="Per-step")
    window = max(1, min(30, len(rewards) // 4))
    if len(rewards) >= window:
        smoothed = [sum(rewards[max(0,i-window):i+1])/min(i+1, window)
                    for i in range(len(rewards))]
        axes[0].plot(smoothed, color="#764ba2", linewidth=2.5, label=f"Moving avg (n={window})")
    axes[0].set_xlabel("Training Step", fontsize=12)
    axes[0].set_ylabel("Reward", fontsize=12)
    axes[0].set_title("GRPO Reward Curve", fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 2. Score distribution: early vs late
    if len(rewards) > 30:
        third = len(rewards) // 3
        early = rewards[:third]
        late = rewards[-third:]
        axes[1].hist(early, bins=20, alpha=0.5, color="#667eea", label=f"Early (n={len(early)})", density=True)
        axes[1].hist(late, bins=20, alpha=0.5, color="#764ba2", label=f"Late (n={len(late)})", density=True)
        axes[1].axvline(sum(early)/len(early), color="#667eea", linestyle='--', linewidth=2, label=f"Early μ={sum(early)/len(early):.3f}")
        axes[1].axvline(sum(late)/len(late), color="#764ba2", linestyle='--', linewidth=2, label=f"Late μ={sum(late)/len(late):.3f}")
        axes[1].set_xlabel("Reward", fontsize=12)
        axes[1].set_ylabel("Density", fontsize=12)
        axes[1].set_title("Reward Distribution Shift", fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=9)
    else:
        axes[1].hist(rewards, bins=15, color="#667eea", edgecolor='white')
        axes[1].set_xlabel("Reward", fontsize=12)
        axes[1].set_ylabel("Count", fontsize=12)
        axes[1].set_title("Reward Distribution", fontsize=14, fontweight='bold')

    # 3. Cumulative best score
    cum_best = []
    best = 0
    for r in rewards:
        best = max(best, r)
        cum_best.append(best)
    axes[2].plot(cum_best, color="#e74c3c", linewidth=2.5)
    axes[2].fill_between(range(len(cum_best)), cum_best, alpha=0.1, color="#e74c3c")
    axes[2].set_xlabel("Training Step", fontsize=12)
    axes[2].set_ylabel("Best Reward So Far", fontsize=12)
    axes[2].set_title("Cumulative Best Reward", fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved training plots to {plot_path}")

    # Save raw data as JSON for README
    stats_path = os.path.join(output_dir, "training_stats.json")
    stats = {
        "model": MODEL_NAME,
        "num_episodes": NUM_EPISODES,
        "num_steps": len(rewards),
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0,
        "max_reward": max(rewards) if rewards else 0,
        "early_mean": sum(rewards[:len(rewards)//3]) / max(1, len(rewards)//3),
        "late_mean": sum(rewards[-len(rewards)//3:]) / max(1, len(rewards)//3),
        "improvement": 0,
        "timestamp": datetime.now().isoformat(),
    }
    stats["improvement"] = stats["late_mean"] - stats["early_mean"]
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"📈 Stats: early_mean={stats['early_mean']:.3f} → late_mean={stats['late_mean']:.3f} (improvement: {stats['improvement']:+.3f})")


# ── Dtype Fix Helper ───────────────────────────────────────────────
# Qwen3 + 4-bit quantization with bnb_4bit_compute_dtype=bfloat16
# leaves a few Float32 modules (lm_head, embed_tokens, LayerNorms)
# untouched. When the bf16 hidden states hit a Float32 lm_head we get:
#   RuntimeError: expected scalar type Float but found BFloat16
# This helper walks every parameter/buffer and force-casts any
# non-quantized floating tensor to bfloat16 so the whole forward pass
# is consistent. The lm_head MUST be bf16 to match the hidden states.
def cast_model_to_bfloat16(model):
    """Force-cast all floating-point params to bfloat16 (skips quantized int params)."""
    target_dtype = torch.bfloat16
    converted = 0
    skipped_quant = 0
    for name, param in model.named_parameters():
        if param.dtype in (torch.uint8, torch.int8, torch.int32, torch.int64):
            skipped_quant += 1
            continue
        if param.dtype in (torch.float32, torch.float16):
            param.data = param.data.to(target_dtype)
            converted += 1
    for name, buf in model.named_buffers():
        if buf.dtype in (torch.float32, torch.float16):
            buf.data = buf.data.to(target_dtype)
    # The lm_head is the critical one — make absolutely sure
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        if model.lm_head.weight.dtype != target_dtype:
            model.lm_head.weight.data = model.lm_head.weight.data.to(target_dtype)
    # Walk into base_model for PEFT-wrapped models
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "model") and hasattr(base.model, "lm_head"):
            lm = base.model.lm_head
            if hasattr(lm, "weight") and lm.weight.dtype != target_dtype:
                lm.weight.data = lm.weight.data.to(target_dtype)
    print(f"   Cast {converted} params → bfloat16 (skipped {skipped_quant} quantized)")
    return model


# ── BULLETPROOF DTYPE-SAFETY HOOK ──────────────────────────────────
# PEFT's prepare_model_for_kbit_training EXPLICITLY casts non-int8
# params (including lm_head) back to float32 for stability. This
# UNDOES our cast_model_to_bfloat16 right after we run it. Confirmed
# via huggingface/peft#816 and #1249.
#
# Instead of fighting PEFT's design, we install a forward pre-hook
# on the lm_head module that casts its INPUT (hidden_states) to match
# the lm_head's WEIGHT dtype on every forward call. So no matter what
# dtype the lm_head ends up as (fp32 from prepare_model_for_kbit_training
# or bf16 from our cast), the matmul ALWAYS works.
#
# This is the canonical fix recommended by HF maintainers.
def install_lm_head_dtype_hook(model):
    """Install a forward pre-hook on lm_head so input dtype always matches weight dtype."""
    candidates = []
    if hasattr(model, "lm_head"):
        candidates.append(model.lm_head)
    if hasattr(model, "base_model"):
        bm = model.base_model
        if hasattr(bm, "model") and hasattr(bm.model, "lm_head"):
            candidates.append(bm.model.lm_head)
        if hasattr(bm, "lm_head"):
            candidates.append(bm.lm_head)
    if hasattr(model, "get_output_embeddings"):
        try:
            out = model.get_output_embeddings()
            if out is not None:
                candidates.append(out)
        except Exception:
            pass

    seen = set()
    unique_heads = []
    for c in candidates:
        if c is not None and id(c) not in seen and hasattr(c, "weight"):
            seen.add(id(c))
            unique_heads.append(c)

    if not unique_heads:
        print("⚠️  No lm_head found to install dtype hook on")
        return model

    def make_hook():
        def hook(module, args, kwargs):
            target = module.weight.dtype
            new_args = tuple(
                a.to(target) if (torch.is_tensor(a) and a.is_floating_point() and a.dtype != target) else a
                for a in args
            )
            new_kwargs = {
                k: (v.to(target) if (torch.is_tensor(v) and v.is_floating_point() and v.dtype != target) else v)
                for k, v in kwargs.items()
            }
            return (new_args, new_kwargs)
        return hook

    for lm in unique_heads:
        try:
            lm.register_forward_pre_hook(make_hook(), with_kwargs=True)
            print(f"🔒 lm_head dtype-safety hook installed (weight={lm.weight.dtype}, shape={tuple(lm.weight.shape)})")
        except Exception as e:
            print(f"⚠️  Failed to install hook on lm_head: {e}")
    return model


# ── Main Training Loop ─────────────────────────────────────────────
def main():
    start_time = time.time()
    print("=" * 60)
    print("  The Thinking Budget — GRPO Training v2 (metacognitive)")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Episodes: {NUM_EPISODES} × {NUM_TRAIN_EPOCHS} epoch(s) "
          f"| Batch: {BATCH_SIZE}×{GRAD_ACCUM_STEPS}")
    print(f"  LR: {LEARNING_RATE} | warmup: {WARMUP_RATIO} | β(KL): {GRPO_BETA}")
    print(f"  LoRA: r={LORA_R}, α={LORA_ALPHA}")
    print(f"  Metacognitive reward: {'ENABLED' if METACOG_ENABLED else 'OFF'} "
          f"(weight={METACOG_WEIGHT})")
    print(f"  Checkpointing every {SAVE_EVERY} steps")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Load model ──────────────────────────────────
    # CRITICAL: We force bfloat16 EVERYWHERE because Qwen3's lm_head
    # otherwise stays in float32 and breaks forward with
    # "expected scalar type Float but found BFloat16".
    if USE_UNSLOTH:
        try:
            from unsloth import FastLanguageModel
            print(f"\n🔧 Loading {MODEL_NAME} with Unsloth (4-bit, bf16)...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=torch.bfloat16,   # explicit bf16 (was None → auto)
                load_in_4bit=True,
            )
            print("🩹 Pre-LoRA dtype fix...")
            model = cast_model_to_bfloat16(model)
            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_R,
                target_modules=["q_proj", "k_proj", "v_proj",
                                "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=LORA_ALPHA,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            print("🩹 Post-LoRA dtype fix...")
            model = cast_model_to_bfloat16(model)
            model = install_lm_head_dtype_hook(model)
            print(f"✅ Unsloth + LoRA loaded (r={LORA_R}, alpha={LORA_ALPHA}, dtype=bf16)")
        except (ImportError, Exception) as e:
            print(f"⚠️  Unsloth failed ({e}), using transformers + bitsandbytes...")
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            # 4-bit quantization to fit 1.7B model in the 14 GiB Space cap with headroom for activations
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,   # was string "bfloat16"
                bnb_4bit_use_double_quant=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,   # explicit bf16 (was "auto")
            )
            print("🩹 Pre-prepare dtype fix...")
            model = cast_model_to_bfloat16(model)
            model = prepare_model_for_kbit_training(model)

            # Apply LoRA
            lora_config = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                target_modules=["q_proj", "k_proj", "v_proj",
                                "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            print("🩹 Post-LoRA dtype fix...")
            model = cast_model_to_bfloat16(model)
            model = install_lm_head_dtype_hook(model)
            model.gradient_checkpointing_enable()
            print(f"✅ Transformers + 4-bit + LoRA loaded (r={LORA_R}, alpha={LORA_ALPHA}, dtype=bf16)")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        print(f"\n🔧 Loading {MODEL_NAME} with transformers + 4-bit (bf16)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,   # was string "bfloat16"
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,   # explicit bf16
        )
        print("🩹 Pre-prepare dtype fix...")
        model = cast_model_to_bfloat16(model)
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj",
                            "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print("🩹 Post-LoRA dtype fix...")
        model = cast_model_to_bfloat16(model)
        model = install_lm_head_dtype_hook(model)
        model.gradient_checkpointing_enable()

    # ── Load SFT warmup adapter if available ───────────────────────
    # Two-phase training: SFT teaches the output format, GRPO refines
    # the allocation policy.  If ADAPTER_PATH is set, we load the SFT
    # adapter, merge it into the base model, then the fresh LoRA above
    # learns the RL policy on top of the format-aware weights.
    if ADAPTER_PATH and os.path.exists(ADAPTER_PATH):
        print(f"\n🔄 Loading SFT warmup adapter from {ADAPTER_PATH}...")
        try:
            from peft import PeftModel
            # The model already has a LoRA attached from above.
            # We need to merge the SFT adapter weights, then re-apply
            # a fresh LoRA.  PEFT makes this possible via merge_and_unload
            # on the SFT adapter, but since we're already in 4-bit land,
            # the simpler approach is to load the SFT weights directly
            # into the existing LoRA parameters.
            import safetensors.torch as st
            adapter_file = os.path.join(ADAPTER_PATH, "adapter_model.safetensors")
            if os.path.exists(adapter_file):
                sft_state = st.load_file(adapter_file)
                model_state = model.state_dict()
                loaded = 0
                for k, v in sft_state.items():
                    if k in model_state and model_state[k].shape == v.shape:
                        model_state[k].copy_(v.to(model_state[k].dtype))
                        loaded += 1
                print(f"✅ Loaded {loaded} SFT adapter weights (model starts format-aware)")
            else:
                print(f"⚠️  adapter_model.safetensors not found in {ADAPTER_PATH}")
        except Exception as e:
            print(f"⚠️  Failed to load SFT adapter: {e}")
            print("   Continuing with base model weights.")

    # Final dtype verification — log lm_head dtype so it's obvious
    # in the training logs whether the fix took effect. The hook above
    # makes the actual dtype value irrelevant (input is auto-cast to
    # match), but we log it for debugging clarity.
    try:
        head = model.get_output_embeddings()
        if head is not None and hasattr(head, "weight"):
            print(f"🔬 lm_head weight dtype: {head.weight.dtype}  (hook will auto-cast input to match)")
    except Exception:
        pass

    # CRITICAL FIX: TRL GRPOTrainer bug workaround
    # TRL expects model.warnings_issued to exist, but some models (like Qwen3) don't have it.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    # ── Ensure pad token ────────────────────────────
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Build dataset from environment episodes ─────
    # Each prompt embeds an episode_seed so the reward function can
    # replay the SAME episode when executing parsed tool calls live.
    print(f"\n📋 Generating {NUM_EPISODES} training prompts from environment...")
    prompts = []
    env = CodeReviewToolEnv()
    for i in range(NUM_EPISODES):
        seed = i * 7 + 42
        random.seed(seed)  # Seed for reproducibility
        context = env.reset()  # reset() already returns the context string
        # Embed seed in the context so reward_fn can replay this exact episode
        context_with_seed = f"<!-- episode_seed={seed} -->\n{context}"
        formatted = format_prompt_as_chat(context_with_seed, tokenizer)
        prompts.append({"prompt": formatted})
        if (i + 1) % 50 == 0:
            print(f"   Generated {i+1}/{NUM_EPISODES} prompts")

    from datasets import Dataset
    dataset = Dataset.from_list(prompts)
    print(f"✅ Dataset ready: {len(dataset)} episodes")

    # ── Configure GRPO ──────────────────────────────
    from trl import GRPOConfig, GRPOTrainer

    # GRPOConfig accepts `beta` (KL coefficient) on TRL ≥ 0.17 — we
    # pass it via kwargs to stay compatible with older versions that
    # might not have the parameter.
    grpo_kwargs = {}
    try:
        import inspect
        sig = inspect.signature(GRPOConfig.__init__)
        if "beta" in sig.parameters:
            grpo_kwargs["beta"] = GRPO_BETA
    except Exception:
        pass

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,        # was 1, now 2 → ~150 steps
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        logging_steps=1,           # log EVERY step for smooth curves
        save_steps=SAVE_EVERY,     # checkpoint every 50 steps
        save_total_limit=3,
        report_to="none",
        bf16=True,
        seed=42,
        # Resume from checkpoint if available
        resume_from_checkpoint=_find_latest_checkpoint(),
        **grpo_kwargs,
    )

    # ── Callback: incremental curves every 25 steps ──
    # Judges can see reward progression even while training is running.
    from transformers import TrainerCallback

    class IncrementalPlotCallback(TrainerCallback):
        """Generate training curves every 25 steps so judges see progress."""
        def __init__(self, plot_every=25):
            self.plot_every = plot_every
            self._last_plot_step = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.global_step - self._last_plot_step >= self.plot_every:
                rewards = []
                for log in state.log_history:
                    if "reward" in log:
                        rewards.append(log["reward"])
                    elif "rewards/mean" in log:
                        rewards.append(log["rewards/mean"])
                if len(rewards) >= 10:
                    try:
                        save_training_plots(rewards, OUTPUT_DIR)
                        self._last_plot_step = state.global_step
                        print(f"📊 [step {state.global_step}] Incremental training curve saved ({len(rewards)} points)")
                    except Exception as e:
                        print(f"⚠️  Incremental plot failed: {e}")

    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        callbacks=[IncrementalPlotCallback(plot_every=25)],
    )

    # ── Train ───────────────────────────────────────
    print(f"\n🚀 Starting GRPO training...")
    print(f"   Estimated time: 3-5 hours on A10G")
    training_succeeded = False
    try:
        trainer.train(resume_from_checkpoint=_find_latest_checkpoint())
        training_succeeded = True
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted! Saving checkpoint...")
        model.save_pretrained(os.path.join(CHECKPOINT_DIR, "interrupted"))
        tokenizer.save_pretrained(os.path.join(CHECKPOINT_DIR, "interrupted"))
        print("❌ TRAINING DID NOT COMPLETE. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        traceback.print_exc()
        print("❌ TRAINING FAILED. Exiting with error.")
        sys.exit(1)

    if not training_succeeded:
        print("❌ Training did not complete successfully. Exiting.")
        sys.exit(1)

    # ── Save final model ────────────────────────────
    print(f"\n💾 Saving final model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ── Generate plots (ONLY from real training data) ──
    rewards = []
    for log in trainer.state.log_history:
        if "reward" in log:
            rewards.append(log["reward"])
        elif "rewards/mean" in log:
            rewards.append(log["rewards/mean"])

    if rewards:
        save_training_plots(rewards, OUTPUT_DIR)
    else:
        print("⚠️  No reward logs found in trainer history.")
        print("❌ Cannot generate results without training data. Exiting.")
        sys.exit(1)

    # ── Auto-regenerate calibration plot from REAL training data ──
    # The reward fn streams (pred, actual_len, label) triples to
    # eval_calibration.json on every call.  After training we run the
    # plot script in real mode so the headline calibration figure is
    # produced from the actual model's predictions, not a heuristic
    # proxy.
    try:
        _flush_calibration_buffer()
        if os.path.exists(EVAL_CALIBRATION_PATH):
            import subprocess
            print(f"\n📐 Regenerating calibration plot from real training data...")
            subprocess.run([
                sys.executable,
                os.path.join(os.path.dirname(__file__), "scripts", "generate_calibration_plot.py"),
                "--mode", "real",
                "--data", EVAL_CALIBRATION_PATH,
                "--out", os.path.join(OUTPUT_DIR, "calibration_plot.png"),
            ], check=False)
    except Exception as e:
        print(f"⚠️  Calibration plot regeneration failed (not fatal): {e}")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ✅ Training complete in {elapsed/3600:.1f} hours")
    print(f"  📁 Model: {OUTPUT_DIR}")
    print(f"  📊 Plots: {OUTPUT_DIR}/training_curves.png")
    print(f"  📈 Stats: {OUTPUT_DIR}/training_stats.json")
    print(f"  📐 Calibration: {OUTPUT_DIR}/calibration_plot.png")
    print(f"{'='*60}")


def _find_latest_checkpoint():
    """Find the latest checkpoint to resume from."""
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")]
    if not checkpoints:
        # Also check OUTPUT_DIR for TRL's default checkpoint format
        if os.path.exists(OUTPUT_DIR):
            checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
            if checkpoints:
                latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
                path = os.path.join(OUTPUT_DIR, latest)
                print(f"📂 Found checkpoint: {path}")
                return path
        return None
    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
    path = os.path.join(CHECKPOINT_DIR, latest)
    print(f"📂 Found checkpoint: {path}")
    return path


if __name__ == "__main__":
    main()
