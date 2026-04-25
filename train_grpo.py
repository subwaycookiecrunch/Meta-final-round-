"""
GRPO Training Script for CodeReviewEnv v3
==========================================
Trains Qwen3-8B to investigate CVE vulnerabilities using MCP tools.
The agent learns to read code, search patterns, and produce triage reports.

Qwen3-8B was chosen for:
  - Superior multi-step REASONING (critical for vulnerability analysis)
  - Built for agentic tool-calling workflows (our 6 MCP tools)
  - 128K native context window (reads more code per investigation)
  - Better at explaining WHY code is vulnerable (higher report scores)

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# ── Configuration ───────────────────────────────────────────────────
# Model: Qwen3-8B — latest-generation reasoning model with native
# support for agentic tool-calling and 128K context. Its multi-step
# reasoning outperforms code-specialized models on security analysis
# tasks where understanding WHY code is vulnerable matters more than
# just recognizing syntax patterns.
MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_DIR = "./grpo_output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Training hyperparameters (tuned for 8B on A10G 24GB VRAM)
NUM_EPISODES = 200           # 200 episodes is enough for clear curves
NUM_GENERATIONS = 2          # reduced from 4 to fit 8B in VRAM
MAX_COMPLETION_LENGTH = 1536 # completions for reasoning + reports
BATCH_SIZE = 1               # 8B needs small batch to fit in memory
GRAD_ACCUM_STEPS = 8         # effective batch = 1 * 8 = 8
LEARNING_RATE = 1e-6         # slightly lower LR for 8B stability
WARMUP_RATIO = 0.05          # gentle warmup prevents early instability
MAX_SEQ_LENGTH = 4096        # practical context for VRAM
LORA_R = 32                  # higher rank = more capacity for 8B
LORA_ALPHA = 64              # 2x rank is standard
SAVE_EVERY = 50              # checkpoint every 50 steps
USE_UNSLOTH = os.environ.get("USE_UNSLOTH", "true").lower() == "true"


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
                env.env.reset(seed=seed, difficulty="easy")
                env._context = env.env._get_observation().metadata.get("context", "")
                env._files = re.findall(r'• (.+?)\s+\[', env._context)
                env.done = False
                env.reward = 0.0

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

        # ── Step 3: Combine scores ────────────────────
        if env_score is not None:
            # Live execution succeeded — weight it heavily
            final = 0.70 * env_score + 0.30 * text_score
        else:
            # Fallback to text-only
            final = text_score

        rewards.append(min(1.0, max(0.0, final)))

    return rewards


# ── Training Plots ──────────────────────────────────────────────────
def save_training_plots(rewards, output_dir):
    """Generate publication-quality training curve plots for judges."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("CodeReviewEnv v3 — GRPO Training on Qwen3-8B", fontsize=16, fontweight='bold', y=1.02)

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


# ── Main Training Loop ─────────────────────────────────────────────
def main():
    start_time = time.time()
    print("=" * 60)
    print("  CodeReviewEnv v3 — GRPO Training")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Episodes: {NUM_EPISODES} | Batch: {BATCH_SIZE}×{GRAD_ACCUM_STEPS}")
    print(f"  LR: {LEARNING_RATE} | LoRA: r={LORA_R}")
    print(f"  Checkpointing every {SAVE_EVERY} steps")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Load model ──────────────────────────────────
    if USE_UNSLOTH:
        try:
            from unsloth import FastLanguageModel
            print(f"\n🔧 Loading {MODEL_NAME} with Unsloth (4-bit)...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
            )
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
            print(f"✅ Unsloth + LoRA loaded (r={LORA_R}, alpha={LORA_ALPHA})")
        except ImportError:
            print("⚠️  Unsloth not available, using transformers (slower)...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype="auto", device_map="auto"
            )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"\n🔧 Loading {MODEL_NAME} with transformers...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype="auto", device_map="auto"
        )

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Build dataset from environment episodes ─────
    # Each prompt embeds an episode_seed so the reward function can
    # replay the SAME episode when executing parsed tool calls live.
    print(f"\n📋 Generating {NUM_EPISODES} training prompts from environment...")
    prompts = []
    env = CodeReviewToolEnv()
    for i in range(NUM_EPISODES):
        seed = i * 7 + 42  # deterministic, reproducible seeds
        env.env.reset(seed=seed, difficulty=random.choice(["easy", "medium"]))
        # Get context from the environment observation
        obs = env.env._get_observation()
        context = obs.metadata.get("context", "")
        env._context = context
        env._files = re.findall(r'• (.+?)\s+\[', context)
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

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
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
    )

    trainer = GRPOTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_fn,
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

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ✅ Training complete in {elapsed/3600:.1f} hours")
    print(f"  📁 Model: {OUTPUT_DIR}")
    print(f"  📊 Plots: {OUTPUT_DIR}/training_curves.png")
    print(f"  📈 Stats: {OUTPUT_DIR}/training_stats.json")
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
