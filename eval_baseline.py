"""
eval_baseline.py — Before/After Comparison
============================================
Demonstrates that GRPO training actually improved the agent's behavior.

Compares:
  1. Untrained baseline    — Qwen3-8B with the system prompt only
  2. Trained model         — Qwen3-8B + LoRA adapter from grpo_output/

On the SAME 5 episodes (deterministic seeds), running the same investigation
loop and reporting per-episode and mean scores.

This is the script judges run to verify the training worked. It produces:
  - eval_baseline_vs_trained.json (raw numbers)
  - eval_baseline_vs_trained.png  (side-by-side bar chart)

Usage:
    python eval_baseline.py
    # or from Colab:  %run eval_baseline.py
"""
import os
import sys
import re
import json
import random
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Config ─────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-8B"
ADAPTER_DIR = "./grpo_output"
EVAL_SEEDS = [42, 137, 256, 512, 1024]    # 5 deterministic episodes
DIFFICULTY = "easy"
MAX_NEW_TOKENS = 768

OUT_JSON = os.path.join(ADAPTER_DIR, "eval_baseline_vs_trained.json")
OUT_PLOT = os.path.join(ADAPTER_DIR, "eval_baseline_vs_trained.png")
os.makedirs(ADAPTER_DIR, exist_ok=True)


# ── System prompt (must match training) ────────────────────────────
SYSTEM_PROMPT = """You are an expert security code investigator. You have 6 tools:
read_file, search_code, get_function_list, flag_vulnerable, skip_file, submit_report.

Investigate the patch, flag vulnerable files with detailed reasoning, skip safe files
briefly, and submit a triage report. Use tool calls in this format:
<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>"""


# ── Tool-call execution against live env ───────────────────────────
def parse_tool_calls(text):
    calls = []
    for m in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
        try:
            data = json.loads(m.group(1))
            name = data.get("name", "")
            args = data.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if name:
                calls.append({"name": name, "args": args})
        except (json.JSONDecodeError, AttributeError):
            pass
    return calls


def execute_episode(env, calls):
    """Execute parsed tool calls and return the final TOTAL SCORE."""
    from openenv.core.env_server import CallToolAction
    final_score = 0.0
    last_text = ""
    for call in calls[:25]:    # cap to prevent runaway
        try:
            obs = env.step(CallToolAction(tool_name=call["name"], arguments=call["args"]))
            text = str(obs.result.data if hasattr(obs.result, 'data') else obs.result)
            last_text = text
            if "TOTAL SCORE:" in text:
                m = re.search(r'TOTAL SCORE: ([\d.]+)', text)
                if m:
                    final_score = float(m.group(1))
                    break
        except Exception:
            continue

    # Auto-submit if model never did
    if final_score == 0.0 and "INVESTIGATION COMPLETE" not in last_text:
        try:
            obs = env.step(CallToolAction(tool_name="submit_report",
                                          arguments={"summary": "Auto-submitted", "confidence": "low"}))
            text = str(obs.result.data if hasattr(obs.result, 'data') else obs.result)
            m = re.search(r'TOTAL SCORE: ([\d.]+)', text)
            if m:
                final_score = float(m.group(1))
        except Exception:
            pass
    return final_score


def build_prompt(context, tokenizer):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\nBegin investigation."},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def run_single_episode(model, tokenizer, seed):
    """Run ONE episode end-to-end and return its score."""
    from code_review_env.server.environment import CodeReviewEnvironment
    env = CodeReviewEnvironment()
    obs = env.reset(seed=seed, difficulty=DIFFICULTY)
    context = obs.metadata.get("context", "")

    prompt = build_prompt(context, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    calls = parse_tool_calls(generated)
    score = execute_episode(env, calls) if calls else 0.0
    return score, len(calls), len(generated)


# ── Load each model and evaluate ───────────────────────────────────
def evaluate(adapter_path=None, label="Baseline"):
    print(f"\n{'='*70}\n  Evaluating: {label}\n{'='*70}")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Install the same dtype-safety hook used in training
    def _hook(module, args, kwargs):
        target = module.weight.dtype
        new_args = tuple(a.to(target) if (torch.is_tensor(a) and a.is_floating_point() and a.dtype != target) else a for a in args)
        new_kwargs = {k: (v.to(target) if (torch.is_tensor(v) and v.is_floating_point() and v.dtype != target) else v) for k, v in kwargs.items()}
        return (new_args, new_kwargs)
    if hasattr(model, "lm_head"):
        model.lm_head.register_forward_pre_hook(_hook, with_kwargs=True)

    if adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"  Loaded LoRA adapter from {adapter_path}")
    else:
        print(f"  No adapter loaded — pure base model")

    model.eval()

    scores = []
    for seed in EVAL_SEEDS:
        score, n_calls, gen_len = run_single_episode(model, tokenizer, seed)
        scores.append(score)
        print(f"  seed={seed:5d}  score={score:.3f}  tool_calls={n_calls:2d}  gen_chars={gen_len}")

    mean = sum(scores) / len(scores)
    print(f"  MEAN: {mean:.3f}")

    del model
    torch.cuda.empty_cache()
    return scores, mean


# ── Plot ───────────────────────────────────────────────────────────
def plot_comparison(baseline_scores, trained_scores):
    fig, ax = plt.subplots(figsize=(11, 6))
    n = len(baseline_scores)
    x = list(range(n))
    width = 0.38

    ax.bar([i - width/2 for i in x], baseline_scores, width,
           label=f"Baseline (untrained)  mean={sum(baseline_scores)/n:.3f}",
           color="#94a3b8", edgecolor="#475569")
    ax.bar([i + width/2 for i in x], trained_scores, width,
           label=f"Trained (GRPO)         mean={sum(trained_scores)/n:.3f}",
           color="#7c3aed", edgecolor="#5b21b6")

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed={s}" for s in EVAL_SEEDS])
    ax.set_ylabel("Total Score (env reward)", fontsize=12)
    ax.set_title("CodeReviewEnv v3 — Baseline vs GRPO-Trained Qwen3-8B",
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(1.0, max(baseline_scores + trained_scores) * 1.2))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate improvement
    delta = sum(trained_scores)/n - sum(baseline_scores)/n
    ax.text(0.02, 0.97, f"Δ = {delta:+.3f}",
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='#fef3c7', edgecolor='#f59e0b'))

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison plot: {OUT_PLOT}")


# ── Main ───────────────────────────────────────────────────────────
def main():
    print(f"Evaluating on {len(EVAL_SEEDS)} episodes (seeds={EVAL_SEEDS})")

    baseline_scores, baseline_mean = evaluate(adapter_path=None, label="Baseline (untrained)")
    trained_scores, trained_mean = evaluate(adapter_path=ADAPTER_DIR, label="Trained (GRPO)")

    delta = trained_mean - baseline_mean
    pct = (delta / baseline_mean * 100) if baseline_mean > 0 else float('inf')

    print(f"\n{'='*70}\n  COMPARISON\n{'='*70}")
    print(f"  Baseline mean: {baseline_mean:.3f}")
    print(f"  Trained mean:  {trained_mean:.3f}")
    print(f"  Improvement:   {delta:+.3f}  ({pct:+.1f}%)")

    results = {
        "model": MODEL_NAME,
        "adapter": ADAPTER_DIR,
        "difficulty": DIFFICULTY,
        "seeds": EVAL_SEEDS,
        "baseline_scores": baseline_scores,
        "trained_scores": trained_scores,
        "baseline_mean": baseline_mean,
        "trained_mean": trained_mean,
        "improvement": delta,
        "improvement_pct": pct,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison data: {OUT_JSON}")

    plot_comparison(baseline_scores, trained_scores)


if __name__ == "__main__":
    main()
