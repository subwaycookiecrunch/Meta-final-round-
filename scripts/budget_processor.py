"""
scripts/budget_processor.py
============================
Inference-time hard budget enforcement for the Thinking Budget environment.

PROBLEM
-------
Reward shaping during training teaches the policy to allocate <think> tokens
intelligently — long on bugs, short on safe files.  But at inference time,
nothing prevents a model from blowing past its own learned budget if the
sampling distribution drifts.

CONTRIBUTION
------------
A `LogitsProcessor` that converts a soft, learned budget into a *hard*
inference-time constraint.  The processor:
  • counts tokens emitted between an opening <think> and its closing </think>
  • when the per-block budget is exceeded, forces the next sampled token to
    be the </think> token id, ending reasoning gracefully
  • respects an *episode-level* budget across all <think> blocks too —
    once the global pool is empty no further reasoning is allowed

This is the inference complement to the training-time reward.  Together
they form a closed loop:  reward shaping during GRPO  →  the policy learns
to allocate;  budget processor at inference  →  the user can hard-cap
compute and the policy still degrades gracefully.

USAGE
-----
    from transformers import AutoTokenizer
    from scripts.budget_processor import ThinkingBudgetProcessor

    tok = AutoTokenizer.from_pretrained(...)
    proc = ThinkingBudgetProcessor(
        tokenizer=tok,
        per_block_budget=400,    # max tokens per <think>...</think>
        episode_budget=2000,     # max total thinking tokens per generation
    )
    out = model.generate(..., logits_processor=[proc])

DEMO
----
The Gradio Space (Tab "🎚 Budget Slider") wraps this processor with a
slider [50 / 100 / 250 / 500 / 1000 / ∞] and a live F1 readout.  Tighter
budgets degrade the untrained baseline catastrophically while the trained
policy adapts — that's the screenshot.
"""
from __future__ import annotations

from typing import List, Optional

try:
    import torch
    from transformers import LogitsProcessor
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - allow import without torch (CI)
    _HAS_TORCH = False
    LogitsProcessor = object  # type: ignore


# ── Tag ids cache ─────────────────────────────────────────────────────────
def _resolve_tag_ids(tokenizer, tag: str) -> List[int]:
    """
    Resolve all token ids that emit `tag` (with and without leading space,
    BPE-merged variants).  We need this because the same logical </think>
    can be tokenized as several different sequences depending on context.
    """
    candidates = {tag, " " + tag, "\n" + tag, "\n\n" + tag}
    ids: List[int] = []
    for cand in candidates:
        try:
            toks = tokenizer.encode(cand, add_special_tokens=False)
            if toks:
                ids.extend(toks)
        except Exception:
            continue
    seen = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


class ThinkingBudgetProcessor(LogitsProcessor):
    """
    Stateful per-batch logits processor that enforces hard <think> budgets.

    State machine (per sequence in the batch):
        0  outside any <think> block
        1  inside an open <think> block — counting tokens

    When per-block or episode budget hits zero, the next-token logits are
    forced to the most-preferred </think> token id, emitting `</think>` and
    transitioning back to state 0.
    """

    def __init__(
        self,
        tokenizer,
        per_block_budget: int = 400,
        episode_budget: Optional[int] = None,
        verbose: bool = False,
    ):
        if not _HAS_TORCH:
            raise RuntimeError("torch + transformers required for ThinkingBudgetProcessor")
        self.per_block_budget = max(1, int(per_block_budget))
        self.episode_budget = int(episode_budget) if episode_budget else None
        self.verbose = verbose

        self.open_ids = _resolve_tag_ids(tokenizer, "<think>")
        self.close_ids = _resolve_tag_ids(tokenizer, "</think>")
        if not self.open_ids or not self.close_ids:
            raise ValueError("tokenizer does not contain <think> / </think> tokens")
        self.preferred_close_id = self.close_ids[0]

        # Per-sequence state (key: id(input_ids tensor))
        self._state: dict = {}

    def _get_seq_state(self, key) -> dict:
        if key not in self._state:
            self._state[key] = {
                "in_block": False,
                "block_used": 0,
                "episode_used": 0,
            }
        return self._state[key]

    def _scan_last_token(self, last_id: int, st: dict) -> None:
        """Update the state machine based on the last emitted token id."""
        if last_id in self.open_ids:
            st["in_block"] = True
            st["block_used"] = 0
        elif last_id in self.close_ids:
            st["in_block"] = False
        elif st["in_block"]:
            st["block_used"] += 1
            st["episode_used"] += 1

    def _budget_exceeded(self, st: dict) -> bool:
        if st["block_used"] >= self.per_block_budget:
            return True
        if self.episode_budget is not None and st["episode_used"] >= self.episode_budget:
            return True
        return False

    def __call__(self, input_ids: "torch.Tensor", scores: "torch.Tensor") -> "torch.Tensor":
        # input_ids: (batch, seq)   scores: (batch, vocab)
        for b in range(input_ids.shape[0]):
            seq = input_ids[b]
            key = (id(input_ids), b)
            st = self._get_seq_state(key)

            # Initial scan: rebuild state from full sequence on first call
            if not st.get("_initialized"):
                st["_initialized"] = True
                for tid in seq.tolist():
                    self._scan_last_token(int(tid), st)
            else:
                self._scan_last_token(int(seq[-1].item()), st)

            if st["in_block"] and self._budget_exceeded(st):
                # Force the next token to be </think>
                forced = torch.full_like(scores[b], float("-inf"))
                forced[self.preferred_close_id] = 0.0
                scores[b] = forced
                if self.verbose:
                    print(f"[budget] seq {b}: forced </think> at "
                          f"block_used={st['block_used']} "
                          f"episode_used={st['episode_used']}")
        return scores

    def reset(self) -> None:
        """Clear all per-sequence state.  Call between generation runs."""
        self._state.clear()


# ── Lightweight character-level fallback (when no tokenizer is available) ──
def enforce_character_budget(
    text: str,
    per_block_budget: int = 400,
    episode_budget: Optional[int] = None,
) -> str:
    """
    Post-hoc enforcement on already-generated text.  Used by the demo when
    we want to show what a budget *would* have done to a recorded trace.
    Truncates each <think> block to `per_block_budget` characters and
    inserts </think>; tracks total budget across all blocks.
    """
    import re
    out = []
    spent = 0
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    last = 0
    for m in pattern.finditer(text):
        out.append(text[last:m.start()])
        block = m.group(1)
        if episode_budget is not None:
            remaining_episode = max(0, episode_budget - spent)
            cap = min(per_block_budget, remaining_episode)
        else:
            cap = per_block_budget
        if len(block) <= cap:
            kept = block
        else:
            kept = block[:cap].rstrip() + "  …[truncated by budget]"
        out.append(f"<think>{kept}</think>")
        spent += len(kept)
        last = m.end()
    out.append(text[last:])
    return "".join(out)


if __name__ == "__main__":
    sample = "Pre <think>" + "x" * 1000 + "</think> mid <think>" + "y" * 50 + "</think> end"
    out = enforce_character_budget(sample, per_block_budget=200, episode_budget=300)
    print(out[:400])
    assert "[truncated by budget]" in out
    print("✅ character-budget smoke test passed")
