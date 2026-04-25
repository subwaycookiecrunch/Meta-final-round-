"""Composable Rubric system for The Thinking Budget.

Uses OpenEnv's official Rubric API (modeled after nn.Module) to expose
the 6-component reward as introspectable, composable sub-rubrics.

This wraps the existing reward logic from environment.py and
metacognitive_reward.py into the framework's Rubric hierarchy,
enabling judges to inspect individual component scores.

Usage:
    rubric = ThinkingBudgetRubric()
    score = rubric(action, observation)
    # Inspect sub-scores:
    for name, child in rubric.named_children():
        print(f"{name}: {child.last_score}")
"""

from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import WeightedRubric


# ── Individual rubric components ──────────────────────────────────

class F1ScoreRubric(Rubric):
    """Precision × recall on vulnerability detection."""
    def forward(self, action, observation) -> float:
        session = getattr(observation, 'session', None)
        if session is None:
            return 0.0
        flagged = set(session.flagged_files)
        bugs = set(session.bugs)
        if not bugs:
            return 1.0 if not flagged else 0.0
        tp = len(flagged & bugs)
        fp = len(flagged - bugs)
        fn = len(bugs - flagged)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class ReportQualityRubric(Rubric):
    """CVE ID + vuln type + code-level details mentioned in report."""
    def forward(self, action, observation) -> float:
        report = getattr(observation, 'report', '') or ''
        score = 0.0
        if 'CVE-' in report:
            score += 0.4
        if any(kw in report.lower() for kw in ['buffer', 'overflow', 'injection',
               'traversal', 'rce', 'xss', 'privilege', 'race condition']):
            score += 0.3
        if any(kw in report.lower() for kw in ['line', 'function', 'variable', 'call']):
            score += 0.3
        return min(score, 1.0)


class InvestigationEfficiencyRubric(Rubric):
    """Strategic use of investigation budget (fewer wasted steps)."""
    def forward(self, action, observation) -> float:
        session = getattr(observation, 'session', None)
        if session is None:
            return 0.0
        used = session.invest_used
        budget = session.invest_budget
        if budget == 0:
            return 1.0
        efficiency = 1.0 - (used / budget)
        return max(0.0, min(1.0, 0.5 + 0.5 * efficiency))


class ThinkingEfficiencyRubric(Rubric):
    """In-environment proxy for selective deep reasoning.
    
    Rewards the model for spending more thinking on buggy files
    and less on safe files.
    """
    def forward(self, action, observation) -> float:
        session = getattr(observation, 'session', None)
        if session is None:
            return 0.0
        trace = getattr(session, 'thinking_trace', [])
        if not trace:
            return 0.5
        bugs = set(session.bugs)
        bug_lens = [t['length'] for t in trace if t.get('file') in bugs]
        safe_lens = [t['length'] for t in trace if t.get('file') not in bugs]
        bug_avg = sum(bug_lens) / len(bug_lens) if bug_lens else 0
        safe_avg = sum(safe_lens) / len(safe_lens) if safe_lens else 0
        if safe_avg == 0:
            return 1.0 if bug_avg > 0 else 0.5
        ratio = bug_avg / safe_avg
        return min(1.0, ratio / 6.0)


class PrecisionBonusRubric(Rubric):
    """Extra reward for zero false positives."""
    def forward(self, action, observation) -> float:
        session = getattr(observation, 'session', None)
        if session is None:
            return 0.0
        flagged = set(session.flagged_files)
        bugs = set(session.bugs)
        fp = len(flagged - bugs)
        return 1.0 if fp == 0 and len(flagged) > 0 else 0.0


class CalibrationRubric(Rubric):
    """Metacognitive calibration: does actual think length match predicted band?"""
    def forward(self, action, observation) -> float:
        session = getattr(observation, 'session', None)
        if session is None:
            return 0.5
        pairs = getattr(session, 'budget_pairs', [])
        if not pairs:
            return 0.5
        bands = {'short': (0, 80), 'medium': (80, 250), 'long': (250, float('inf'))}
        scores = []
        for pred, actual_len in pairs:
            lo, hi = bands.get(pred, (0, float('inf')))
            if lo <= actual_len < hi:
                scores.append(1.0)
            else:
                dist = min(abs(actual_len - lo), abs(actual_len - hi))
                scores.append(max(0.0, 1.0 - dist / 200.0))
        return sum(scores) / len(scores) if scores else 0.5


class DifficultyAwarenessRubric(Rubric):
    """Metacognitive difficulty awareness: long on bugs, short on safe."""
    def forward(self, action, observation) -> float:
        session = getattr(observation, 'session', None)
        if session is None:
            return 0.5
        triples = getattr(session, 'difficulty_triples', [])
        if not triples:
            return 0.5
        bugs = set(session.bugs)
        scores = []
        for pred, _len, filepath in triples:
            is_bug = filepath in bugs
            if pred == 'long' and is_bug:
                scores.append(1.0)
            elif pred == 'short' and not is_bug:
                scores.append(1.0)
            elif pred == 'medium':
                scores.append(0.5)
            else:
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.5


class CouplingRubric(Rubric):
    """Action coupling: predictions followed by real tool calls.
    
    Acts as a quality gate — orphan predictions cannot game the score.
    Returns 0.5-1.0 (used as a multiplicative modifier in the composite).
    """
    def forward(self, action, observation) -> float:
        session = getattr(observation, 'session', None)
        if session is None:
            return 0.5
        preds = getattr(session, 'prediction_count', 0)
        coupled = getattr(session, 'coupled_count', 0)
        if preds == 0:
            return 0.5
        ratio = coupled / preds
        return 0.5 + 0.5 * ratio


# ── Composite rubric ──────────────────────────────────────────────

class MetacognitiveCompositeRubric(Rubric):
    """Metacognitive score = (0.5·calibration + 0.5·difficulty) × (0.5 + 0.5·coupling).
    
    The multiplicative coupling term makes the reward non-gameable:
    a policy that emits perfect predictions but never grounds them
    in tool calls gets a 0.5× penalty.
    """
    def __init__(self):
        super().__init__()
        self.calibration = CalibrationRubric()
        self.difficulty_awareness = DifficultyAwarenessRubric()
        self.coupling = CouplingRubric()

    def forward(self, action, observation) -> float:
        cal = self.calibration(action, observation)
        diff = self.difficulty_awareness(action, observation)
        coup = self.coupling(action, observation)
        base = 0.5 * cal + 0.5 * diff
        return base * (0.5 + 0.5 * coup)


class ThinkingBudgetRubric(Rubric):
    """Top-level composable rubric for The Thinking Budget.
    
    total = 0.50 · env_score + 0.30 · metacog_score + 0.20 · text_score
    
    Each component is a proper OpenEnv Rubric subclass, enabling
    introspection via rubric.named_children() and rubric.last_score.
    """
    def __init__(self):
        super().__init__()
        # Environment components (50% of total)
        self.env = WeightedRubric(
            rubrics=[
                F1ScoreRubric(),
                ReportQualityRubric(),
                InvestigationEfficiencyRubric(),
                ThinkingEfficiencyRubric(),
                PrecisionBonusRubric(),
            ],
            weights=[0.35, 0.20, 0.15, 0.15, 0.15],
        )
        # Metacognitive components (30% of total)
        self.metacog = MetacognitiveCompositeRubric()

    def forward(self, action, observation) -> float:
        env_score = self.env(action, observation)
        metacog_score = self.metacog(action, observation)
        # text_score is computed separately from completion text
        # (not available in the Rubric action/observation interface)
        text_score = 0.5  # default; overridden at training time
        return 0.50 * env_score + 0.30 * metacog_score + 0.20 * text_score
