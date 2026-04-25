#!/usr/bin/env python3
"""
scripts/generate_calibration_plot.py
=====================================
Produces grpo_output/calibration_plot.png — the metacognitive-calibration
hero figure for the paper / blog / video.

The plot has THREE panels:

  ┌──────────────────┬──────────────────┬──────────────────┐
  │ A. Confusion     │ B. Calibration   │ C. Allocation    │
  │   matrix:        │   curve:         │   by ground-     │
  │   predicted band │   |actual − pred │   truth label    │
  │   vs actual band │   midpoint|      │                  │
  └──────────────────┴──────────────────┴──────────────────┘

Modes
-----
  --mode heuristic  (default) Build the plot from the heuristic-proxy
                    policy.  This is the figure we ship pre-training to
                    show what calibration *should* look like.
  --mode real       Read calibration data from
                    grpo_output/eval_calibration.json (produced by
                    eval_baseline.py when run on a model trained with
                    metacognitive_reward).

Output
------
  grpo_output/calibration_plot.png
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cve_training_data.json"
OUT_DIR = ROOT / "grpo_output"
DEFAULT_OUT = OUT_DIR / "calibration_plot.png"
DEFAULT_REAL = OUT_DIR / "eval_calibration.json"

BANDS = ["short", "medium", "long"]
MID = {"short": 40, "medium": 165, "long": 400}
RNG_BANDS = {"short": (0, 80), "medium": (80, 250), "long": (250, 800)}


# ── Heuristic data generator ──────────────────────────────────────────────
def _risk(f: dict, cvss: float) -> float:
    feat = f.get("features", [0, 0, 0, 0])
    churn, complexity, _, _ = feat
    s = 0.4 * (churn / 100.0) + 0.4 * (complexity / 100.0) + 0.2 * (cvss / 10.0)
    if f.get("is_test_file"):
        s *= 0.4
    return s


def _band_for_risk(normalized: float, label: int) -> str:
    """The ORACLE choice — what the trained policy *should* predict."""
    if label == 1:
        return "long" if normalized > 0.4 else "medium"
    return "short" if normalized < 0.5 else "medium"


def heuristic_calibration_data(
    n_episodes: int = 30, rng: random.Random | None = None,
) -> Dict[str, List]:
    """Generate (predicted_band, actual_length, label) triples by simulating
    a metacog policy that emits a band, then thinks for a length sampled
    inside that band with realistic noise."""
    rng = rng or random.Random(7)
    with open(DATA) as fh:
        rows = json.load(fh)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["cveId"], r["repo"])].append(r)
    eps = []
    for (_cve, _repo), files in groups.items():
        if any(f["label"] == 1 for f in files):
            eps.append(files)
        if len(eps) >= n_episodes:
            break

    pred, actual_len, label = [], [], []
    for files in eps:
        cvss = files[0].get("cvss", 0.0)
        risks = [_risk(f, cvss) for f in files]
        rmax = max(risks) if risks else 1.0
        rmin = min(risks) if risks else 0.0
        for f, r in zip(files, risks):
            normalized = (r - rmin) / max(1e-6, rmax - rmin) if rmax > rmin else 0.0
            band = _band_for_risk(normalized, f["label"])
            lo, hi = RNG_BANDS[band]
            # Calibrated: 80% of samples land inside the predicted band
            if rng.random() < 0.85:
                length = rng.randint(lo + 5, max(lo + 6, hi - 5))
            else:
                # 15% miscalibration: sample from a neighbouring band
                length = rng.randint(20, 600)
            pred.append(band)
            actual_len.append(length)
            label.append(f["label"])
    return {"pred": pred, "actual_len": actual_len, "label": label}


# ── Real-mode loader ──────────────────────────────────────────────────────
def real_calibration_data(path: Path) -> Dict[str, List]:
    with open(path) as fh:
        return json.load(fh)


# ── Plotting ──────────────────────────────────────────────────────────────
def _band_for_length(L: int) -> str:
    if L < 80:
        return "short"
    if L < 250:
        return "medium"
    return "long"


def plot(data: Dict[str, List], out_path: Path, title_suffix: str) -> None:
    pred = data["pred"]
    actual = data["actual_len"]
    label = data["label"]
    actual_band = [_band_for_length(L) for L in actual]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Metacognitive Calibration — does the agent know how hard the problem is?"
        f"  {title_suffix}",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # ── Panel A: confusion matrix predicted vs actual band ───────────────
    cm = np.zeros((3, 3), dtype=float)
    for p, a in zip(pred, actual_band):
        cm[BANDS.index(p), BANDS.index(a)] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = axes[0].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks(range(3))
    axes[0].set_yticks(range(3))
    axes[0].set_xticklabels(BANDS)
    axes[0].set_yticklabels(BANDS)
    axes[0].set_xlabel("Actual <think> band")
    axes[0].set_ylabel("Predicted band")
    axes[0].set_title("A. Calibration confusion\n(diag = perfect calibration)")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                         color="white" if cm_norm[i, j] > 0.5 else "black",
                         fontsize=10)
    fig.colorbar(im, ax=axes[0], fraction=0.045, pad=0.04)

    # On-diagonal calibration accuracy (single number)
    diag = float(np.trace(cm) / max(1, cm.sum()))
    axes[0].text(0.5, -0.18, f"diag accuracy = {diag:.2f}",
                 transform=axes[0].transAxes,
                 ha="center", fontsize=11, fontweight="bold",
                 color="#2c3e50")

    # ── Panel B: |actual − band midpoint| as calibration error ───────────
    errs = [abs(L - MID[p]) for p, L in zip(pred, actual)]
    median_err = float(np.median(errs))
    axes[1].hist(errs, bins=30, color="#7faecf", edgecolor="white", alpha=0.85)
    axes[1].axvline(median_err, color="#a23a30", ls="--", lw=2.0,
                    label=f"median error = {median_err:.0f} chars")
    axes[1].set_xlabel("|actual length − predicted-band midpoint|  (characters)")
    axes[1].set_ylabel("Number of decisions")
    axes[1].set_title("B. Calibration error distribution")
    axes[1].legend(fontsize=10, loc="upper right")
    axes[1].grid(True, alpha=0.25)

    # ── Panel C: allocation by ground-truth label ─────────────────────────
    by_band_buggy = [BANDS.index(p) for p, lbl in zip(pred, label) if lbl == 1]
    by_band_safe = [BANDS.index(p) for p, lbl in zip(pred, label) if lbl == 0]
    counts_buggy = [by_band_buggy.count(i) for i in range(3)]
    counts_safe = [by_band_safe.count(i) for i in range(3)]

    x = np.arange(3)
    width = 0.38
    axes[2].bar(x - width / 2, counts_safe, width, color="#7faecf", label="safe files",
                edgecolor="white")
    axes[2].bar(x + width / 2, counts_buggy, width, color="#d6584d", label="buggy files",
                edgecolor="white")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(BANDS)
    axes[2].set_xlabel("Predicted budget band")
    axes[2].set_ylabel("Number of decisions")
    axes[2].set_title("C. Difficulty awareness — who gets 'long'?")
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.25, axis="y")

    long_on_bug = counts_buggy[2] / max(1, sum(counts_buggy))
    long_on_safe = counts_safe[2] / max(1, sum(counts_safe))
    axes[2].text(0.5, -0.18,
                 f"P(long | buggy) = {long_on_bug:.2f}    "
                 f"P(long | safe)  = {long_on_safe:.2f}",
                 transform=axes[2].transAxes,
                 ha="center", fontsize=11, fontweight="bold",
                 color="#2c3e50")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Wrote {out_path}")
    print(f"   diag={diag:.2f}  median_err={median_err:.0f}  "
          f"P(long|buggy)={long_on_bug:.2f}  P(long|safe)={long_on_safe:.2f}")


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["heuristic", "real"], default="heuristic")
    ap.add_argument("--data", default=str(DEFAULT_REAL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.mode == "real":
        path = Path(args.data)
        if not path.exists():
            print(f"❌ {path} not found, falling back to heuristic.", file=sys.stderr)
            args.mode = "heuristic"
        else:
            data = real_calibration_data(path)
            plot(data, Path(args.out),
                 title_suffix="(real trained-model calibration)")
            return

    rng = random.Random(args.seed)
    data = heuristic_calibration_data(n_episodes=40, rng=rng)
    plot(data, Path(args.out),
         title_suffix="(heuristic proxy — replace with real traces post-training)")


if __name__ == "__main__":
    main()
