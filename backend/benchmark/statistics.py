"""Statistical reporting (V5).

Honest statistics for benchmark evaluation:
- bootstrap confidence intervals (deterministic, seeded)
- Wilson score intervals for binomial proportions
- full_report(): every metric labeled MEASURED, with CIs and sample-size warnings
- sample_size_warning(): INSUFFICIENT SAMPLE SIZE instead of misleading numbers

numpy only. No new dependencies.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from . import metrics


def bootstrap_ci(y_true, y_prob, metric_fn: Callable, n_boot: int = 1000,
                 seed: int = 42, ci: float = 0.95) -> Dict:
    """Bootstrap confidence interval for a metric. Deterministic given seed.

    metric_fn(y_true_sample, y_prob_sample) -> float. Returns
    {"lo":..., "hi":..., "method":"bootstrap", "n_boot":..., "seed":..., "ci":...}.
    Samples with NaN metric results are skipped.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = metric_fn(y_true[idx], y_prob[idx])
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            vals.append(v)
    alpha = 1.0 - ci
    if not vals:
        return {"lo": None, "hi": None, "method": "bootstrap",
                "n_boot": n_boot, "seed": seed, "ci": ci,
                "note": "no finite bootstrap samples"}
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return {"lo": round(lo, 4), "hi": round(hi, 4), "method": "bootstrap",
            "n_boot": n_boot, "seed": seed, "ci": ci,
            "n_finite": len(vals)}


def binomial_ci(successes: int, n: int, ci: float = 0.95) -> Dict:
    """Wilson score interval for a binomial proportion. Deterministic."""
    if n <= 0:
        return {"lo": None, "hi": None, "method": "wilson",
                "note": "n=0, interval undefined"}
    from math import sqrt
    # z for 95% ~ 1.96; use normal quantile approximation via fixed table
    z_table = {0.90: 1.6449, 0.95: 1.96, 0.99: 2.5758}
    z = z_table.get(round(ci, 2), 1.96)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"lo": round(max(0.0, center - half), 4),
            "hi": round(min(1.0, center + half), 4),
            "method": "wilson", "ci": ci,
            "successes": int(successes), "n": int(n)}


def sample_size_warning(n: int, n_classes: int = 2,
                        min_per_class: int = 30) -> List[str]:
    """Detect weak evaluations. Returns a list of warning strings (may be empty)."""
    warnings: List[str] = []
    if n <= 0:
        warnings.append("NOT MEASURED — no samples")
        return warnings
    if n < n_classes * min_per_class:
        warnings.append(
            f"INSUFFICIENT SAMPLE SIZE — n={n} below {min_per_class}/class "
            f"guideline for {n_classes} classes; treat estimates as unreliable")
    elif n < 100:
        warnings.append(
            f"SMALL SAMPLE — n={n}; estimates are noisy, use confidence intervals")
    return warnings


def _labeled(value, status: str = "MEASURED") -> Dict:
    return {"value": value, "status": status}


def full_report(y_true, y_prob, threshold: float = 0.5,
                n_boot: int = 1000, seed: int = 42) -> Dict:
    """Complete statistical report. Every metric labeled MEASURED.

    Includes bootstrap CIs for accuracy and F1, Wilson interval for accuracy,
    confusion matrix, calibration (ECE), and sample-size warnings.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = int(len(y_true))
    base = metrics.full_evaluation(y_true, y_prob, threshold)

    report: Dict = {
        "n": n,
        "threshold": threshold,
        "status": "MEASURED",
        "metrics": {},
        "warnings": sample_size_warning(n),
    }

    # scalar metrics -> labeled
    for key in ("accuracy", "balanced_accuracy", "precision", "recall",
                "specificity", "false_positive_rate", "false_negative_rate",
                "f1", "roc_auc", "pr_auc", "brier_score"):
        report["metrics"][key] = _labeled(base.get(key))

    # confusion counts -> labeled
    report["confusion_matrix"] = {
        k: _labeled(base.get(k)) for k in ("tp", "tn", "fp", "fn")
    }

    # calibration -> labeled
    cal = base.get("calibration", {}) or {}
    report["calibration"] = {
        "ece": _labeled(cal.get("ece")),
        "brier_score": _labeled(base.get("brier_score")),
        "n_bins": cal.get("n_bins"),
        "bins": cal.get("bins", []),
    }

    # confidence intervals (measured, deterministic)
    acc_fn = lambda yt, yp: metrics.binary_metrics(yt, yp, threshold)["accuracy"]
    f1_fn = lambda yt, yp: metrics.binary_metrics(yt, yp, threshold)["f1"]
    tp_tn = base.get("tp", 0) + base.get("tn", 0)
    report["confidence_intervals"] = {
        "accuracy_bootstrap": bootstrap_ci(y_true, y_prob, acc_fn,
                                           n_boot=n_boot, seed=seed),
        "f1_bootstrap": bootstrap_ci(y_true, y_prob, f1_fn,
                                     n_boot=n_boot, seed=seed + 1),
        "accuracy_wilson": binomial_ci(tp_tn, n),
    }
    return report
