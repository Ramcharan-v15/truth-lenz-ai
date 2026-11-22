"""Benchmark metrics (Phase 10).

Pure-numpy implementations. Binary metrics over (y_true, y_prob) with a
decision threshold, plus threshold-free metrics (ROC-AUC, PR-AUC) and
calibration error (ECE). Probabilities are expected on a 0..1 scale.

RULE: do not publish benchmark claims unless backed by an actual evaluation
run - the runner records dataset size, version and date with every result.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    """Accuracy, precision, recall, F1, specificity, FPR, FNR + confusion counts."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    return {
        "n": int(len(y_true)),
        "threshold": threshold,
        "accuracy": round(_safe_div(tp + tn, tp + tn + fp + fn), 4),
        "precision": round(_safe_div(tp, tp + fp), 4),
        "recall": round(_safe_div(tp, tp + fn), 4),
        "f1": round(_safe_div(2 * tp, 2 * tp + fp + fn), 4),
        "specificity": round(_safe_div(tn, tn + fp), 4),
        "false_positive_rate": round(_safe_div(fp, fp + tn), 4),
        "false_negative_rate": round(_safe_div(fn, fn + tp), 4),
        **cm,
    }


def roc_auc(y_true, y_prob) -> float:
    """Area under ROC curve via the Mann-Whitney rank statistic."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # P(score(pos) > score(neg)) + 0.5 * P(tie)
    wins = np.sum(pos[:, None] > neg[None, :])
    ties = np.sum(pos[:, None] == neg[None, :])
    return round(float((wins + 0.5 * ties) / (len(pos) * len(neg))), 4)


def pr_auc(y_true, y_prob) -> float:
    """Average precision (area under precision-recall curve)."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    order = np.argsort(-y_prob, kind="stable")
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted == 1)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return float("nan")
    precision = tp_cum / np.arange(1, len(y_sorted) + 1)
    recall = tp_cum / n_pos
    # step-function integration
    ap = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
    ap += float(recall[0] * precision[0])
    return round(ap, 4)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> Dict:
    """ECE: mean |accuracy - confidence| weighted by bin occupancy."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    detail: List[Dict] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        count = int(np.sum(mask))
        if count == 0:
            detail.append({"bin": [round(lo, 2), round(hi, 2)], "n": 0,
                           "accuracy": None, "mean_confidence": None})
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += (count / len(y_true)) * abs(acc - conf)
        detail.append({"bin": [round(lo, 2), round(hi, 2)], "n": count,
                       "accuracy": round(acc, 4), "mean_confidence": round(conf, 4)})
    return {"ece": round(float(ece), 4), "n_bins": n_bins, "bins": detail}


def brier_score(y_true, y_prob) -> float:
    """Mean squared error of probabilistic predictions. Lower is better."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return round(float(np.mean((y_prob - y_true) ** 2)), 4)


def balanced_accuracy(y_true, y_prob, threshold: float = 0.5) -> float:
    """Mean of sensitivity and specificity. Robust to class imbalance."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    recalls = []
    for cls in (0, 1):
        m = y_true == cls
        if m.sum() == 0:
            continue
        recalls.append(float(np.mean(y_pred[m] == cls)))
    if not recalls:
        return 0.0
    return round(float(np.mean(recalls)), 4)


def full_evaluation(y_true, y_prob, threshold: float = 0.5) -> Dict:
    """All metrics in one dict."""
    out = binary_metrics(y_true, y_prob, threshold)
    out["roc_auc"] = roc_auc(y_true, y_prob)
    out["pr_auc"] = pr_auc(y_true, y_prob)
    out["calibration"] = expected_calibration_error(y_true, y_prob)
    out["brier_score"] = brier_score(y_true, y_prob)
    out["balanced_accuracy"] = balanced_accuracy(y_true, y_prob, threshold)
    return out
