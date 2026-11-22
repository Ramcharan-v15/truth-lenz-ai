"""Evaluate on the held-out test split (V3, Phase 10).

Uses the REAL benchmark metrics (backend/benchmark/metrics.py).
Per-category breakdowns expose failure modes honestly.

Usage:
    .venv/bin/python -m ml.evaluation.evaluate --run ml/experiments/run1

Writes evaluation_report.json (this is the source of truth for MODEL_CARD.md
and BENCHMARKS.md — never hand-edit the numbers).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from backend.benchmark.metrics import full_evaluation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ml/experiments/run1")
    args = ap.parse_args()

    with open(os.path.join(args.run, "model.json")) as fh:
        model = json.load(fh)
    d = np.load(os.path.join(args.run, "features_test.npz"),
                allow_pickle=True)
    X, y = d["X"], d["y"]
    cats = [str(c) for c in d["cats"]]
    from ml.preprocessing.features import feature_names as _fn
    keep = [n in model["feature_names"] for n in _fn()]
    X = X[:, np.array(keep)]

    w = np.array(model["weights"])
    b = model["bias"]
    mu = np.array(model["standardization"]["mean"])
    sigma = np.array(model["standardization"]["std"])
    Xs = (X - mu) / sigma
    logits = Xs @ w + b
    T = model.get("calibration", {}).get("params", {}).get("temperature", 1.0)
    p = expit(logits / T)

    report = {
        "model": model["name"],
        "version": model["version"],
        "evaluated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_test": int(len(y)),
        "calibration": model.get("calibration", {}).get("method", "none"),
        "overall": _safe(full_evaluation(y, p)),
        "by_category": {},
        "note": ("Held-out SYNTHETIC test set. Metrics describe this dataset only; "
                 "do not generalize to real-world imagery."),
    }
    for cat in sorted(set(cats)):
        m = np.array([c == cat for c in cats])
        if m.sum() >= 2:
            report["by_category"][cat] = _safe(full_evaluation(y[m], p[m]))
    with open(os.path.join(args.run, "evaluation_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    o = report["overall"]
    print(f"n={report['n_test']} acc={o['accuracy']} f1={o['f1']} "
          f"roc_auc={o['roc_auc']} ece={o['calibration']['ece']}")
    for cat, m in report["by_category"].items():
        print(f"  {cat}: n={m['n']} acc={m['accuracy']} f1={m['f1']}")


def _safe(obj):
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    return obj


if __name__ == "__main__":
    main()
