"""Fit calibration on the held-out calibration split (V3, Phase 11).

Fits temperature scaling: p_cal = sigmoid(logit / T).
Reads model.json + cached features from the training run.

Usage:
    .venv/bin/python -m ml.calibration.calibrate --run ml/experiments/run1

Appends calibration to model.json and writes calibration_report.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ml/experiments/run1")
    args = ap.parse_args()

    with open(os.path.join(args.run, "model.json")) as fh:
        model = json.load(fh)
    d = np.load(os.path.join(args.run, "features_calibration.npz"))
    X, y = d["X"], d["y"]
    # apply the same variance-threshold feature mask used in training
    from ml.preprocessing.features import feature_names as _fn
    keep = [n in model["feature_names"] for n in _fn()]
    X = X[:, np.array(keep)]

    w = np.array(model["weights"])
    b = model["bias"]
    mu = np.array(model["standardization"]["mean"])
    sigma = np.array(model["standardization"]["std"])
    Xs = (X - mu) / sigma
    logits = Xs @ w + b

    # fit temperature on calibration split (held-out from training)
    def nll(T):
        p = expit(logits / T)
        eps = 1e-12
        return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    res = minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded")
    T = float(res.x)

    p_raw = expit(logits)
    p_cal = expit(logits / T)
    ece_raw = _ece(y, p_raw)
    ece_cal = _ece(y, p_cal)

    model["calibration"] = {
        "method": "temperature_scaling",
        "params": {"temperature": round(T, 4), "from_logits": True},
        "fitted_on": "calibration split (held-out from training)",
        "n_calibration": int(len(y)),
        "ece_before": round(ece_raw, 4),
        "ece_after": round(ece_cal, 4),
    }
    with open(os.path.join(args.run, "model.json"), "w") as fh:
        json.dump(model, fh, indent=2)
    with open(os.path.join(args.run, "calibration_report.json"), "w") as fh:
        json.dump(model["calibration"], fh, indent=2)
    print(f"T={T:.4f} | ECE {ece_raw:.4f} -> {ece_cal:.4f}")


def _ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p > bins[i]) & (p <= bins[i + 1])
        if m.sum():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


if __name__ == "__main__":
    main()
