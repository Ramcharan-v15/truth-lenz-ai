"""Train the forensic stacking model (V3, Phase 9).

L2-regularized logistic regression on forensic features, optimized with
scipy L-BFGS. Deterministic: fixed seed, recorded hyperparameters.

Usage:
    .venv/bin/python -m ml.training.train --dataset ml/datasets --out ml/experiments/run1

Outputs:
    features_{train,calibration,test}.npz   cached feature matrices
    model.json                               weights + standardization + metadata
    train_metrics.json                       training-set diagnostics
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ml.preprocessing.features import extract_features, feature_names, label_for  # noqa: E402


def _sigmoid(z):
    return expit(z)


def _load_split(dataset_dir: str, split: str):
    from backend.services.pipeline import run_analysis
    X, y, paths, cats = [], [], [], []
    split_dir = os.path.join(dataset_dir, split)
    for category in sorted(os.listdir(split_dir)):
        cdir = os.path.join(split_dir, category)
        if not os.path.isdir(cdir):
            continue
        for f in sorted(os.listdir(cdir)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                continue
            p = os.path.join(cdir, f)
            with open(p, "rb") as fh:
                data = fh.read()
            ext = os.path.splitext(f)[1].lower().lstrip(".")
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                    "webp": "image/webp", "bmp": "image/bmp"}.get(ext, "image/jpeg")
            rep = run_analysis(f, mime, data, include_visuals=False)
            X.append(extract_features(rep["signals"]))
            y.append(label_for(category))
            paths.append(p)
            cats.append(category)
    return np.array(X), np.array(y), paths, cats


def _fit_logreg(X: np.ndarray, y: np.ndarray, l2_c: float, seed: int):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w0 = rng.normal(0, 0.01, d + 1)

    def loss(w):
        z = X @ w[:d] + w[d]
        p = _sigmoid(z)
        eps = 1e-12
        nll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        return nll + 0.5 * l2_c * np.sum(w[:d] ** 2)

    def grad(w):
        z = X @ w[:d] + w[d]
        p = _sigmoid(z)
        err = (p - y) / n
        g_w = X.T @ err + l2_c * w[:d]
        g_b = np.sum(err)
        return np.concatenate([g_w, [g_b]])

    res = minimize(loss, w0, jac=grad, method="L-BFGS-B",
                   options={"maxiter": 500})
    return res.x[:d], float(res.x[d]), {"success": bool(res.success),
                                        "nit": int(res.nit),
                                        "final_loss": float(res.fun)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml/datasets")
    ap.add_argument("--out", default="ml/experiments/run1")
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args()

    t0 = time.perf_counter()
    os.makedirs(args.out, exist_ok=True)
    cfg = {"seed": args.seed, "l2_c": args.l2, "model": "l2-logistic-regression",
           "features": feature_names(), "dataset": os.path.abspath(args.dataset)}

    splits = {}
    for split in ("train", "calibration", "test"):
        cache = os.path.join(args.out, f"features_{split}.npz")
        if os.path.isfile(cache):
            d = np.load(cache, allow_pickle=True)
            X, y = d["X"], d["y"]
            paths, cats = list(d["paths"]), list(d["cats"])
            print(f"{split}: loaded cached features {X.shape}")
        else:
            print(f"{split}: extracting features (this takes a while)...")
            X, y, paths, cats = _load_split(args.dataset, split)
            np.savez(cache, X=X, y=y, paths=np.array(paths), cats=np.array(cats))
            print(f"{split}: {X.shape}, positives={int(y.sum())}")
        splits[split] = (X, y, paths, cats)

    Xtr, ytr, _, _ = splits["train"]
    # variance threshold: drop near-constant features (they carry no signal
    # and their ~0 std explodes z-scores on new data — found 2026-10-06 when a
    # real photo produced |z| ~ 3.5e8 on constant quality features)
    std_raw = Xtr.std(0)
    keep = std_raw > 1e-3
    dropped = [n for n, k in zip(feature_names(), keep) if not k]
    print(f"variance threshold: keeping {int(keep.sum())}/{len(keep)} features; "
          f"dropped: {dropped}")
    Xtrk = Xtr[:, keep]
    mu, sigma = Xtrk.mean(0), Xtrk.std(0) + 1e-8
    Xtrs = (Xtrk - mu) / sigma

    w, b, opt = _fit_logreg(Xtrs, ytr, args.l2, args.seed)
    ptr = _sigmoid(Xtrs @ w + b)
    acc = float(np.mean((ptr >= 0.5).astype(int) == ytr))

    model = {
        "name": "tl-forensic-stacking",
        "version": "1.0.0",
        "architecture": "l2-logistic-regression on forensic features",
        "feature_names": [n for n, k in zip(feature_names(), keep) if k],
        "dropped_features": dropped,
        "weights": [float(x) for x in w],
        "bias": b,
        "standardization": {"mean": [float(x) for x in mu],
                            "std": [float(x) for x in sigma]},
        "variance_threshold": 1e-3,
        "hyperparameters": {"l2_c": args.l2, "seed": args.seed,
                            "optimizer": "L-BFGS-B", "max_iter": 500},
        "training": {
            "n_train": int(len(ytr)),
            "train_accuracy": round(acc, 4),
            "optimizer_info": opt,
            "trained_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "training_time_s": round(time.perf_counter() - t0, 1),
        },
        "dataset": {"path": cfg["dataset"],
                    "manifest_sha256": _sha(os.path.join(args.dataset, "manifest.json"))},
        "honest_label": ("demonstration model trained on synthetic procedural data; "
                         "not validated on real-world imagery"),
    }
    with open(os.path.join(args.out, "model.json"), "w") as fh:
        json.dump(model, fh, indent=2)
    with open(os.path.join(args.out, "train_metrics.json"), "w") as fh:
        json.dump({"train_accuracy": round(acc, 4), "n": int(len(ytr)),
                   "positives": int(ytr.sum())}, fh, indent=2)
    print(f"train accuracy: {acc:.4f} | model -> {args.out}/model.json")


def _sha(path: str) -> str:
    import hashlib as _h
    try:
        return _h.sha256(open(path, "rb").read()).hexdigest()[:16]
    except Exception:
        return "n/a"


if __name__ == "__main__":
    main()
