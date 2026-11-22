"""Benchmark runner (Phase 10).

Evaluates the pipeline over a labelled dataset directory:

    <dataset>/
      authentic/        camera photos, label 0 / 0
      ai_generated/     synthetic images,  label 1 (AI) / 0 (manip)
      manipulated/      edited photos,      label 0 (AI) / 1 (manip)
      screenshots/      screen captures,   label 0 / 0 (distractor category)
      compressed/       compressed photos, label 0 / 0

Two tasks are evaluated separately: AI-generation detection and manipulation
detection. Per-category breakdowns expose failure modes (e.g. "great on
pristine images, fails on screenshots").

Run: python -m backend.benchmark.runner --dataset benchmark/datasets/sample
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

from .metrics import full_evaluation

# category -> (ai_label, manip_label)
CATEGORY_LABELS = {
    "authentic": (0, 0),
    "ai_generated": (1, 0),
    "manipulated": (0, 1),
    "screenshots": (0, 0),
    "compressed": (0, 0),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dataset_fingerprint(dataset_dir: str) -> str:
    h = hashlib.sha256()
    for root, _, files in os.walk(dataset_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                p = os.path.join(root, f)
                h.update(p.encode())
                h.update(str(os.path.getsize(p)).encode())
    return h.hexdigest()[:16]


def collect(dataset_dir: str) -> List[Dict[str, Any]]:
    items = []
    for category, (ai_label, manip_label) in CATEGORY_LABELS.items():
        cdir = os.path.join(dataset_dir, category)
        if not os.path.isdir(cdir):
            continue
        for f in sorted(os.listdir(cdir)):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                items.append({"path": os.path.join(cdir, f),
                              "category": category,
                              "ai_label": ai_label,
                              "manip_label": manip_label})
    return items


def _json_safe(obj):
    """Replace NaN/Infinity with None so results are strict-JSON compliant."""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _attributes(rep: Dict[str, Any]) -> Dict[str, str]:
    """Bias-analysis attributes (V3, Phase 25) derived from a report."""
    sigs = {s["name"]: s for s in rep.get("signals", [])}
    q = sigs.get("quality", {}).get("technical", {})
    c = sigs.get("compression", {}).get("technical", {})
    w, h = rep.get("width", 0), rep.get("height", 0)
    mp = (w * h) / 1e6 if w and h else 0
    jq = c.get("estimated_jpeg_quality")
    attrs = {
        "screenshot": "yes" if q.get("screenshot_likely") else "no",
        "quality_tier": str(q.get("quality_tier", "unknown")),
        "compression": ("heavy" if (jq is not None and jq <= 60)
                        else "medium" if (jq is not None and jq <= 85)
                        else "light" if jq is not None else "unknown"),
        "resolution": ("small" if mp < 0.3 else "medium" if mp < 2.0 else "large"),
    }
    return attrs


def run_benchmark(dataset_dir: str, run_fn, dataset_version: str = "1.0",
                  include_visuals: bool = False) -> Dict[str, Any]:
    """run_fn(filename, mime, data) -> report dict."""
    items = collect(dataset_dir)
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    for it in items:
        with open(it["path"], "rb") as fh:
            data = fh.read()
        ext = os.path.splitext(it["path"])[1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "webp": "image/webp", "bmp": "image/bmp"}.get(ext.lstrip("."), "image/jpeg")
        try:
            rep = run_fn(os.path.basename(it["path"]), mime, data)
            ens = rep["ensemble"]
            rows.append({**it,
                         "ai_prob": ens["ai_probability"] / 100.0,
                         "manip_prob": ens["manipulation_probability"] / 100.0,
                         "verdict": ens["verdict"],
                         "attributes": _attributes(rep),
                         "error": None})
        except Exception as exc:
            rows.append({**it, "ai_prob": None, "manip_prob": None,
                         "verdict": None, "error": f"{type(exc).__name__}"})

    ok = [r for r in rows if r["error"] is None]
    tasks: Dict[str, Any] = {}
    for task, prob_key, label_key in (("ai_detection", "ai_prob", "ai_label"),
                                      ("manipulation_detection", "manip_prob", "manip_label")):
        y_true = np.array([r[label_key] for r in ok])
        y_prob = np.array([r[prob_key] for r in ok])
        task_entry: Dict[str, Any] = {"overall": full_evaluation(y_true, y_prob),
                                      "by_category": {}}
        for cat in CATEGORY_LABELS:
            sub = [r for r in ok if r["category"] == cat]
            if len(sub) >= 2:
                task_entry["by_category"][cat] = full_evaluation(
                    np.array([r[label_key] for r in sub]),
                    np.array([r[prob_key] for r in sub]))
            elif sub:
                task_entry["by_category"][cat] = {"n": len(sub),
                                                  "note": "too few samples for metrics"}
        tasks[task] = task_entry

    elapsed = time.perf_counter() - started

    # ---- bias & limitations analysis (V3, Phase 25) ----
    bias: Dict[str, Any] = {}
    for attr in ("screenshot", "quality_tier", "compression", "resolution"):
        groups: Dict[str, Any] = {}
        for val in sorted({r["attributes"][attr] for r in ok}):
            sub = [r for r in ok if r["attributes"][attr] == val]
            if len(sub) >= 4:
                yt = np.array([r["ai_label"] for r in sub])
                yp = np.array([r["ai_prob"] for r in sub])
                # needs both classes for ROC; otherwise report accuracy only
                m = full_evaluation(yt, yp) if len(set(yt)) > 1 else {
                    "n": len(sub),
                    "accuracy": round(float(np.mean((yp >= 0.5).astype(int) == yt)), 4),
                    "note": "single-class group: accuracy only"}
                groups[val] = m
        bias[attr] = groups

    return _json_safe({
        "benchmark": "truth-lenz-ai",
        "dataset": os.path.abspath(dataset_dir),
        "dataset_version": dataset_version,
        "dataset_fingerprint": dataset_fingerprint(dataset_dir),
        "evaluated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "n_images": len(ok),
        "n_failed": len(rows) - len(ok),
        "categories": {c: sum(1 for r in ok if r["category"] == c)
                       for c in CATEGORY_LABELS},
        "tasks": tasks,
        "bias_analysis": bias,
        "elapsed_s": round(elapsed, 1),
        "note": ("Metrics describe THIS dataset only. Do not generalize to "
                 "all images. Small datasets produce noisy estimates."),
    })


def main() -> None:
    ap = argparse.ArgumentParser(description="Truth Lenz AI benchmark runner")
    ap.add_argument("--dataset", required=True, help="dataset directory")
    ap.add_argument("--version", default="1.0", help="dataset version label")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from backend.services.pipeline import run_analysis

    result = run_benchmark(args.dataset, run_analysis, dataset_version=args.version)
    out = args.out or os.path.join(args.dataset, "results.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    ai = result["tasks"]["ai_detection"]["overall"]
    mn = result["tasks"]["manipulation_detection"]["overall"]
    print(f"n={result['n_images']} failed={result['n_failed']}")
    print(f"AI task:        acc={ai['accuracy']} f1={ai['f1']} roc_auc={ai['roc_auc']} ece={ai['calibration']['ece']}")
    print(f"Manip task:     acc={mn['accuracy']} f1={mn['f1']} roc_auc={mn['roc_auc']} ece={mn['calibration']['ece']}")
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
