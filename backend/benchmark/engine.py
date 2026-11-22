"""Benchmark engine (V5).

Runs a labeled dataset manifest through the forensic pipeline, computes
per-group statistics, writes a LEAKAGE_REPORT.json, and persists a
reproducible experiment. Marks the experiment INVALIDATED when leakage is
found. Every number comes from actual computation — never fabricated.

Manifest records (dicts) support:
  image_id, path (relative to manifest dir or absolute), class, split,
  generator_family, source, sha256, perceptual_hash, dataset_type,
  hard_negative (bool), hard_negative_category, attributes {camera,
  compression, resolution, screenshot, post_processing, ...}

class -> AI-detection label: ai_generated/ai_edited = 1 (positive),
authentic/manipulated/real_edited/screenshot_reencoded/hard_negatives = 0.
Records with class "unknown" are excluded (cannot label honestly).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .. import validation as _val
from . import failures as _failures
from . import generators as _generators
from . import hard_negatives as _hn
from .runner import _json_safe
from .statistics import full_report

AI_POSITIVE_CLASSES = {"ai_generated", "ai_edited"}
AI_NEGATIVE_CLASSES = {"authentic", "manipulated", "real_edited",
                       "screenshot_reencoded", "hard_negatives", "screenshots",
                       "compressed"}
MANIP_POSITIVE_CLASSES = {"manipulated", "real_edited", "ai_edited"}


def _mime_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp", "bmp": "image/bmp"}.get(ext.lstrip("."), "image/jpeg")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_run_fn(filename, mime, data):
    from backend.services.pipeline import run_analysis
    return run_analysis(filename, mime, data)


def _forensic_pattern(rep: Dict) -> str:
    sigs = rep.get("signals", []) or []
    if not sigs:
        return "unknown"
    best = max(sigs, key=lambda s: abs(float(s.get("ai_score", 50)) - 50))
    if abs(float(best.get("ai_score", 50)) - 50) < 10:
        return "mixed"
    return str(best.get("name", "unknown"))


def _pipeline_attributes(rep: Dict) -> Dict:
    """Bias attributes derived from a report (mirrors runner._attributes)."""
    sigs = {s["name"]: s for s in rep.get("signals", [])}
    q = sigs.get("quality", {}).get("technical", {})
    c = sigs.get("compression", {}).get("technical", {})
    w, h = rep.get("width", 0), rep.get("height", 0)
    mp = (w * h) / 1e6 if w and h else 0
    jq = c.get("estimated_jpeg_quality")
    return {
        "screenshot": "yes" if q.get("screenshot_likely") else "no",
        "compression": ("heavy" if (jq is not None and jq <= 60)
                        else "medium" if (jq is not None and jq <= 85)
                        else "light" if jq is not None else "unknown"),
        "resolution": ("small" if mp < 0.3 else "medium" if mp < 2.0 else "large"),
    }


def _group_reports(rows: List[Dict], key_fn, min_n: int = 2) -> Dict:
    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        groups.setdefault(key_fn(r), []).append(r)
    out = {}
    for k, rs in sorted(groups.items()):
        if len(rs) >= min_n:
            yt = np.array([x["y_true"] for x in rs])
            yp = np.array([x["ai_prob"] for x in rs])
            out[str(k)] = full_report(yt, yp)
        else:
            out[str(k)] = {"status": "INSUFFICIENT_SAMPLE", "n": len(rs)}
    return out


def run_benchmark(dataset_manifest, model_id: str,
                  split_config: Optional[Dict] = None,
                  run_fn: Optional[Callable] = None,
                  output_dir: Optional[str] = None,
                  blind: bool = False,
                  seen_families: tuple = (),
                  task: str = "ai_detection",
                  dataset_type: str = "unknown",
                  seed: int = 42) -> Dict:
    """Evaluate a manifest through the pipeline. Returns the experiment record."""
    from datasets.leakage import check_split_integrity, write_leakage_report

    if isinstance(dataset_manifest, str):
        manifest_dir = os.path.dirname(os.path.abspath(dataset_manifest))
        with open(dataset_manifest) as fh:
            manifest = json.load(fh)
    else:
        manifest = dataset_manifest
        manifest_dir = os.path.abspath(
            manifest.get("manifest_dir", os.path.dirname(__file__)))

    records = list(manifest.get("records", []))
    dataset_id = manifest.get("dataset_id", "unknown-dataset")
    dataset_version = str(manifest.get("version", "1.0"))
    split_config = split_config or {"type": "standard", "eval_split": "test"}
    eval_split = split_config.get("eval_split", "test")
    run_fn = run_fn or _default_run_fn

    experiment_id = _val.next_experiment_id()
    out_dir = output_dir or os.path.join(_val.RESULTS_DIR, experiment_id)
    os.makedirs(out_dir, exist_ok=True)

    # ---- resolve paths + hashes -----------------------------------------
    eval_records = []
    for r in records:
        if r.get("split") != eval_split:
            continue
        p = r.get("path")
        if p and not os.path.isabs(p):
            p = os.path.join(manifest_dir, p)
        rec = dict(r)
        rec["_abs_path"] = p
        if p and os.path.isfile(p) and not rec.get("sha256"):
            try:
                rec["sha256"] = _sha256_file(p)
            except Exception:
                pass
        eval_records.append(rec)

    # ---- leakage ----------------------------------------------------------
    leak_records = []
    for r in records:
        leak_records.append({
            "image_id": r.get("image_id", "?"),
            "sha256": r.get("sha256") or "",
            "perceptual_hash": r.get("perceptual_hash"),
            "split": r.get("split", "none"),
        })
    leakage = check_split_integrity(leak_records)
    leak_path = write_leakage_report(leakage, os.path.join(out_dir, "LEAKAGE_REPORT.json"))

    def _finish(status: str, metrics: Dict) -> Dict:
        repro = _val.build_reproducibility(            dataset_id=dataset_id, dataset_version=dataset_version,
            dataset_hash=manifest.get("dataset_hash", "unknown"),
            model_id=model_id, model_version="see-registry",
            model_hash="see-registry", config_hash="see-config",
            seed=seed, software_version="5.0.0",
            extra={"split_config": split_config, "blind": blind,
                   "task": task, "dataset_type": dataset_type,
                   "leakage_report": leak_path,
                   "seen_families": list(seen_families)})
        # NaN/Inf -> None so saved experiments are strict-JSON compliant
        metrics = _json_safe(metrics)
        path = _val.save_experiment(experiment_id, repro, metrics, status=status)
        rec = _val.load_experiment(experiment_id)
        rec["results_path"] = path
        rec["output_dir"] = out_dir
        return rec

    if not leakage.get("valid", False):
        return _finish("invalidated", {
            "status": "INVALIDATED",
            "reason": "DATA LEAKAGE — cross-split duplicates detected",
            "leakage": leakage,
            "dataset_type": dataset_type,
            "split_config": split_config,
            "blind": blind,
            "note": "Experiment INVALIDATED until leakage is corrected.",
        })

    # ---- inference ----------------------------------------------------------
    pos_classes = MANIP_POSITIVE_CLASSES if task == "manipulation_detection" else AI_POSITIVE_CLASSES
    neg_classes = (set(AI_POSITIVE_CLASSES) | set(AI_NEGATIVE_CLASSES)) - pos_classes
    rows: List[Dict] = []
    for r in eval_records:
        cls = r.get("class", "unknown")
        if cls not in pos_classes and cls not in neg_classes:
            continue  # unknown class: cannot label honestly
        p = r.get("_abs_path")
        row = {"image_id": r.get("image_id"), "class": cls,
               "generator_family": r.get("generator_family", "unknown"),
               "source": r.get("source", "unknown"),
               "split": r.get("split"),
               "hard_negative": bool(r.get("hard_negative")),
               "hard_negative_category": r.get("hard_negative_category", "other"),
               "case_id": None, "y_true": 1 if cls in pos_classes else 0,
               "ai_prob": None, "y_pred": None, "verdict": None,
               "verdict_reason": None, "ood_status": "UNKNOWN",
               "attributes": dict(r.get("attributes", {}) or {}),
               "error": None}
        try:
            if not p or not os.path.isfile(p):
                raise FileNotFoundError(f"image not found: {p}")
            with open(p, "rb") as fh:
                data = fh.read()
            rep = run_fn(os.path.basename(p), _mime_for(p), data)
            ens = rep.get("ensemble", {})
            prob = float(ens.get("ai_probability", 50)) / 100.0
            dets = ens.get("detectors", []) or []
            row.update({
                "case_id": rep.get("case_id"),
                "ai_prob": round(prob, 4),
                "y_pred": int(prob >= 0.5),
                "verdict": ens.get("verdict"),
                "verdict_reason": ens.get("verdict_reason"),
                "ood_status": ens.get("ml_ood_status", "UNKNOWN"),
                "attributes": {**_pipeline_attributes(rep), **row["attributes"]},
            })
            row["attributes"].setdefault("camera", "unknown")
            row["attributes"].setdefault("post_processing", "unknown")
            row["attributes"]["forensic_pattern"] = _forensic_pattern(rep)
            row["attributes"]["model_contribution"] = (
                dets[0].get("probability_ai") if dets and dets[0].get("available") else None)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    ok = [r for r in rows if r["error"] is None]
    with open(os.path.join(out_dir, "rows.json"), "w") as fh:
        json.dump(ok, fh, indent=2)

    # ---- statistics ---------------------------------------------------------
    metrics_out: Dict[str, Any] = {
        "status": "MEASURED",
        "dataset_type": dataset_type,
        "split_config": split_config,
        "blind": blind,
        "task": task,
        "n_evaluated": len(ok),
        "n_failed": len(rows) - len(ok),
        "n_excluded_unknown_class": sum(1 for r in eval_records
                                        if r.get("class", "unknown") not in pos_classes
                                        and r.get("class", "unknown") not in neg_classes),
    }
    if ok:
        yt = np.array([r["y_true"] for r in ok])
        yp = np.array([r["ai_prob"] for r in ok])
        metrics_out["overall"] = full_report(yt, yp)
        metrics_out["groups"] = {
            "class": _group_reports(ok, lambda r: r["class"]),
            "generator_family": _group_reports(ok, lambda r: r["generator_family"]),
            "source": _group_reports(ok, lambda r: r["source"]),
            "ood_status": _group_reports(ok, lambda r: r["ood_status"]),
            "attributes": {
                "screenshot": _group_reports(ok, lambda r: r["attributes"].get("screenshot", "unknown")),
                "compression": _group_reports(ok, lambda r: r["attributes"].get("compression", "unknown")),
            },
        }
        metrics_out["generators"] = _generators.evaluate_generators(ok, seen_families)
        metrics_out["hard_negatives"] = _hn.evaluate_hard_negatives(ok)
        metrics_out["failures"] = _failures.analyze_failures(ok)
        # light prediction rows for API re-analysis
        metrics_out["predictions"] = [
            {k: r.get(k) for k in ("image_id", "case_id", "class",
                                   "generator_family", "source", "ood_status",
                                   "hard_negative", "hard_negative_category",
                                   "y_true", "ai_prob", "y_pred", "verdict",
                                   "verdict_reason", "attributes")}
            for r in ok
        ]
    else:
        metrics_out["overall"] = {"status": "NOT_MEASURED", "n": 0,
                                  "note": "no rows evaluated"}

    return _finish("completed", metrics_out)
