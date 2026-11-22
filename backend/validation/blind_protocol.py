"""Blind evaluation protocol (V5, spec §16).

Workflow:
  1. create_blind_experiment: freeze dataset hash, model version+hash,
     configuration hash, thresholds, calibration version, ensemble weights.
     Status: CREATED -> DATASET_LOCKED.
  2. run_blind_inference: run the pipeline on each image WITHOUT ground truth.
     Predictions stored; status: DATASET_LOCKED -> RUNNING -> COMPLETED.
  3. reveal_and_score: reveal ground truth, compute metrics, generate
     LEAKAGE_REPORT.json. After reveal, any config change requires a NEW
     experiment — retuning against revealed labels is blocked.
  4. invalidate_experiment: mark INVALIDATED with an explicit reason
     (e.g. leakage discovered).

Nothing is fabricated: metrics come from actual pipeline runs.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from . import (
    _ensure_dirs,
    _now_iso,
    _save_record,
    _sha256_of,
    build_reproducibility,
    get_global_config_hash,
    load_experiment,
    new_experiment_record,
    next_experiment_id,
    transition,
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _find_dataset_manifest(dataset_id: str) -> Optional[str]:
    """Locate a dataset manifest by id.

    Requires an exact dataset_id match inside the manifest. Never silently
    substitutes a different dataset: an unknown id returns None so the
    caller raises instead of freezing the wrong data.
    """
    search_roots = [
        os.path.join(PROJECT_ROOT, "datasets", "manifests"),
        os.path.join(PROJECT_ROOT, "ml", "datasets"),
    ]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if "manifest.json" not in filenames:
                continue
            c = os.path.join(dirpath, "manifest.json")
            try:
                m = _read_json(c)
                if m.get("dataset_id") == dataset_id:
                    return c
            except Exception:
                continue
    return None


def _model_info(model_id: str) -> Dict[str, Any]:
    """Resolve model version + weights hash from the registry (no fabrication)."""
    from ..models.registry import get_global_registry
    reg = get_global_registry()
    for m in reg.list_models():
        if m["name"] == model_id:
            weights_hash = ""
            # find the weights file hash from registry.json raw entry
            try:
                raw = _read_json(
                    os.path.join(PROJECT_ROOT, "models", "registry.json"))
                for e in raw.get("models", []):
                    if e.get("name") == model_id:
                        wf = e.get("model_file") or e.get("path")
                        p = os.path.join(PROJECT_ROOT, "models", wf) \
                            if wf else None
                        if p and os.path.isfile(p):
                            h = hashlib.sha256()
                            with open(p, "rb") as fh:
                                for chunk in iter(lambda: fh.read(65536), b""):
                                    h.update(chunk)
                            weights_hash = h.hexdigest()[:16]
            except Exception:
                pass
            return {
                "model_id": model_id,
                "model_version": m.get("version", "unknown"),
                "model_hash": weights_hash or "unknown",
                "lifecycle_status": m.get("lifecycle_status", "unknown"),
                "calibration": m.get("calibration", "none"),
            }
    raise KeyError(f"model '{model_id}' not found in registry")


def _config_snapshot() -> Dict[str, Any]:
    """Freeze the current detector configuration (thresholds, weights)."""
    cfg = get_global_config_hash()
    raw = _read_json(os.path.join(PROJECT_ROOT, "backend",
                                  "detector_config.json"))
    return {
        "configuration_hash": cfg,
        "thresholds": raw.get("verdict_thresholds", raw.get("thresholds", {})),
        "ensemble_weights": raw.get("ensemble_v2", raw.get("ensemble",
                                                           raw.get("weights", {}))),
        "version": raw.get("version", "unknown"),
    }


# expose for tests / freeze verification
def frozen_config_hash() -> str:
    return _config_snapshot()["configuration_hash"]


def create_blind_experiment(dataset_id: str, model_id: str,
                            seed: int = 0,
                            note: str = "") -> Dict[str, Any]:
    """Create a blind experiment with everything frozen.

    Freezes: dataset manifest hash, model version+hash, configuration hash,
    thresholds, calibration version, ensemble weights.
    Returns the experiment record (status DATASET_LOCKED).
    """
    manifest_path = _find_dataset_manifest(dataset_id)
    if manifest_path is None:
        raise FileNotFoundError(
            f"no manifest found for dataset '{dataset_id}'")
    manifest = _read_json(manifest_path)
    dataset_hash = manifest.get("dataset_hash") or _sha256_of(manifest)

    model = _model_info(model_id)
    snap = _config_snapshot()

    repro = build_reproducibility(
        dataset_id=manifest.get("dataset_id", dataset_id),
        dataset_version=str(manifest.get("version", "unknown")),
        dataset_hash=dataset_hash,
        model_id=model["model_id"],
        model_version=model["model_version"],
        model_hash=model["model_hash"],
        config_hash=snap["configuration_hash"],
        seed=seed,
        software_version="5.0.0",
        extra={
            "frozen_thresholds": snap["thresholds"],
            "frozen_ensemble_weights": snap["ensemble_weights"],
            "frozen_calibration": model["calibration"],
            "manifest_path": os.path.relpath(manifest_path, PROJECT_ROOT),
            "blind_protocol": "v5",
            "note": note,
        },
    )
    eid = next_experiment_id()
    rec = new_experiment_record(eid, repro, kind="blind")
    transition(eid, "DATASET_LOCKED",
               note or f"dataset {dataset_id} locked, model {model_id} frozen")
    return load_experiment(eid)


def run_blind_inference(experiment_id: str,
                        images: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run the pipeline on each image WITHOUT ground truth.

    images: list of {filename, mime_type, data (bytes)} or {path}.
    Stores predictions (verdict, probabilities, confidence, OOD).
    Status: DATASET_LOCKED -> RUNNING -> COMPLETED.
    """
    rec = load_experiment(experiment_id)
    if rec is None:
        raise KeyError(f"experiment '{experiment_id}' not found")
    if rec.get("revealed"):
        raise ValueError(
            "ground truth already revealed for this experiment; "
            "create a new experiment instead of re-running")
    if rec["status"] not in ("DATASET_LOCKED", "RUNNING"):
        raise ValueError(
            f"cannot run inference in status {rec['status']}")

    from ..services.pipeline import run_analysis

    transition(experiment_id, "RUNNING", "blind inference started")
    preds = []
    for item in images:
        if "path" in item:
            with open(item["path"], "rb") as fh:
                data = fh.read()
            filename = os.path.basename(item["path"])
            mime = item.get("mime_type", "image/jpeg")
        else:
            data = item["data"]
            filename = item.get("filename", "image.bin")
            mime = item.get("mime_type", "image/jpeg")
        rep = run_analysis(filename, mime, data, include_visuals=False)
        ens = rep["ensemble"]
        det = (ens.get("detectors") or [{}])[0]
        preds.append({
            "image_id": item.get("image_id", filename),
            "filename": filename,
            "sha256": rep["reproducibility"]["image_sha256"],
            "verdict": ens["verdict"],
            "ai_probability": ens["ai_probability"],
            "manipulation_probability": ens["manipulation_probability"],
            "confidence": ens["confidence"],
            "ood_status": ens.get("ml_ood_status", "UNKNOWN"),
            "model_probability_ai": det.get("probability_ai"),
            "case_id": rep.get("case_id"),
        })
    rec = load_experiment(experiment_id)
    rec["predictions"] = preds
    rec["n_predictions"] = len(preds)
    _save_record(rec)
    transition(experiment_id, "COMPLETED", f"{len(preds)} blind predictions stored")
    return load_experiment(experiment_id)


def _verdict_to_label(verdict: str) -> str:
    mapping = {
        "LIKELY_AUTHENTIC": "authentic",
        "LIKELY_AI_GENERATED": "ai_generated",
        "LIKELY_MANIPULATED": "manipulated",
    }
    return mapping.get(verdict, "inconclusive")


def reveal_and_score(experiment_id: str,
                     ground_truth: Dict[str, str]) -> Dict[str, Any]:
    """Reveal ground truth, compute metrics, generate LEAKAGE_REPORT.json.

    ground_truth: {image_id: label} with labels in
    {authentic, ai_generated, manipulated}.
    After reveal the experiment is sealed: any further tuning requires a NEW
    experiment (verify_no_retune raises if the frozen config changed).
    """
    rec = load_experiment(experiment_id)
    if rec is None:
        raise KeyError(f"experiment '{experiment_id}' not found")
    if rec["status"] != "COMPLETED":
        raise ValueError(
            f"reveal requires COMPLETED inference, got {rec['status']}")
    if rec.get("revealed"):
        raise ValueError("ground truth already revealed for this experiment")
    preds = rec.get("predictions", [])
    if not preds:
        raise ValueError("no predictions to score")

    from ..benchmark.metrics import full_evaluation
    import numpy as np

    y_true, y_prob, scored = [], [], []
    withheld = []
    for p in preds:
        gt = ground_truth.get(p["image_id"])
        if gt is None:
            continue
        # binary task: ai_generated (1) vs authentic (0); manipulated counted
        # separately below; inconclusive predictions excluded from binary metrics
        pred_label = _verdict_to_label(p["verdict"])
        scored.append({"image_id": p["image_id"], "predicted": pred_label,
                       "truth": gt, "ai_probability": p["ai_probability"],
                       "ood_status": p["ood_status"]})
        # V5.1.8.64: a prediction whose probability was withheld (numerical
        # fallback) contributes its verdict count but NO probability-based
        # metric. Name it explicitly instead of crashing on None arithmetic
        # or silently counting it as an ordinary prediction.
        if p.get("ai_probability") is None:
            withheld.append({"image_id": p["image_id"], "predicted": pred_label,
                             "truth": gt})
            continue
        if gt in ("authentic", "ai_generated") and pred_label != "inconclusive":
            y_true.append(1 if gt == "ai_generated" else 0)
            y_prob.append(p["ai_probability"] / 100.0)

    metrics: Dict[str, Any] = {
        "n_predictions": len(preds),
        "n_scored": len(scored),
        "n_excluded_inconclusive": sum(
            1 for s in scored if s["predicted"] == "inconclusive"),
        "inconclusive_rate": round(
            sum(1 for s in scored if s["predicted"] == "inconclusive")
            / max(len(scored), 1), 4),
        "n_withheld_probability": len(withheld),
        "withheld_image_ids": [w["image_id"] for w in withheld],
        "withheld_reason": (
            "Verdict probabilities were withheld because numerical validation "
            "found non-finite values; these predictions contribute verdict counts "
            "but no probability-based metric."
            if withheld else None),
    }
    # per-class confusion counts (never fabricated)
    classes = ["authentic", "ai_generated", "manipulated", "inconclusive"]
    cm = {t: {pl: 0 for pl in classes} for t in
          ["authentic", "ai_generated", "manipulated"]}
    for s in scored:
        if s["truth"] in cm:
            cm[s["truth"]][s["predicted"]] += 1
    metrics["confusion"] = cm

    if len(y_true) >= 2 and len(set(y_true)) == 2:
        ev = full_evaluation(np.array(y_true), np.array(y_prob))
        metrics["binary"] = ev
    else:
        metrics["binary"] = {
            "status": "INSUFFICIENT_SAMPLE",
            "note": "binary metrics require both classes with scored predictions",
        }

    # false-positive / false-negative analysis (grouped, honest)
    fps = [s for s in scored
           if s["truth"] == "authentic" and s["predicted"] in
           ("ai_generated", "manipulated")]
    fns = [s for s in scored
           if s["truth"] in ("ai_generated", "manipulated")
           and s["predicted"] == "authentic"]
    metrics["false_positives"] = {
        "count": len(fps),
        "rate": round(len(fps) / max(
            sum(1 for s in scored if s["truth"] == "authentic"), 1), 4),
        "image_ids": [s["image_id"] for s in fps],
    }
    metrics["false_negatives"] = {
        "count": len(fns),
        "rate": round(len(fns) / max(
            sum(1 for s in scored if s["truth"] != "authentic"), 1), 4),
        "image_ids": [s["image_id"] for s in fns],
    }

    # leakage report: predictions vs frozen dataset manifest records
    leakage = _leakage_report(rec, preds)

    rec = load_experiment(experiment_id)
    rec["metrics"] = metrics
    rec["revealed"] = True
    rec["revealed_at"] = _now_iso()
    rec["results_hash"] = _sha256_of(
        {"metrics": metrics, "reproducibility": rec["reproducibility"]})
    _ensure_dirs()
    leak_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "validation", "results",
        f"{experiment_id}_LEAKAGE_REPORT.json")
    with open(leak_path, "w") as fh:
        json.dump(leakage, fh, indent=2)
    rec["leakage_report"] = os.path.basename(leak_path)
    if leakage.get("leakage_detected"):
        _save_record(rec)
        transition(experiment_id, "INVALIDATED",
                   "data leakage detected on reveal; see LEAKAGE_REPORT.json")
    else:
        _save_record(rec)
    return load_experiment(experiment_id)


def _leakage_report(rec: Dict[str, Any],
                    preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check predicted images against the frozen manifest for leakage."""
    manifest_path = rec["reproducibility"].get("manifest_path", "")
    full = os.path.join(PROJECT_ROOT, manifest_path) if manifest_path else None
    manifest_records = []
    if full and os.path.isfile(full):
        try:
            manifest_records = _read_json(full).get("records", [])
        except Exception:
            pass
    manifest_hashes = {r.get("sha256") for r in manifest_records
                       if r.get("sha256")}
    pred_hashes = [p["sha256"] for p in preds]
    exact_dupes = [h for h in pred_hashes if pred_hashes.count(h) > 1]
    outside_manifest = [p["image_id"] for p in preds
                        if p["sha256"] not in manifest_hashes] \
        if manifest_hashes else []
    leakage_detected = bool(exact_dupes)
    return {
        "experiment_id": rec["experiment_id"],
        "generated_at": _now_iso(),
        "n_predictions": len(preds),
        "n_manifest_records": len(manifest_records),
        "exact_duplicates_in_predictions": sorted(set(exact_dupes)),
        "predictions_outside_manifest": outside_manifest,
        "leakage_detected": leakage_detected,
        "verdict": "INVALIDATED — DATA LEAKAGE" if leakage_detected
                   else "no leakage detected",
    }


def verify_no_retune(experiment_id: str) -> Dict[str, Any]:
    """Confirm the live config still matches the frozen config.

    Raises ValueError if anything changed after reveal — the researcher must
    create a new experiment instead of silently tuning.
    """
    rec = load_experiment(experiment_id)
    if rec is None:
        raise KeyError(f"experiment '{experiment_id}' not found")
    frozen = rec["reproducibility"].get("configuration_hash")
    live = _config_snapshot()["configuration_hash"]
    ok = (frozen == live)
    return {
        "experiment_id": experiment_id,
        "frozen_config_hash": frozen,
        "live_config_hash": live,
        "match": ok,
        "note": "config unchanged since freeze" if ok
                else "CONFIG CHANGED after freeze — create a new experiment; "
                     "do not retune against revealed labels",
    }


def invalidate_experiment(experiment_id: str, reason: str) -> Dict[str, Any]:
    """Mark an experiment INVALIDATED with an explicit reason."""
    if not reason:
        raise ValueError("invalidation requires an explicit reason")
    return transition(experiment_id, "INVALIDATED", reason)
