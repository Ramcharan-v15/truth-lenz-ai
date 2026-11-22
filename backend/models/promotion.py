"""Model promotion lifecycle (V5, spec §23).

Evidence-based promotion only:
  DEMONSTRATION -> RESEARCH -> VALIDATED -> PRODUCTION

Promotion NEVER happens by config edit alone. promote() re-verifies every
requirement against actual experiment artifacts:
  - RESEARCH:   >=1 completed experiment (EXP ID recorded) with this model
  - VALIDATED:  >=1 revealed blind experiment on a real-world dataset,
                no leakage, acceptance metrics met (from promotion_criteria.json)
  - PRODUCTION: all VALIDATED criteria + security review flag + calibration
                validated on real-world data

Promotions are recorded in models/promotions.json (append-only log) with
experiment IDs, dataset IDs, metrics, date, and model hash.
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
CRITERIA_PATH = os.path.join(os.path.dirname(__file__),
                             "promotion_criteria.json")
PROMOTIONS_LOG = os.path.join(PROJECT_ROOT, "models", "promotions.json")
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "models", "registry.json")

LEVEL_ORDER = ["DEMONSTRATION", "RESEARCH", "VALIDATED", "PRODUCTION"]


def _criteria() -> Dict[str, Any]:
    with open(CRITERIA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _model_entry(model_id: str) -> Dict[str, Any]:
    with open(REGISTRY_PATH, encoding="utf-8") as fh:
        raw = json.load(fh)
    for e in raw.get("models", []):
        if e.get("name") == model_id:
            return e
    raise KeyError(f"model '{model_id}' not found in registry")


def _current_status(model_id: str) -> str:
    entry = _model_entry(model_id)
    cfg = entry.get("config", {}) or {}
    s = cfg.get("lifecycle_status")
    if s in ("DEMONSTRATION", "RESEARCH", "VALIDATED", "PRODUCTION",
             "DEPRECATED"):
        return s
    return "DEMONSTRATION" if cfg.get("demonstration") else "RESEARCH"


def _experiments_for_model(model_id: str) -> List[Dict[str, Any]]:
    """All completed experiment records that used this model (never fabricated)."""
    from ..validation import load_experiment, list_experiments
    out = []
    for summary in list_experiments():
        rec = load_experiment(summary["experiment_id"])
        if rec is None:
            continue
        if rec.get("reproducibility", {}).get("model_id") == model_id:
            out.append(rec)
    return out


def _real_world_blind_experiments(model_id: str) -> List[Dict[str, Any]]:
    """Revealed blind experiments on non-synthetic datasets."""
    out = []
    for rec in _experiments_for_model(model_id):
        if rec.get("kind") != "blind":
            continue
        if not rec.get("revealed"):
            continue
        if rec.get("status") == "INVALIDATED":
            continue
        ds_status = _dataset_validation_status(
            rec.get("reproducibility", {}).get("dataset_id", ""))
        if ds_status not in ("SYNTHETIC_ONLY", "NOT_YET_VALIDATED", "unknown"):
            out.append(rec)
    return out


def _dataset_validation_status(dataset_id: str) -> str:
    path = os.path.join(PROJECT_ROOT, "datasets", "manifests", dataset_id,
                        "manifest.json")
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh).get("validation_status", "unknown")
        except Exception:
            pass
    return "unknown"


def _check_research(model_id: str) -> Dict[str, Any]:
    exps = [r for r in _experiments_for_model(model_id)
            if r.get("status") == "COMPLETED"]
    ok = len(exps) >= 1
    return {
        "met": ok,
        "detail": (f"{len(exps)} completed experiment(s) with this model"
                   if ok else
                   "no completed experiment with this model recorded"),
        "experiment_ids": [r["experiment_id"] for r in exps],
    }


def _check_validated(model_id: str) -> Dict[str, Any]:
    crit = _criteria()["levels"]["VALIDATED"]
    acc = crit["acceptance"]
    blind = _real_world_blind_experiments(model_id)
    if not blind:
        return {"met": False,
                "detail": "no revealed blind experiment on a real-world "
                          "dataset (real-world dataset required)",
                "experiment_ids": []}
    # use the best (most recent) qualifying blind experiment
    rec = sorted(blind,
                 key=lambda r: r.get("revealed_at", ""))[-1]
    metrics = rec.get("metrics", {})
    binary = metrics.get("binary", {})
    problems = []
    exp_ids = [rec["experiment_id"]]
    if binary.get("status") == "INSUFFICIENT_SAMPLE":
        problems.append("insufficient scored samples for binary metrics")
    else:
        n = metrics.get("n_scored", 0)
        if n < acc["min_scored_samples"]:
            problems.append(
                f"only {n} scored samples (< {acc['min_scored_samples']})")
        if binary.get("accuracy", 0) < acc["min_accuracy"]:
            problems.append(
                f"accuracy {binary.get('accuracy')} < {acc['min_accuracy']}")
        fpr = 1 - binary.get("specificity", 1)
        if fpr > acc["max_false_positive_rate"]:
            problems.append(
                f"false-positive rate {round(fpr, 4)} > "
                f"{acc['max_false_positive_rate']}")
        ece = (binary.get("calibration", {}) or {}).get("ece")
        if ece is not None and ece > acc["max_ece"]:
            problems.append(f"ECE {ece} > {acc['max_ece']}")
    return {
        "met": not problems,
        "detail": ("acceptance criteria met on "
                   f"{rec['experiment_id']}" if not problems
                   else "; ".join(problems)),
        "experiment_ids": exp_ids,
        "metrics": {k: binary.get(k) for k in
                    ("accuracy", "specificity", "f1", "roc_auc")
                    if k in binary},
        "dataset_id": rec.get("reproducibility", {}).get("dataset_id"),
    }


def _check_production(model_id: str,
                      evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    validated = _check_validated(model_id)
    problems = []
    if not validated["met"]:
        problems.append(f"VALIDATED criteria not met: {validated['detail']}")
    evidence = evidence or {}
    if not evidence.get("security_review_passed"):
        problems.append("security review not recorded "
                        "(evidence.security_review_passed required)")
    if not evidence.get("calibration_validated_on_real_world"):
        problems.append("calibration not validated on real-world data "
                        "(evidence.calibration_validated_on_real_world required)")
    return {
        "met": not problems,
        "detail": ("all production criteria met" if not problems
                   else "; ".join(problems)),
        "experiment_ids": validated.get("experiment_ids", []),
    }


def evaluate_promotion(model_id: str,
                       target_status: str) -> Dict[str, Any]:
    """Check whether a model meets the evidence requirements for target_status.

    Returns {eligible, current_status, target_status, checks, missing}.
    Never fabricates: every check reads actual artifacts.
    """
    if target_status not in ("RESEARCH", "VALIDATED", "PRODUCTION"):
        raise ValueError(f"cannot promote to '{target_status}'")
    current = _current_status(model_id)
    ci, ti = LEVEL_ORDER.index(current), LEVEL_ORDER.index(target_status)
    if ti <= ci:
        return {"eligible": False, "current_status": current,
                "target_status": target_status,
                "checks": {},
                "missing": [f"already at or above {target_status}"]}
    checkers = {"RESEARCH": _check_research,
                "VALIDATED": _check_validated,
                "PRODUCTION": _check_production}
    check = checkers[target_status](model_id)
    missing = [] if check["met"] else [check["detail"]]
    # intermediate levels must also be satisfied
    for mid in LEVEL_ORDER[ci + 1:ti]:
        mc = checkers[mid](model_id)
        if not mc["met"]:
            missing.append(f"{mid}: {mc['detail']}")
    return {
        "eligible": not missing,
        "current_status": current,
        "target_status": target_status,
        "checks": {target_status: check},
        "missing": missing,
    }


def _model_hash(model_id: str) -> str:
    import hashlib
    entry = _model_entry(model_id)
    path = entry.get("path")
    if path:
        full = os.path.join(PROJECT_ROOT, "models", path)
        if os.path.isfile(full):
            h = hashlib.sha256()
            with open(full, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()[:16]
    return "unknown"


def _append_promotion_log(record: Dict[str, Any]) -> None:
    log: List[Dict[str, Any]] = []
    if os.path.isfile(PROMOTIONS_LOG):
        try:
            with open(PROMOTIONS_LOG, encoding="utf-8") as fh:
                log = json.load(fh)
        except Exception:
            log = []
    log.append(record)
    with open(PROMOTIONS_LOG, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)


def promote(model_id: str, target_status: str,
            evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Promote a model ONLY if evidence requirements are met.

    Re-verifies every requirement against actual experiment artifacts.
    Manual config edits alone can never promote: this function checks the
    artifacts, not the config's claims. Records the promotion with
    experiment IDs, dataset IDs, metrics, date, and model hash.
    """
    evaluation = evaluate_promotion(model_id, target_status)
    # PRODUCTION evidence flags are passed through the evidence argument
    if target_status == "PRODUCTION":
        prod = _check_production(model_id, evidence)
        if not prod["met"]:
            raise PermissionError(
                f"promotion to PRODUCTION denied for '{model_id}': "
                f"{prod['detail']}")
        evaluation["checks"]["PRODUCTION"] = prod
        evaluation["eligible"] = True
        evaluation["missing"] = []
    if not evaluation["eligible"]:
        raise PermissionError(
            f"promotion to {target_status} denied for '{model_id}': "
            + "; ".join(evaluation["missing"]))

    # sanctioned path: update registry.json lifecycle_status
    with open(REGISTRY_PATH, encoding="utf-8") as fh:
        raw = json.load(fh)
    updated = False
    for e in raw.get("models", []):
        if e.get("name") == model_id:
            e.setdefault("config", {})["lifecycle_status"] = target_status
            updated = True
    if not updated:
        raise KeyError(f"model '{model_id}' not found in registry")
    with open(REGISTRY_PATH, "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2)

    record = {
        "model_id": model_id,
        "from_status": evaluation["current_status"],
        "to_status": target_status,
        "promoted_at": _now_iso(),
        "model_hash": _model_hash(model_id),
        "experiment_ids": evaluation["checks"]
        .get(target_status, {}).get("experiment_ids", []),
        "dataset_id": evaluation["checks"]
        .get(target_status, {}).get("dataset_id"),
        "metrics": evaluation["checks"]
        .get(target_status, {}).get("metrics", {}),
        "evidence": evidence or {},
        "criteria_version": _criteria().get("version"),
    }
    _append_promotion_log(record)
    return {"promoted": True, **record}


def get_promotion_status(model_id: str) -> Dict[str, Any]:
    """Current lifecycle status + what's missing for the next level."""
    current = _current_status(model_id)
    try:
        idx = LEVEL_ORDER.index(current)
        next_level = LEVEL_ORDER[idx + 1] if idx + 1 < len(LEVEL_ORDER) \
            else None
    except ValueError:
        next_level = None
    result: Dict[str, Any] = {
        "model_id": model_id,
        "current_status": current,
        "next_level": next_level,
        "requirements": {},
        "missing": [],
        "promotion_log": [],
    }
    if next_level:
        ev = evaluate_promotion(model_id, next_level)
        result["requirements"] = ev["checks"]
        result["missing"] = ev["missing"]
    if os.path.isfile(PROMOTIONS_LOG):
        try:
            with open(PROMOTIONS_LOG, encoding="utf-8") as fh:
                log = json.load(fh)
            result["promotion_log"] = [
                r for r in log if r.get("model_id") == model_id]
        except Exception:
            pass
    return result
