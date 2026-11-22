"""Experiment tracking (V4, spec §30-31).

Every benchmark/validation run gets:
  experiment_id: EXP-2026-000001 (sequential per year, persisted)
  reproducibility bundle: dataset/model/config hashes, seeds, versions, timestamps
  results: stored JSON with results hash

Nothing is fabricated: metrics come from actual evaluation runs.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Any, Dict, Optional

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                           "validation", "results")
COUNTER_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
                            "validation", "experiment_counter.json")


def _ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(COUNTER_PATH), exist_ok=True)


def next_experiment_id() -> str:
    """Allocate EXP-YYYY-NNNNNN, sequential per year, persisted."""
    _ensure_dirs()
    year = _dt.date.today().year
    try:
        data = json.load(open(COUNTER_PATH))
    except Exception:
        data = {}
    n = int(data.get(str(year), 0)) + 1
    data[str(year)] = n
    with open(COUNTER_PATH, "w") as fh:
        json.dump(data, fh)
    return f"EXP-{year}-{n:06d}"


def _sha256_of(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def build_reproducibility(
    dataset_id: str,
    dataset_version: str,
    dataset_hash: str,
    model_id: str,
    model_version: str,
    model_hash: str,
    config_hash: str,
    seed: int,
    software_version: str = "4.0.0",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the reproducibility metadata bundle (spec §31)."""
    bundle = {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "dataset_hash": dataset_hash,
        "model_id": model_id,
        "model_version": model_version,
        "model_hash": model_hash,
        "configuration_hash": config_hash,
        "random_seed": seed,
        "software_version": software_version,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        bundle.update(extra)
    bundle["reproducibility_hash"] = _sha256_of(bundle)
    return bundle


def save_experiment(experiment_id: str, reproducibility: Dict[str, Any],
                    metrics: Dict[str, Any],
                    status: str = "completed") -> str:
    """Persist experiment results. Returns the results file path."""
    _ensure_dirs()
    record = {
        "experiment_id": experiment_id,
        "status": status,
        "reproducibility": reproducibility,
        "metrics": metrics,
    }
    record["results_hash"] = _sha256_of(
        {"metrics": metrics, "reproducibility": reproducibility})
    path = os.path.join(RESULTS_DIR, f"{experiment_id}.json")
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    return path


def load_experiment(experiment_id: str) -> Optional[Dict[str, Any]]:
    _ensure_dirs()
    path = os.path.join(RESULTS_DIR, f"{experiment_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def list_experiments() -> list:
    _ensure_dirs()
    out = []
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith(".json"):
            try:
                rec = json.load(open(os.path.join(RESULTS_DIR, f)))
                out.append({
                    "experiment_id": rec["experiment_id"],
                    "status": rec.get("status"),
                    "created_at": rec["reproducibility"].get("created_at"),
                    "dataset_id": rec["reproducibility"].get("dataset_id"),
                    "model_id": rec["reproducibility"].get("model_id"),
                    "results_hash": rec.get("results_hash"),
                })
            except Exception:
                continue
    return out


# ---------------------------------------------------------------------------
# V5: experiment lifecycle (spec §32)
# ---------------------------------------------------------------------------

EXPERIMENT_STATUSES = {
    "CREATED", "DATASET_LOCKED", "RUNNING", "COMPLETED",
    "FAILED", "INVALIDATED", "SUPERSEDED",
}

# Legal transitions. After COMPLETED only INVALIDATED (with reason) or
# SUPERSEDED are allowed — never silent re-tuning. INVALIDATED/FAILED are
# terminal; a corrected run is a NEW experiment.
LEGAL_TRANSITIONS = {
    "CREATED": {"DATASET_LOCKED", "FAILED"},
    "DATASET_LOCKED": {"RUNNING", "FAILED", "INVALIDATED"},
    "RUNNING": {"COMPLETED", "FAILED", "INVALIDATED"},
    "COMPLETED": {"INVALIDATED", "SUPERSEDED"},
    "FAILED": set(),
    "INVALIDATED": set(),
    "SUPERSEDED": set(),
}


class IllegalTransitionError(ValueError):
    pass


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _save_record(rec: Dict[str, Any]) -> None:
    _ensure_dirs()
    path = os.path.join(RESULTS_DIR, f"{rec['experiment_id']}.json")
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)


def new_experiment_record(experiment_id: str,
                          reproducibility: Dict[str, Any],
                          kind: str = "benchmark") -> Dict[str, Any]:
    """Create a fresh experiment record with lifecycle status CREATED."""
    rec = {
        "experiment_id": experiment_id,
        "kind": kind,
        "status": "CREATED",
        "status_history": [
            {"status": "CREATED", "at": _now_iso(), "note": "experiment created"}
        ],
        "reproducibility": reproducibility,
        "metrics": {},
        "predictions": [],
        "revealed": False,
    }
    rec["results_hash"] = _sha256_of(
        {"metrics": rec["metrics"], "reproducibility": reproducibility})
    _save_record(rec)
    return rec


def transition(experiment_id: str, new_status: str,
               note: str = "") -> Dict[str, Any]:
    """Move an experiment to a new lifecycle status.

    Validates the transition is legal. Blocks silent changes after COMPLETED:
    only INVALIDATED (explicit reason required) or SUPERSEDED are allowed.
    Records every transition in status_history with a timestamp.
    """
    if new_status not in EXPERIMENT_STATUSES:
        raise IllegalTransitionError(f"unknown status '{new_status}'")
    rec = load_experiment(experiment_id)
    if rec is None:
        raise KeyError(f"experiment '{experiment_id}' not found")
    current = rec.get("status", "CREATED")
    allowed = LEGAL_TRANSITIONS.get(current, set())
    if new_status not in allowed:
        raise IllegalTransitionError(
            f"illegal transition {current} -> {new_status} for {experiment_id}. "
            f"After completion an experiment cannot be silently changed; "
            f"create a new experiment instead.")
    if new_status == "INVALIDATED" and not note:
        raise IllegalTransitionError(
            "INVALIDATED requires an explicit reason (no silent invalidation)")
    rec["status"] = new_status
    rec.setdefault("status_history", []).append(
        {"status": new_status, "at": _now_iso(), "note": note})
    _save_record(rec)
    return rec


def get_status_history(experiment_id: str) -> list:
    rec = load_experiment(experiment_id)
    if rec is None:
        raise KeyError(f"experiment '{experiment_id}' not found")
    return rec.get("status_history", [])


def get_global_config_hash() -> str:
    """SHA-256 (16 hex) of the canonical detector configuration.

    This is the freeze anchor for blind experiments: if the live config hash
    differs from the frozen one after reveal, retuning is blocked.
    """
    cfg_path = os.path.join(os.path.dirname(__file__), "..",
                            "detector_config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
