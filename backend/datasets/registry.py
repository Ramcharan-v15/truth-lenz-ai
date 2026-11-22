"""V5 dataset registry (real-world dataset ingestion system).

- register_dataset(metadata) -> RWD-YYYY-NNNNNN (sequential, persisted)
- License states: LICENSE_VERIFIED / LICENSE_REQUIRES_REVIEW / RESTRICTED / UNKNOWN
- Default license_status is UNKNOWN. Legal usability is NEVER claimed without
  explicit verification via update_license_status(..., "LICENSE_VERIFIED", ...).

Registry storage: datasets/registry.json (project root datasets/ dir).
This is an additive V5 layer; the V4 datasets/ manifest/leakage/split/blind
modules are untouched and reused.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import threading
from typing import Any, Dict, List, Optional

LICENSE_STATES = {
    "LICENSE_VERIFIED",
    "LICENSE_REQUIRES_REVIEW",
    "RESTRICTED",
    "UNKNOWN",
}

_REGISTRY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "registry.json")
)

_lock = threading.Lock()

_METADATA_FIELDS = [
    "dataset_id", "name", "version", "source", "creator", "acquisition_date",
    "license", "usage_restrictions", "attribution_requirements", "url",
    "provenance", "collection_methodology", "image_count", "class_definitions",
    "generator_info", "camera_info", "license_status",
]


def _load() -> Dict[str, Any]:
    try:
        with open(_REGISTRY_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data.setdefault("datasets", {})
    data.setdefault("counter", {})
    return data


def _save(data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(_REGISTRY_PATH), exist_ok=True)
    tmp = _REGISTRY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, _REGISTRY_PATH)


def _next_id(data: Dict[str, Any]) -> str:
    year = _dt.date.today().year
    n = int(data["counter"].get(str(year), 0)) + 1
    data["counter"][str(year)] = n
    return f"RWD-{year}-{n:06d}"


def register_dataset(metadata: Dict[str, Any]) -> str:
    """Register a dataset. Returns the assigned RWD-YYYY-NNNNNN id.

    license_status defaults to UNKNOWN and is never auto-verified.
    """
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict")
    name = metadata.get("name")
    if not name or not isinstance(name, str):
        raise ValueError("metadata['name'] is required")
    status = metadata.get("license_status", "UNKNOWN")
    if status not in LICENSE_STATES:
        raise ValueError(f"invalid license_status: {status!r}")
    # Never allow self-certification through registration alone: a caller
    # passing LICENSE_VERIFIED at registration is downgraded to
    # LICENSE_REQUIRES_REVIEW until explicitly verified.
    if status == "LICENSE_VERIFIED":
        status = "LICENSE_REQUIRES_REVIEW"

    with _lock:
        data = _load()
        dataset_id = _next_id(data)
        record = {"dataset_id": dataset_id, "registered_at":
                  _dt.datetime.now().isoformat(timespec="seconds"),
                  "license_status": status,
                  "license_notes": metadata.get("license_notes", "")}
        for field in _METADATA_FIELDS:
            if field in ("dataset_id", "license_status"):
                continue
            record[field] = metadata.get(field, "not_provided")
        # normalize list-ish fields
        for lf in ("class_definitions", "generator_info", "camera_info"):
            if record[lf] == "not_provided":
                record[lf] = []
        data["datasets"][dataset_id] = record
        _save(data)
    return dataset_id


def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    return _load()["datasets"].get(dataset_id)


def list_datasets() -> List[Dict[str, Any]]:
    ds = _load()["datasets"]
    return [ds[k] for k in sorted(ds)]


def update_license_status(dataset_id: str, status: str,
                          notes: str = "") -> Dict[str, Any]:
    """Explicitly set license status. This is the ONLY path to LICENSE_VERIFIED.

    Verification must be a human decision recorded with notes; the system
    never infers it.
    """
    if status not in LICENSE_STATES:
        raise ValueError(f"invalid license_status: {status!r}")
    if status == "LICENSE_VERIFIED" and not notes.strip():
        raise ValueError("LICENSE_VERIFIED requires verification notes "
                         "(who verified, what was checked).")
    with _lock:
        data = _load()
        rec = data["datasets"].get(dataset_id)
        if rec is None:
            raise KeyError(f"unknown dataset: {dataset_id}")
        rec["license_status"] = status
        rec["license_notes"] = notes
        rec["license_updated_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        _save(data)
        return rec


def is_legally_usable(dataset_id: str) -> bool:
    """True ONLY when license_status is LICENSE_VERIFIED. Everything else
    means: do not treat as legally usable."""
    rec = get_dataset(dataset_id)
    return bool(rec) and rec.get("license_status") == "LICENSE_VERIFIED"


def registry_path() -> str:
    return _REGISTRY_PATH
