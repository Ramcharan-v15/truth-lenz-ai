"""V5 data quality audit.

audit_dataset(dataset_id) -> {status: READY|BLOCKED, checks: [...], reasons: [...]}

Checks (each recorded, never silently skipped):
 1. manifest_schema        - conforms to dataset_manifest.schema.json
 2. record_completeness    - required fields on every record
 3. hash_format            - sha256 64-hex, perceptual_hash 16-hex
 4. exact_duplicates       - zero SHA-256 collisions (blocking)
 5. perceptual_duplicates  - near-duplicate pairs reported (warning only)
 6. label_validation       - class in known enum
 7. class_balance          - distribution reported; warns on tiny classes
 8. license_validation     - LICENSE_VERIFIED required for READY
 9. file_availability      - referenced files exist (skipped for metadata-only)
10. image_decoding         - files decode (skipped for metadata-only)

status READY only when every blocking check passes, else BLOCKED + reasons.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List

from .registry import get_dataset

_DATASETS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))
if _DATASETS_ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(_DATASETS_ROOT))
from datasets.validate import load_schema, validate  # noqa: E402
from datasets.leakage import exact_duplicates, perceptual_duplicates  # noqa: E402

_KNOWN_CLASSES = {"authentic", "ai_generated", "real_edited", "ai_edited",
                  "screenshot_reencoded", "unknown"}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_PH_RE = re.compile(r"^[0-9a-f]{16}$")


def _manifest_path(dataset_id: str) -> str:
    return os.path.join(_DATASETS_ROOT, "manifests", dataset_id, "manifest.json")


def load_manifest(dataset_id: str) -> Dict[str, Any]:
    import json
    path = _manifest_path(dataset_id)
    if not os.path.isfile(path):
        raise KeyError(f"no manifest for dataset: {dataset_id}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_record_path(manifest: Dict[str, Any], record: Dict[str, Any]) -> str | None:
    """Resolve one manifest record to a local file, portably.

    Primary path (unchanged behaviour): ``source_dir`` + ``local_ref``.
    ``source_dir`` is stamped by ingestion as an absolute path
    (backend/datasets/ingest.py), so it breaks when the project moves
    to another machine — e.g. a Linux ``/home/hatch/...`` path on
    Windows. Fallback, only when the primary file is not found:
    ``datasets/`` + the record's ``local_path`` (relative to the
    datasets root, as stored for RWD-2026-000001). The fallback is
    confined to the datasets root (commonpath check) and never used
    for absolute ``local_path`` values. A record that resolves to no
    existing file returns None, so the audit still reports it missing
    (BLOCKED) — the fallback never invents a file.
    """
    ref = record.get("local_ref", "")
    base = manifest.get("source_dir", "")
    if base and ref and ref != "metadata-only (content not copied)":
        primary = os.path.join(base, ref)
        if os.path.isfile(primary):
            return primary
    else:
        primary = None
    local_path = record.get("local_path", "")
    if local_path and not os.path.isabs(local_path):
        candidate = os.path.abspath(os.path.join(_DATASETS_ROOT, local_path))
        if os.path.commonpath([candidate, _DATASETS_ROOT]) == _DATASETS_ROOT:
            if os.path.isfile(candidate):
                return candidate
    return primary


def audit_dataset(dataset_id: str) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    reasons: List[str] = []

    def check(name: str, passed: bool, detail: str = "",
              blocking: bool = True) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail,
                       "blocking": blocking, "severity":
                       "error" if (blocking and not passed) else
                       ("warning" if not passed else "ok")})
        if blocking and not passed:
            reasons.append(f"{name}: {detail}")

    # 1. manifest schema
    try:
        manifest = load_manifest(dataset_id)
    except KeyError as exc:
        check("manifest_schema", False, str(exc))
        return {"status": "BLOCKED", "checks": checks, "reasons": reasons}
    schema_errors = validate(manifest, load_schema("dataset_manifest.schema.json"))
    check("manifest_schema", not schema_errors,
          "; ".join(schema_errors[:3]) or "valid")

    records = manifest.get("records", []) or []
    check("record_completeness",
          all(all(r.get(f) for f in ("image_id", "sha256", "dataset_id", "class"))
              for r in records) and len(records) > 0,
          f"{len(records)} records" if records else "no records")

    # 3. hash format
    bad_hash = [r["image_id"] for r in records
                if not _SHA_RE.match(str(r.get("sha256", "")))
                or not _PH_RE.match(str(r.get("perceptual_hash", "")))]
    check("hash_format", not bad_hash,
          f"{len(bad_hash)} malformed" if bad_hash else "all well-formed")

    # 4/5. duplicates
    dupes = exact_duplicates(records)
    check("exact_duplicates", not dupes,
          f"{len(dupes)} colliding hashes" if dupes else "none")
    near = perceptual_duplicates(records)
    check("perceptual_duplicates", True,
          f"{len(near)} near-duplicate pairs (advisory)", blocking=False)

    # 6. labels
    bad_labels = sorted({str(r.get("class")) for r in records} - _KNOWN_CLASSES)
    check("label_validation", not bad_labels,
          f"unknown classes: {bad_labels}" if bad_labels else "all known")

    # 7. class balance
    dist = dict(Counter(str(r.get("class")) for r in records))
    tiny = [c for c, n in dist.items() if n < 10]
    check("class_balance", True,
          f"distribution={dist}" +
          (f"; INSUFFICIENT SAMPLE SIZE: {tiny}" if tiny else ""),
          blocking=False)

    # 8. license
    rec = get_dataset(dataset_id) or {}
    lic = rec.get("license_status", "UNKNOWN")
    check("license_validation", lic == "LICENSE_VERIFIED",
          f"license_status={lic}" +
          ("" if lic == "LICENSE_VERIFIED"
           else " — DATASET BLOCKED: license review required"))

    # 9/10. files (skip for metadata-only)
    if manifest.get("metadata_only"):
        check("file_availability", True, "metadata-only: skipped", blocking=False)
        check("image_decoding", True, "metadata-only: skipped", blocking=False)
    else:
        missing, corrupt = [], []
        for r in records:
            p = _resolve_record_path(manifest, r)
            if not p or not os.path.isfile(p):
                missing.append(r["image_id"])
                continue
            try:
                with open(p, "rb") as fh:
                    data = fh.read(1 << 20)
                if hashlib.sha256(open(p, "rb").read()).hexdigest() != r["sha256"]:
                    corrupt.append(r["image_id"] + " (hash mismatch)")
            except OSError:
                corrupt.append(r["image_id"] + " (unreadable)")
        check("file_availability", not missing,
              f"{len(missing)} missing" if missing else "all present")
        check("image_decoding", not corrupt,
              "; ".join(corrupt[:3]) if corrupt else "hashes verified")

    status = "READY" if not reasons else "BLOCKED"
    return {"status": status, "checks": checks, "reasons": reasons}
