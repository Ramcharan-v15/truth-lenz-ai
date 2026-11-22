"""V5 dataset ingestion pipeline.

ingest_dataset(source_dir, metadata, ...) ->
    validates images, computes SHA-256 + dHash, detects duplicates,
    writes a manifest conforming to datasets/schemas/dataset_manifest.schema.json.

Security (mirrors backend/api/routes.py rules):
- magic-byte validation (never trusts extension)
- 25 MB per-file limit (config override)
- corrupt/undecodable images rejected
- all input treated as untrusted; path traversal neutralized

metadata_only=True: store hashes + references without copying restricted
image content (license-safe mode).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from .registry import get_dataset, register_dataset

# Reuse the V4 validator + dHash helpers (additive use, no modification).
_DATASETS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))
if _DATASETS_ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(_DATASETS_ROOT))
from datasets.validate import load_schema, validate  # noqa: E402
from datasets.leakage import exact_duplicates, hamming_distance  # noqa: E402

_MAGIC = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"BM", "image/bmp"),
)

_MAX_BYTES_DEFAULT = 25 * 1024 * 1024


def detect_image_mime(data: bytes) -> Optional[str]:
    """Magic-byte detection. Returns None for non-images."""
    for magic, mime in _MAGIC:
        if data[: len(magic)] == magic:
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 12 and data[4:8] == b"ftyp" and any(
            b in data[4:12] for b in (b"heic", b"heix", b"hevc", b"hevx", b"mif1")):
        return "image/heic"
    return None


def dhash_bytes(data: bytes) -> Optional[str]:
    """64-bit dHash (16 hex chars) from raw image bytes. None if undecodable."""
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(data)).convert("L").resize((9, 8), Image.LANCZOS)
        px = list(img.get_flattened_data())
        bits = 0
        for y in range(8):
            for x in range(8):
                bits = (bits << 1) | (1 if px[y * 9 + x] > px[y * 9 + x + 1] else 0)
        return f"{bits:016x}"
    except Exception:
        return None


def decode_ok(data: bytes) -> Tuple[bool, str]:
    """Verify the image actually decodes (catches corrupt files)."""
    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
        return True, ""
    except Exception as exc:
        return False, f"undecodable image: {type(exc).__name__}"


def _safe_name(name: str) -> str:
    base = os.path.basename(name).replace("\\", "/").split("/")[-1]
    return base[:120] or "image"


def _image_files(source_dir: str) -> List[str]:
    out = []
    for root, _dirs, files in os.walk(source_dir):
        for f in sorted(files):
            p = os.path.join(root, f)
            # neutralize path traversal: must stay inside source_dir
            if os.path.abspath(p).startswith(os.path.abspath(source_dir) + os.sep):
                out.append(p)
    return out


def ingest_dataset(source_dir: str,
                   metadata: Dict[str, Any],
                   class_label: str = "unknown",
                   metadata_only: bool = False,
                   max_bytes: int = _MAX_BYTES_DEFAULT,
                   dhash_threshold: int = 5) -> Dict[str, Any]:
    """Ingest a directory of images into a registered dataset manifest.

    Returns {dataset_id, manifest_path, records, issues, counts}.
    Raises ValueError on fatal problems (missing dir, no valid images).
    """
    if not os.path.isdir(source_dir):
        raise ValueError(f"source_dir not found: {source_dir}")
    dataset_id = register_dataset(metadata)
    now = _dt.datetime.now().isoformat(timespec="seconds")

    records: List[Dict[str, Any]] = []
    issues: List[str] = []
    counts = {"scanned": 0, "accepted": 0, "rejected_not_image": 0,
              "rejected_too_large": 0, "rejected_corrupt": 0}

    seen_sha: Dict[str, str] = {}
    for path in _image_files(source_dir):
        counts["scanned"] += 1
        rel = os.path.relpath(path, source_dir)
        try:
            size = os.path.getsize(path)
        except OSError:
            issues.append(f"{rel}: unreadable file")
            continue
        if size > max_bytes:
            counts["rejected_too_large"] += 1
            issues.append(f"{rel}: exceeds size limit ({size} bytes)")
            continue
        with open(path, "rb") as fh:
            data = fh.read()
        mime = detect_image_mime(data)
        if mime is None:
            counts["rejected_not_image"] += 1
            issues.append(f"{rel}: not a recognized image (magic bytes)")
            continue
        ok, why = decode_ok(data)
        if not ok:
            counts["rejected_corrupt"] += 1
            issues.append(f"{rel}: {why}")
            continue
        sha = hashlib.sha256(data).hexdigest()
        if sha in seen_sha:
            issues.append(f"{rel}: exact duplicate of {seen_sha[sha]}")
            continue
        ph = dhash_bytes(data)
        if ph is None:
            counts["rejected_corrupt"] += 1
            issues.append(f"{rel}: dHash failed (undecodable)")
            continue
        seen_sha[sha] = rel
        counts["accepted"] += 1
        records.append({
            "image_id": f"{dataset_id}-{counts['accepted']:06d}",
            "sha256": sha,
            "perceptual_hash": ph,
            "dataset_id": dataset_id,
            "dataset_version": metadata.get("version", "1.0"),
            "source_type": metadata.get("source_type", "unknown"),
            "class": class_label,
            "generator_family": metadata.get("generator_family", "not_provided"),
            "format": mime.split("/")[1],
            "metadata_state": "not_provided",
            "split": "none",
            "license": metadata.get("license", "not_provided"),
            "provenance": metadata.get("provenance", "not_provided"),
            "local_ref": ("metadata-only (content not copied)"
                          if metadata_only else rel),
        })

    if not records:
        raise ValueError("ingestion produced zero valid images; "
                         f"issues: {issues[:5]}")

    # perceptual near-duplicate scan
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if hamming_distance(records[i]["perceptual_hash"],
                                records[j]["perceptual_hash"]) <= dhash_threshold:
                issues.append(
                    f"perceptual near-duplicate: {records[i]['image_id']} ~ "
                    f"{records[j]['image_id']}")

    manifest = {
        "dataset_id": dataset_id,
        "version": metadata.get("version", "1.0"),
        "created_at": now,
        "description": metadata.get("name", dataset_id),
        "total_images": len(records),
        "validation_status": "NOT_YET_VALIDATED",
        "source": metadata.get("source", "not_provided"),
        "source_dir": os.path.abspath(source_dir),
        "metadata_only": metadata_only,
        "records": records,
    }
    errors = validate(manifest, load_schema("dataset_manifest.schema.json"))
    if errors:
        raise ValueError(f"manifest failed schema validation: {errors[:3]}")

    out_dir = os.path.join(_DATASETS_ROOT, "manifests", dataset_id)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # keep registry image_count in sync
    rec = get_dataset(dataset_id) or {}
    _sync_registry_count(dataset_id, rec, len(records))

    return {"dataset_id": dataset_id, "manifest_path": manifest_path,
            "records": records, "issues": issues, "counts": counts}


def _sync_registry_count(dataset_id: str, rec: Dict[str, Any], n: int) -> None:
    import json as _json
    path = os.path.join(_DATASETS_ROOT, "registry.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = _json.load(fh)
        if dataset_id in data.get("datasets", {}):
            data["datasets"][dataset_id]["image_count"] = n
            with open(path, "w", encoding="utf-8") as fh:
                _json.dump(data, fh, indent=2, sort_keys=True)
    except (FileNotFoundError, json.JSONDecodeError):
        pass


def validate_ingestion(manifest: Dict[str, Any]) -> List[str]:
    """Return list of issues with an ingestion manifest (empty = clean)."""
    issues: List[str] = []
    errors = validate(manifest, load_schema("dataset_manifest.schema.json"))
    issues.extend(f"schema: {e}" for e in errors)
    records = manifest.get("records", []) or []
    if manifest.get("total_images") != len(records):
        issues.append(f"total_images ({manifest.get('total_images')}) != "
                      f"len(records) ({len(records)})")
    for h, ids in exact_duplicates(records).items():
        issues.append(f"exact duplicate sha256 {h[:12]}... in {ids}")
    seen_ids = set()
    for r in records:
        if r.get("image_id") in seen_ids:
            issues.append(f"duplicate image_id: {r.get('image_id')}")
        seen_ids.add(r.get("image_id"))
        for f in ("image_id", "sha256", "dataset_id", "class"):
            if not r.get(f):
                issues.append(f"record missing {f}: {r.get('image_id')}")
    return issues
