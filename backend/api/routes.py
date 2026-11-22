"""Truth Lenz AI API routes with security hardening:

* magic-byte validation (never trusts extension or claimed MIME)
* upload size limits, per-IP rate limiting
* filename sanitization
* in-memory processing - no temp files, nothing persisted unless save=true
* safe error handling - no raw tracebacks leak to clients
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from .local_guard import require_loopback, require_trusted_browser_origin
from ..services import pipeline, reporting, storage
from ..utils.image_io import load_config

router = APIRouter(prefix="/api")
_config = load_config()
MAX_MB = _config.get("quality", {}).get("max_upload_mb", 25)
MAX_BYTES = MAX_MB * 1024 * 1024
BATCH_LIMIT = 8
# Stage 3 (S3-3): aggregate batch cap, measured as the SUM OF DECODED
# FILE BYTES (not the transport body size, which includes multipart
# framing and is capped separately by the body-limit middleware).
# A batch whose files total more than this is rejected whole, before
# any analysis runs.
BATCH_MAX_BYTES = 100 * 1024 * 1024
# Stage 3 remediation (F1): blind-experiment runs share the batch
# aggregate cap and per-file cap, with their own file-count cap.
# 20 was chosen because no documented workflow uses more (batch
# itself is capped at 8); it bounds memory while leaving generous
# headroom over the batch limit. The count check runs before any
# file is read, mirroring the batch handler.
RUN_MAX_FILES = 20
# Stage 3 (S3-3): cap for privileged JSON request bodies (dataset
# registration, blind-experiment create/reveal/invalidate, model
# promotion). Enforced by _read_json_limited before parsing.
JSON_MAX_BYTES = 1 * 1024 * 1024

# ---- rate limiting: 30 analyze-requests per minute per IP ----
_RATE: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT, _RATE_WINDOW = 30, 60.0


def _check_rate(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = [t for t in _RATE[ip] if now - t < _RATE_WINDOW]
    _RATE[ip] = window
    if len(window) >= _RATE_LIMIT:
        raise HTTPException(429, "Rate limit exceeded. Please wait a moment and try again.")
    window.append(now)


# ---- Stage 3 (S3-4): privileged-mutation rate limits ----
# Same 429 shape as analysis limits, but separate buckets per route
# group so analysis bursts cannot lock out operators (and vice
# versa), and so destructive/expensive operations get stricter caps.
# Keys are (client IP, group); the deployment stays single-process
# loopback-only, so an in-memory dict is the correct store — no proxy
# trust, no distributed infrastructure.
# Stage 3 remediation (F2): the prune -> check -> append sequence is
# guarded by a lock. Synchronous routes run in Starlette's threadpool,
# so without the lock concurrent requests could interleave between
# the check and the append and jointly exceed a bucket's limit. The
# lock covers only the bucket update (no awaits inside, no route
# work), so it cannot serialize unrelated expensive work.
_MUTATION_RATE: dict[tuple[str, str], list[float]] = defaultdict(list)
_MUTATION_LIMITS = {"destructive": 10, "expensive": 5, "mutation": 30}
_MUTATION_WINDOW = 60.0
_MUTATION_LOCK = threading.Lock()


def _check_mutation_rate(request: Request, group: str) -> None:
    ip = request.client.host if request.client else "unknown"
    limit = _MUTATION_LIMITS[group]
    key = (ip, group)
    with _MUTATION_LOCK:
        now = time.time()
        window = [t for t in _MUTATION_RATE[key] if now - t < _MUTATION_WINDOW]
        _MUTATION_RATE[key] = window
        if len(window) >= limit:
            raise HTTPException(429, "Rate limit exceeded. Please wait a moment and try again.")
        window.append(now)


def _sanitize_filename(name: str) -> str:
    name = (name or "upload").split("/")[-1].split("\\")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._") or "upload"
    return name[:120]


def _detect_mime(data: bytes, claimed: str) -> str:
    """Identify the real format from magic bytes."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] == b"BM":
        return "image/bmp"
    if data[4:8] == b"ftyp" and any(b in data[4:12] for b in (b"heic", b"heix", b"hevc", b"hevx", b"mif1")):
        return "image/heic"
    # fall back to claimed type only if it looks plausible
    if claimed in ("image/jpeg", "image/png", "image/webp", "image/bmp"):
        return claimed
    raise ValueError("Unsupported or unrecognized image format. "
                     "Supported: JPG, PNG, WEBP, BMP (HEIC where decodable).")


async def _read_upload(f: UploadFile) -> tuple[str, str, bytes]:
    name = _sanitize_filename(f.filename)
    try:
        data = await f.read()
        if not data:
            raise HTTPException(400, "Empty file received.")
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"File too large. Maximum is {MAX_MB} MB.")
        try:
            mime = _detect_mime(data, (f.content_type or "").lower())
        except ValueError as exc:
            raise HTTPException(415, str(exc))
        return name, mime, data
    finally:
        # Stage 3 (S3-3): always release the uploaded file's spool
        # storage, on success and on every rejection path. The caller
        # only ever uses the returned bytes, never the file object.
        await f.close()


async def _read_json_limited(request: Request, limit: int = JSON_MAX_BYTES):
    """Read a JSON request body with a hard size cap (Stage 3, S3-3).

    Streams the body in chunks with a running total instead of calling
    ``request.json()`` (which buffers the whole body unbounded). Raises
    413 once the cap is crossed, 400 for an empty or malformed body.
    HTTPExceptions propagate unchanged so callers never convert an
    oversize rejection into a generic 400.
    """
    total = 0
    chunks = []
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(413, "Request body too large. Maximum is 1 MB.")
        chunks.append(chunk)
    raw = b"".join(chunks)
    if not raw:
        raise HTTPException(400, "Request body must be valid JSON.")
    try:
        return json.loads(raw)
    except ValueError:
        raise HTTPException(400, "Request body must be valid JSON.")


def _friendly_error(exc: Exception) -> HTTPException:
    msg = str(exc)
    if "decompression bomb" in msg.lower() or "image pixels" in msg.lower():
        return HTTPException(400, "Image dimensions are unreasonably large.")
    if "corrupt" in msg.lower() or "truncated" in msg.lower() or "decode" in msg.lower():
        return HTTPException(400, "Unable to analyze this image: the file appears corrupted. "
                                  "Try exporting it again from its source.")
    if "heic" in msg.lower():
        return HTTPException(415, "HEIC decoding is not available in this environment. "
                                  "Please convert to JPG or PNG and retry.")
    return HTTPException(400, "Unable to analyze this image. "
                              "Possible reason: unsupported or damaged file. "
                              "What you can try: re-export as JPG/PNG and upload again.")


@router.get("/health")
def health():
    return {"status": "ok", "service": "truth-lenz-ai", "version": _config.get("version", "1.0.0"),
            "analyzers": [a.name for a in pipeline.ANALYZERS]}


@router.get("/config")
def get_config():
    """Public detector configuration: weights, thresholds, bands (documented)."""
    return _config


@router.post("/analyze", dependencies=[Depends(require_trusted_browser_origin)])
async def analyze(request: Request, file: UploadFile = File(...),
                  save: bool = Query(False, description="Persist to optional history")):
    _check_rate(request)
    name, mime, data = await _read_upload(file)
    try:
        report = pipeline.run_analysis(name, mime, data)
    except HTTPException:
        raise
    except Exception as exc:
        raise _friendly_error(exc)
    if save:
        storage.save_analysis(report)
        report["saved_to_history"] = True
    return JSONResponse(report)


@router.post("/compare", dependencies=[Depends(require_trusted_browser_origin)])
async def compare(request: Request, file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    _check_rate(request)
    # Stage 3 remediation (F3): read and validate BOTH uploads before
    # analyzing either. Previously the first image was analyzed (and
    # a case counter allocated) before the second file's size was
    # checked, so a rejected request still left a persistent side
    # effect. Now a rejection happens before any analysis runs.
    name_a, mime_a, data_a = await _read_upload(file_a)
    name_b, mime_b, data_b = await _read_upload(file_b)
    out = []
    for name, mime, data in ((name_a, mime_a, data_a), (name_b, mime_b, data_b)):
        try:
            # keep previews: the compare UI shows them side-by-side
            r = pipeline.run_analysis(name, mime, data, include_visuals=True)
        except HTTPException:
            raise
        except Exception as exc:
            raise _friendly_error(exc)
        out.append(r)
    a, b = out
    # V5.1.8.64: verdict probabilities may be null when numerical
    # validation forced an INCONCLUSIVE fallback. Never do arithmetic
    # on nulls (that used to raise TypeError -> bare 500, concealing
    # both analyses). Unavailable deltas stay null - never zero, which
    # would falsely claim identical probabilities. The ml_ai_delta
    # guard below already followed this pattern; ai/manip now match it.
    a_ens, b_ens = a["ensemble"], b["ensemble"]

    def _delta(key: str):
        av, bv = a_ens.get(key), b_ens.get(key)
        if av is None or bv is None:
            return None
        return round(av - bv, 1)

    ai_delta = _delta("ai_probability")
    manip_delta = _delta("manipulation_probability")
    deltas_unavailable = ai_delta is None or manip_delta is None
    diff = {
        "verdict_a": a_ens["verdict"], "verdict_b": b_ens["verdict"],
        "ai_delta": ai_delta,
        "manip_delta": manip_delta,
        "comparison_note": (
            "Verdict probabilities withheld for one or both images "
            "(invalid numeric values); probability deltas are unavailable, not zero."
            if deltas_unavailable else None),
        "ml_ai_a": a_ens.get("ml_ai_probability"),
        "ml_ai_b": b_ens.get("ml_ai_probability"),
        "ml_ai_delta": (round(a_ens["ml_ai_probability"] - b_ens["ml_ai_probability"], 1)
                        if a_ens.get("ml_ai_probability") is not None
                        and b_ens.get("ml_ai_probability") is not None else None),
        "signal_deltas": [
            {"signal": sa["display_name"],
             "ai_a": sa["ai_score"], "ai_b": sb["ai_score"],
             "ai_delta": round(sa["ai_score"] - sb["ai_score"], 1),
             "manip_a": sa["manip_score"], "manip_b": sb["manip_score"],
             "manip_delta": round(sa["manip_score"] - sb["manip_score"], 1)}
            for sa, sb in zip(a["signals"], b["signals"])
        ],
        "metadata_a": next(s for s in a["signals"] if s["name"] == "metadata")["technical"],
        "metadata_b": next(s for s in b["signals"] if s["name"] == "metadata")["technical"],
    }
    return JSONResponse({"image_a": a, "image_b": b, "comparison": diff})


@router.post("/batch", dependencies=[Depends(require_trusted_browser_origin)])
async def batch(request: Request, files: list[UploadFile] = File(...)):
    _check_rate(request)
    if len(files) > BATCH_LIMIT:
        raise HTTPException(400, f"Batch limited to {BATCH_LIMIT} images.")
    # Stage 3 (S3-3): read ALL files first, enforcing the per-file cap
    # (in _read_upload) and the 100 MB aggregate decoded-bytes cap,
    # BEFORE any expensive analysis runs. A breach rejects the whole
    # batch with 413. The finally block closes any files that were
    # never read (e.g. after an aggregate breach) so their spool
    # storage is released; files _read_upload already closed are
    # closed idempotently a second time.
    read: list[tuple[str, str, bytes]] = []
    total = 0
    try:
        for f in files:
            name, mime, data = await _read_upload(f)
            total += len(data)
            if total > BATCH_MAX_BYTES:
                raise HTTPException(413, "Batch total exceeds 100 MB.")
            read.append((name, mime, data))
    finally:
        for f in files:
            await f.close()
    rows = []
    for name, mime, data in read:
        try:
            r = pipeline.run_analysis(name, mime, data, include_visuals=False)
        except Exception:
            rows.append({"filename": name, "error": "Unable to analyze this image."})
            continue
        ens = r["ensemble"]
        quality = next(s for s in r["signals"] if s["name"] == "quality")
        rows.append({
            "filename": name, "analysis_id": r["analysis_id"],
            "verdict": ens["verdict"],
            "ai_probability": ens["ai_probability"],
            "manipulation_probability": ens["manipulation_probability"],
            "confidence": ens["confidence"],
            "confidence_label": ens["confidence_label"],
            "image_quality": quality["technical"].get("quality_tier", "Unknown"),
            "dimensions": f"{r['width']}x{r['height']}",
        })
    return JSONResponse({"results": rows, "count": len(rows)})


@router.get("/history")
def history(limit: int = Query(50, le=200)):
    return {"analyses": storage.list_analyses(limit), "stats": storage.stats()}


@router.get("/analysis/{analysis_id}")
def get_analysis(analysis_id: str):
    report = storage.get_analysis(analysis_id)
    if not report:
        raise HTTPException(404, "Analysis not found in history.")
    return JSONResponse(report)


@router.delete("/history/{analysis_id}", dependencies=[Depends(require_loopback), Depends(require_trusted_browser_origin)])
def delete_one(request: Request, analysis_id: str):
    _check_mutation_rate(request, "destructive")
    if not storage.delete_analysis(analysis_id):
        raise HTTPException(404, "Analysis not found.")
    return {"deleted": analysis_id}


@router.delete("/history", dependencies=[Depends(require_loopback), Depends(require_trusted_browser_origin)])
def delete_history(request: Request):
    _check_mutation_rate(request, "destructive")
    return {"deleted_count": storage.delete_all()}


@router.get("/report/{analysis_id}")
def report(analysis_id: str, format: str = Query("json", pattern="^(json|csv|pdf)$")):
    rep = storage.get_analysis(analysis_id)
    if not rep:
        raise HTTPException(404, "Analysis not found. Save it to history first (save=true).")
    if format == "json":
        return Response(reporting.to_json(rep), media_type="application/json",
                        headers={"Content-Disposition": f"attachment; filename=truthlenz-{analysis_id}.json"})
    if format == "csv":
        return Response(reporting.to_csv(rep), media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=truthlenz-{analysis_id}.csv"})
    pdf = reporting.to_pdf(rep)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=truthlenz-{analysis_id}.pdf"})


# ===========================================================================
# V2 endpoints
# ===========================================================================

@router.get("/models")
def model_status():
    """ML detector registry status. Honest empty state when nothing configured."""
    from ..models.registry import get_global_registry
    try:
        reg = get_global_registry()
        return _public_model_status(reg)
    except Exception as exc:
        return {"total": 0, "ready": 0, "models": [],
                "errors": {"registry": f"{type(exc).__name__}: {exc}"},
                "ml_available": False}


@router.post("/models/reload", dependencies=[Depends(require_loopback), Depends(require_trusted_browser_origin)])
def model_reload(request: Request):
    """Reload models/registry.json without restarting (operator action)."""
    _check_mutation_rate(request, "mutation")
    from ..models.registry import get_global_registry
    pipeline._REGISTRY = None  # force rebuild on next analysis
    reg = get_global_registry()
    pipeline._REGISTRY = reg
    return _public_model_status(reg)


def _public_model_status(reg) -> dict:
    """API-facing registry status.

    Stage 3 (S3-2): the internal status includes the absolute server
    ``models_dir`` path, which must never reach API responses. Strip
    it here, at the route layer, so internal registry behavior (and
    ``get_status()`` itself) is unchanged.
    """
    status = reg.get_status()
    status.pop("models_dir", None)
    return status


@router.get("/models/{model_id}")
def model_detail(model_id: str):
    """Single model record: full registry entry with calibration, license, dataset info."""
    from ..models.registry import get_global_registry
    reg = get_global_registry()
    for m in reg.list_models():
        if m["name"] == model_id:
            return m
    raise HTTPException(404, f"Model '{model_id}' not found in registry.")


@router.get("/benchmarks")
def benchmark_list():
    """List all saved benchmark evaluations (dataset, version, date, summary)."""
    import json as _json
    import os
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmark", "datasets"))
    out = []
    if os.path.isdir(base):
        for dataset in sorted(os.listdir(base)):
            rp = os.path.join(base, dataset, "results.json")
            if os.path.isfile(rp):
                try:
                    with open(rp) as fh:
                        r = _json.load(fh)
                    out.append({
                        "dataset": dataset,
                        "dataset_version": r.get("dataset_version"),
                        "evaluated_at": r.get("evaluated_at"),
                        "n_images": r.get("n_images"),
                        "ai_accuracy": r.get("tasks", {}).get("ai_detection", {}).get("overall", {}).get("accuracy"),
                        "manip_accuracy": r.get("tasks", {}).get("manipulation_detection", {}).get("overall", {}).get("accuracy"),
                    })
                except Exception:
                    continue
    return {"benchmarks": out, "count": len(out)}


@router.get("/robustness")
def robustness_info():
    """Describe the robustness transform suite (variants, not a run)."""
    from ..benchmark.robustness import TRANSFORMS, robustness_score
    return {
        "transforms": [{"id": t["id"], "name": t["name"]} for t in TRANSFORMS],
        "score_formula": ("robustness_score = max(0, 100 - mean(|Δai|, |Δmanip|) "
                          "- 25·[verdict flipped]); range 0-100; >90 stable, "
                          "70-90 moderate, <70 brittle"),
        "usage": "POST /api/robustness with an image to run the full suite.",
        "interpretation": ("High robustness score = predictions survive ordinary processing. "
                           "Low score = brittle detector; treat verdicts on processed images cautiously."),
    }


# ---------------------------------------------------------------------------
# V4: datasets, validation, experiments
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# V5: real-world dataset registry (registered BEFORE /datasets/{dataset_id}
# so FastAPI matches the literal "registry"/"register" paths first)
# ---------------------------------------------------------------------------

@router.post("/datasets/register", dependencies=[Depends(require_loopback), Depends(require_trusted_browser_origin)])
async def datasets_register(request: Request):
    """Register a real-world dataset. Returns RWD-YYYY-NNNNNN id.

    license_status defaults to UNKNOWN; LICENSE_VERIFIED requires the
    separate update_license_status call with verification notes.
    """
    _check_mutation_rate(request, "mutation")
    from ..datasets.registry import register_dataset
    try:
        metadata = await _read_json_limited(request)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Request body must be JSON metadata.")
    if not isinstance(metadata, dict):
        raise HTTPException(400, "Request body must be a JSON object.")
    try:
        dataset_id = register_dataset(metadata)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"dataset_id": dataset_id,
            "license_status": "UNKNOWN",
            "note": ("Registered. License NOT verified — legal usability is "
                     "not claimed. Use the license update flow to verify.")}


@router.get("/datasets/registry")
def datasets_registry_list():
    """List registered real-world datasets with license states."""
    from ..datasets.registry import list_datasets
    ds = list_datasets()
    return {"datasets": [
        {"dataset_id": d["dataset_id"], "name": d.get("name"),
         "version": d.get("version"), "image_count": d.get("image_count"),
         "license": d.get("license"), "license_status": d.get("license_status"),
         "registered_at": d.get("registered_at")}
        for d in ds], "count": len(ds)}


@router.get("/datasets/registry/{dataset_id}")
def datasets_registry_detail(dataset_id: str):
    """Full registry record + data quality audit."""
    from ..datasets.quality import audit_dataset
    from ..datasets.registry import get_dataset
    rec = get_dataset(dataset_id)
    if rec is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not registered.")
    try:
        audit = audit_dataset(dataset_id)
    except KeyError:
        audit = {"status": "BLOCKED",
                 "checks": [],
                 "reasons": ["no manifest ingested yet"]}
    return {"record": rec, "quality_audit": audit}


@router.get("/datasets/registry/{dataset_id}/card")
def datasets_registry_card(dataset_id: str):
    """Dataset card (markdown)."""
    import os as _os
    from ..datasets.cards import card_path, generate_card
    from ..datasets.registry import get_dataset
    if get_dataset(dataset_id) is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not registered.")
    path = card_path(dataset_id)
    if not _os.path.isfile(path):
        generate_card(dataset_id)
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    return Response(content=content, media_type="text/markdown")


def _datasets_dir() -> str:
    import os
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))


@router.get("/datasets")
def datasets_list():
    """List dataset manifests (synthetic, real_world, generators, hard_negatives, robustness)."""
    import json as _json
    import os
    base = os.path.join(_datasets_dir(), "manifests")
    out = []
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            mp = os.path.join(base, name, "manifest.json")
            if os.path.isfile(mp):
                try:
                    m = _json.load(open(mp))
                    out.append({
                        "dataset_id": m.get("dataset_id", name),
                        "version": m.get("version"),
                        "total_images": m.get("total_images"),
                        "validation_status": m.get("validation_status", "NOT_YET_VALIDATED"),
                        "description": (m.get("description") or "")[:120],
                    })
                except Exception:
                    continue
    return {"datasets": out, "count": len(out),
            "note": "Real-world datasets: NOT YET VALIDATED. Synthetic only."}


@router.get("/datasets/{dataset_id}")
def dataset_detail(dataset_id: str):
    """Full dataset manifest."""
    import json as _json
    import os
    mp = os.path.join(_datasets_dir(), "manifests", dataset_id, "manifest.json")
    if not os.path.isfile(mp):
        raise HTTPException(404, f"Dataset '{dataset_id}' not found.")
    return _json.load(open(mp))


@router.get("/datasets/{dataset_id}/validate")
def dataset_validate(dataset_id: str):
    """Validate a dataset manifest against its schema. Honest: reports problems."""
    import json as _json
    import os
    try:
        import jsonschema
        has_js = True
    except ImportError:
        has_js = False
    mp = os.path.join(_datasets_dir(), "manifests", dataset_id, "manifest.json")
    if not os.path.isfile(mp):
        raise HTTPException(404, f"Dataset '{dataset_id}' not found.")
    manifest = _json.load(open(mp))
    issues = []
    for field in ("dataset_id", "version", "total_images", "validation_status"):
        if field not in manifest:
            issues.append(f"missing required field: {field}")
    if has_js:
        sp = os.path.join(_datasets_dir(), "schemas", "dataset_manifest.schema.json")
        if os.path.isfile(sp):
            schema = _json.load(open(sp))
            try:
                jsonschema.validate(manifest, schema)
            except Exception as exc:
                issues.append(f"schema violation: {str(exc)[:200]}")
    else:
        issues.append("jsonschema not installed: structural checks only")
    return {"dataset_id": dataset_id, "valid": len(issues) == 0, "issues": issues}


@router.get("/datasets/{dataset_id}/leakage")
def dataset_leakage(dataset_id: str):
    """Leakage report for a dataset (exact + perceptual duplicates across splits)."""
    import json as _json
    import os
    import sys
    sys.path.insert(0, os.path.join(_datasets_dir(), ".."))
    try:
        from datasets.leakage import check_split_integrity
    except ImportError as exc:
        raise HTTPException(501, f"leakage module unavailable: {exc}")
    mp = os.path.join(_datasets_dir(), "manifests", dataset_id, "manifest.json")
    if not os.path.isfile(mp):
        raise HTTPException(404, f"Dataset '{dataset_id}' not found.")
    manifest = _json.load(open(mp))
    records = manifest.get("records", [])
    if not records and manifest.get("records_file"):
        rf = os.path.join(_datasets_dir(), "manifests", dataset_id,
                          manifest["records_file"])
        if os.path.isfile(rf):
            records = [_json.loads(l) for l in open(rf) if l.strip()]
    report = check_split_integrity(records)
    return {"dataset_id": dataset_id, **report}


@router.get("/validation")
def validation_status():
    """Real-world validation status dashboard data."""
    return {
        "validation_status": "PARTIAL_REAL_WORLD_VALIDATION",
        "synthetic_evaluation": {
            "status": "MEASURED",
            "model": "tl-forensic-stacking v1.0.0",
            "source": "ml/experiments/run1/evaluation_report.json",
            "note": "Synthetic held-out only. Not real-world performance.",
        },
        "real_world": {
            "status": "MEASURED",
            "experiment_id": "EXP-2026-000273",
            "dataset_id": "RWD-2026-000001",
            "dataset": "Columbia Image Splicing Detection Evaluation Dataset",
            "dataset_type": "REAL-WORLD IMAGE SPLICING / MANIPULATION (not AI generation)",
            "n": 1831,
            "n_authentic": 921,
            "n_spliced": 910,
            "task": "splicing/manipulation detection (positive=spliced)",
            "inconclusive_rate": 0.8869,
            "decisive_n": 207,
            "confusion": {"TP": 156, "FP": 51, "TN": 0, "FN": 0},
            "accuracy_decisive": 0.7536,
            "accuracy_95ci": [0.6907, 0.8073],
            "balanced_accuracy_decisive": 0.50,
            "precision_decisive": 0.7536,
            "recall_decisive": 1.0,
            "specificity_decisive": 0.0,
            "f1_decisive": 0.8595,
            "population_fpr": 0.0554,
            "population_fnr": 0.0,
            "brier_decisive": 0.2030,
            "ece_decisive": 0.1475,
            "ood": "all 1831 MILD_SHIFT",
            "note": ("Blind evaluation. System abstains on 88.7% (honest OOD). "
                     "When decisive, biased toward 'manipulated': never predicts "
                     "authentic confidently. Balanced accuracy 0.50 = chance level "
                     "on decisive subset. Covers splicing ONLY; AI-generation "
                     "detection NOT measured on real data."),
        },
        "real_world_ai_generation": {"status": "NOT_MEASURED",
            "note": "No real-world AI-generated image dataset evaluated."},
        "cross_generator": {"status": "NOT_MEASURED"},
        "hard_negatives": {
            "status": "MEASURED",
            "source": "synthetic hard-negative class (n=30 test)",
            "accuracy": 0.9,
            "note": "Synthetic hard negatives only.",
        },
        "robustness": {
            "status": "MEASURED",
            "source": "ROBUSTNESS.md (2026-10-06, synthetic images)",
            "note": "Overall 82.9 (AI) / 94.6 (authentic) on synthetic test images.",
        },
        "ood": {"status": "MEASURED_ON_REAL_DATA",
                "note": "4-level OOD guard active; all 1831 Columbia images flagged MILD_SHIFT."},
    }


@router.get("/validation/{experiment_id}")
def validation_detail(experiment_id: str):
    """Single experiment record with reproducibility metadata."""
    from ..validation import load_experiment
    rec = load_experiment(experiment_id)
    if not rec:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found.")
    return rec


@router.get("/experiments")
def experiments_list():
    """List all tracked experiments."""
    from ..validation import list_experiments
    exps = list_experiments()
    return {"experiments": exps, "count": len(exps)}


@router.get("/experiments/{experiment_id}")
def experiment_detail(experiment_id: str):
    return validation_detail(experiment_id)


@router.get("/calibration")
def calibration_info():
    """Calibration status for registered models."""
    from ..models.registry import get_global_registry
    reg = get_global_registry()
    out = []
    for m in reg.list_models():
        out.append({
            "model": m["name"],
            "version": m["version"],
            "calibration": m.get("calibration"),
            "calibration_detail": m.get("calibration_detail"),
            "status": m["status"],
        })
    return {"models": out,
            "note": "Raw model scores are never presented as probabilities."}


@router.get("/cross-generator")
def cross_generator_info():
    """Cross-generator evaluation status."""
    return {
        "status": "NOT_MEASURED",
        "note": ("No multi-generator real-world dataset available. "
                 "Infrastructure ready: generator-held-out splitting in "
                 "datasets/split.py. Do not claim generalization."),
    }


@router.get("/hard-negatives")
def hard_negatives_info():
    """Hard-negative evaluation status."""
    return {
        "status": "MEASURED_ON_SYNTHETIC",
        "synthetic": {
            "dataset": "ml/datasets/test/hard_negatives",
            "n": 30,
            "accuracy": 0.9,
            "false_positive_rate": 0.1,
        },
        "real_world": "NOT_MEASURED",
        "question": "How often does TRUTH LENZ AI incorrectly accuse a real photograph?",
        "answer": "On synthetic hard negatives: 10% FPR. On real photos: NOT MEASURED.",
    }


@router.get("/ood")
def ood_info():
    """OOD system status and levels."""
    return {
        "levels": ["IN_DISTRIBUTION", "MILD_SHIFT", "OUT_OF_DISTRIBUTION", "UNKNOWN"],
        "method": ("feature-space distance: max|z| of standardized forensic features "
                   "vs training distribution; ≤3 in-distribution, 3-6 mild shift, >6 OOD"),
        "policy": ("OOD does not mean AI. OOD increases uncertainty and may "
                   "contribute to INCONCLUSIVE."),
        "model_status": "tl-forensic-stacking: OOD guard active",
    }


# ---------------------------------------------------------------------------
# V5: benchmark engine, failures, generators, scorecard
# ---------------------------------------------------------------------------

def _load_experiment_or_404(experiment_id: str):
    from ..validation import load_experiment
    rec = load_experiment(experiment_id)
    if not rec:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found.")
    return rec


@router.get("/experiments/{experiment_id}/report.pdf")
def experiment_report_pdf(experiment_id: str):
    """V5 validation experiment report as PDF (§50)."""
    from ..validation import load_experiment
    from ..services import reporting
    from fastapi.responses import Response
    rec = load_experiment(experiment_id)
    if not rec:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found.")
    pdf = reporting.validation_report_pdf(rec)
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition":
                             f"attachment; filename={experiment_id}-report.pdf"})


@router.get("/benchmark/scorecard")
def benchmark_scorecard(experiment_ids: str = Query(None, description="comma-separated experiment IDs; default all")):
    """Generalization scorecard from recorded experiment artifacts."""
    from ..benchmark.scorecard import build_scorecard
    ids = [i.strip() for i in experiment_ids.split(",") if i.strip()] if experiment_ids else None
    return build_scorecard(ids)


@router.get("/benchmark/failures/{experiment_id}")
def benchmark_failures(experiment_id: str):
    """FP/FN/inconclusive analysis for an experiment."""
    from ..benchmark.failures import analyze_failures
    rec = _load_experiment_or_404(experiment_id)
    metrics = rec.get("metrics") or {}
    if metrics.get("status") == "INVALIDATED":
        return {"experiment_id": experiment_id, "status": "INVALIDATED",
                "reason": metrics.get("reason"),
                "note": "Leakage invalidated this experiment; failure analysis withheld."}
    rows = metrics.get("predictions") or []
    if not rows:
        # fall back to stored failures block
        stored = metrics.get("failures")
        if stored:
            return {"experiment_id": experiment_id, **stored}
        return {"experiment_id": experiment_id, "status": "NOT_MEASURED",
                "note": "no prediction rows stored for this experiment"}
    out = analyze_failures(rows)
    out["experiment_id"] = experiment_id
    return out


@router.get("/benchmark/generators/{experiment_id}")
def benchmark_generators(experiment_id: str,
                         seen: str = Query(None, description="comma-separated seen families; default from experiment")):
    """Seen vs unseen generator breakdown for an experiment."""
    from ..benchmark.generators import evaluate_generators
    rec = _load_experiment_or_404(experiment_id)
    metrics = rec.get("metrics") or {}
    if metrics.get("status") == "INVALIDATED":
        return {"experiment_id": experiment_id, "status": "INVALIDATED",
                "reason": metrics.get("reason")}
    seen_fams = [s.strip() for s in seen.split(",") if s.strip()] if seen else (
        (rec.get("reproducibility") or {}).get("seen_families") or [])
    rows = metrics.get("predictions") or []
    if not rows:
        stored = metrics.get("generators")
        if stored:
            return {"experiment_id": experiment_id, **stored}
        return {"experiment_id": experiment_id, "status": "NOT_MEASURED",
                "note": "no prediction rows stored for this experiment"}
    out = evaluate_generators(rows, seen_fams)
    out["experiment_id"] = experiment_id
    return out


@router.get("/research/{analysis_id}")
def research_view(analysis_id: str):
    """Research mode: full raw report + model/config provenance, unredacted."""
    rep = storage.get_analysis(analysis_id)
    if not rep:
        raise HTTPException(404, "Analysis not found. Save it to history first (save=true).")
    ens = rep["ensemble"]
    return JSONResponse({
        "analysis_id": analysis_id,
        "report_hash": rep.get("report_hash"),
        "reproducibility": rep.get("reproducibility"),
        "explanations": rep.get("explanations"),
        "stages": rep.get("stages"),
        "signals": rep.get("signals"),       # raw analyzer scores, timings, technical
        "ensemble": ens,                      # includes detectors[], hybrid weights
        "limitations": rep.get("limitations"),
        "note": "Raw research view. Scores are model/signal outputs, not ground truth.",
    })


@router.post("/explain", dependencies=[Depends(require_trusted_browser_origin)])
async def explain(request: Request, file: UploadFile = File(...),
                  level: str = Query("basic", pattern="^(basic|technical|forensic)$")):
    """Deterministic explanation for an uploaded image + LLM layer status.

    The LLM explanation layer is a separate, optional component. When not
    configured the endpoint returns 501-style status explicitly - the
    deterministic explanation is always available.
    """
    from ..llm.interface import get_explainer
    _check_rate(request)
    name, mime, data = await _read_upload(file)
    try:
        rep = pipeline.run_analysis(name, mime, data, include_visuals=False)
    except HTTPException:
        raise
    except Exception as exc:
        raise _friendly_error(exc)
    explainer = get_explainer()
    llm_status = {"provider": explainer.get_provider_name(), "configured": False}
    return JSONResponse({
        "level": level,
        "deterministic_explanation": rep["explanations"][level],
        "llm": {**llm_status,
                "note": ("LLM explanation layer is not configured. The deterministic "
                         "explanation above is generated from computed evidence only; "
                         "no language model scored this image.")},
    })


def _is_within_directory(base: str, candidate: str) -> bool:
    """Canonical path containment (Stage 3, S3-2).

    Both sides are resolved with realpath (symlinks resolved) and must
    share the base as their common path. A plain prefix check is not
    enough: it misses symlink escapes and prefix collisions.
    """
    try:
        resolved_base = os.path.realpath(base)
        return os.path.commonpath([resolved_base, os.path.realpath(candidate)]) == resolved_base
    except ValueError:
        return False


def _safe_dataset_dir(name: str) -> str:
    base = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmark", "datasets"))
    path = os.path.realpath(os.path.join(base, name))
    if not _is_within_directory(base, path) or not os.path.isdir(path):
        raise HTTPException(400, "Unknown dataset. Use a directory inside benchmark/datasets/.")
    return path


@router.post("/benchmark/run", dependencies=[Depends(require_loopback), Depends(require_trusted_browser_origin)])
def benchmark_run(request: Request, dataset: str = Query(..., description="dataset dir name inside benchmark/datasets/"),
                  version: str = Query("1.0")):
    """Run the benchmark framework over a labelled dataset (operator action)."""
    _check_mutation_rate(request, "expensive")
    import os
    from ..benchmark.runner import run_benchmark
    ds = _safe_dataset_dir(dataset)
    result = run_benchmark(ds, pipeline.run_analysis, dataset_version=version)
    out = os.path.join(ds, "results.json")
    import json as _json
    with open(out, "w") as fh:
        _json.dump(result, fh, indent=2)
    return JSONResponse(result)


@router.get("/benchmark/results")
def benchmark_results(dataset: str = Query(...)):
    import json as _json
    import os
    ds = _safe_dataset_dir(dataset)
    rp = os.path.join(ds, "results.json")
    if not os.path.isfile(rp):
        raise HTTPException(404, "No benchmark results for this dataset yet. Run /api/benchmark/run first.")
    with open(rp) as fh:
        return JSONResponse(_json.load(fh))


@router.post("/robustness", dependencies=[Depends(require_trusted_browser_origin)])
async def robustness(request: Request, file: UploadFile = File(...)):
    """Robustness test: how do ordinary transformations change the prediction?"""
    from ..benchmark.robustness import evaluate_robustness
    _check_rate(request)
    name, mime, data = await _read_upload(file)
    try:
        result = evaluate_robustness(pipeline.run_analysis, data, name)
    except HTTPException:
        raise
    except Exception as exc:
        raise _friendly_error(exc)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# V5: blind evaluation protocol + experiment lifecycle + model promotion
# ---------------------------------------------------------------------------

@router.post("/experiments/blind", dependencies=[Depends(require_loopback), Depends(require_trusted_browser_origin)])
async def create_blind_experiment_api(request: Request):
    """Create a blind experiment: freeze dataset, model, config, thresholds."""
    _check_mutation_rate(request, "mutation")
    from ..validation.blind_protocol import create_blind_experiment
    try:
        body = await _read_json_limited(request)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Expected JSON body.")
    dataset_id = body.get("dataset_id")
    model_id = body.get("model_id")
    if not dataset_id or not model_id:
        raise HTTPException(400, "dataset_id and model_id are required.")
    try:
        rec = create_blind_experiment(
            dataset_id, model_id,
            seed=int(body.get("seed", 0)),
            note=str(body.get("note", "")))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    return JSONResponse({
        "experiment_id": rec["experiment_id"],
        "status": rec["status"],
        "frozen": {
            "dataset_hash": rec["reproducibility"]["dataset_hash"],
            "model_version": rec["reproducibility"]["model_version"],
            "model_hash": rec["reproducibility"]["model_hash"],
            "configuration_hash": rec["reproducibility"]["configuration_hash"],
            "calibration": rec["reproducibility"].get("frozen_calibration"),
        },
    })


@router.post("/experiments/{experiment_id}/run", dependencies=[Depends(require_loopback), Depends(require_trusted_browser_origin)])
async def run_blind_experiment_api(experiment_id: str, request: Request,
                                   files: list[UploadFile] = File(...)):
    """Run blind inference on uploaded images (no ground truth).

    Stage 3 remediation (F1): uploads are capped like batch uploads
    (20 files, 25 MB per file via ``_read_upload``, 100 MB aggregate
    decoded bytes) and are ALL read and validated before any blind
    inference runs, so a limit breach rejects the whole run with no
    inference and no experiment-state change. The finally block
    closes every uploaded file, including ones never read after a
    breach. Transport is capped at 101 MB by the body-limit
    middleware's experiment-run pattern.
    """
    from ..validation.blind_protocol import run_blind_inference
    _check_rate(request)
    if len(files) > RUN_MAX_FILES:
        raise HTTPException(400, f"Blind experiment runs are limited to {RUN_MAX_FILES} images.")
    images = []
    total = 0
    try:
        for f in files:
            name, mime, data = await _read_upload(f)
            total += len(data)
            if total > BATCH_MAX_BYTES:
                raise HTTPException(413, "Blind run total exceeds 100 MB.")
            images.append({"image_id": name, "filename": name,
                           "mime_type": mime, "data": data})
    finally:
        for f in files:
            await f.close()
    try:
        rec = run_blind_inference(experiment_id, images)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    except Exception as exc:
        raise _friendly_error(exc)
    return JSONResponse({
        "experiment_id": rec["experiment_id"],
        "status": rec["status"],
        "n_predictions": rec.get("n_predictions", 0),
        "predictions": [
            {k: p[k] for k in ("image_id", "verdict", "ai_probability",
                               "confidence", "ood_status", "case_id")
             if k in p}
            for p in rec.get("predictions", [])
        ],
    })


@router.post("/experiments/{experiment_id}/reveal", dependencies=[Depends(require_loopback), Depends(require_trusted_browser_origin)])
async def reveal_blind_experiment_api(experiment_id: str, request: Request):
    """Reveal ground truth and score. After reveal, retuning is blocked."""
    _check_mutation_rate(request, "mutation")
    from ..validation.blind_protocol import reveal_and_score
    try:
        body = await _read_json_limited(request)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Expected JSON body.")
    ground_truth = body.get("ground_truth")
    if not isinstance(ground_truth, dict) or not ground_truth:
        raise HTTPException(400, "ground_truth dict {image_id: label} required.")
    try:
        rec = reveal_and_score(experiment_id, ground_truth)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return JSONResponse({
        "experiment_id": rec["experiment_id"],
        "status": rec["status"],
        "revealed": rec.get("revealed"),
        "metrics": rec.get("metrics", {}),
        "leakage_report": rec.get("leakage_report"),
        "results_hash": rec.get("results_hash"),
    })


@router.post("/experiments/{experiment_id}/invalidate", dependencies=[Depends(require_loopback), Depends(require_trusted_browser_origin)])
async def invalidate_experiment_api(experiment_id: str, request: Request):
    """Mark an experiment INVALIDATED with an explicit reason."""
    _check_mutation_rate(request, "mutation")
    from ..validation.blind_protocol import invalidate_experiment
    try:
        body = await _read_json_limited(request)
    except HTTPException:
        raise
    except Exception:
        body = {}
    reason = (body or {}).get("reason", "")
    if not reason:
        raise HTTPException(400, "A reason is required to invalidate.")
    try:
        rec = invalidate_experiment(experiment_id, reason)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return JSONResponse({"experiment_id": rec["experiment_id"],
                         "status": rec["status"]})


@router.get("/experiments/{experiment_id}/status-history")
def experiment_status_history_api(experiment_id: str):
    """Full lifecycle status history with timestamps."""
    from ..validation import get_status_history
    try:
        return {"experiment_id": experiment_id,
                "history": get_status_history(experiment_id)}
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@router.get("/models/{model_id}/promotion")
def model_promotion_status_api(model_id: str):
    """Promotion status: current level + what's missing for the next level."""
    from ..models.promotion import get_promotion_status
    try:
        return get_promotion_status(model_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@router.post("/models/{model_id}/promote", dependencies=[Depends(require_loopback), Depends(require_trusted_browser_origin)])
async def promote_model_api(model_id: str, request: Request):
    """Attempt promotion. Denied unless evidence requirements are met."""
    _check_mutation_rate(request, "mutation")
    from ..models.promotion import promote
    try:
        body = await _read_json_limited(request)
    except HTTPException:
        raise
    except Exception:
        body = {}
    target = (body or {}).get("target_status", "")
    evidence = (body or {}).get("evidence", {})
    if not target:
        raise HTTPException(400, "target_status is required.")
    try:
        return promote(model_id, target, evidence)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except (PermissionError, ValueError) as exc:
        raise HTTPException(403, str(exc))
