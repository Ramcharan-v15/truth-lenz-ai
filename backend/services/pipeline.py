"""Analysis pipeline: decode -> context -> analyzers (in forensic stage
order) -> ML detectors -> ensemble. Each analyzer runs in isolation; one
failure never kills the pipeline."""
from __future__ import annotations

import hashlib
import math
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from ..analyzers.ai_artifacts import AIArtifactAnalyzer
from ..analyzers.base import (AIDetectorProvider, AnalysisContext,
                              AnalyzerResult, NullDetectorProvider)
from ..analyzers.compression import CompressionAnalyzer
from ..analyzers.consistency import ConsistencyAnalyzer
from ..analyzers.ensemble import EnsembleEngine, VERDICT_EXPLANATIONS
from ..analyzers.frequency import FrequencyAnalyzer
from ..analyzers.manipulation import ManipulationAnalyzer
from ..analyzers.metadata import MetadataAnalyzer
from ..analyzers.noise import NoiseAnalyzer
from ..analyzers.pixel_forensics import PixelForensicsAnalyzer
from ..analyzers.quality import QualityAnalyzer
from ..analyzers.statistics import StatisticsAnalyzer
from ..models.registry import get_global_registry
from ..utils.image_io import decode_image, load_config, png_data_uri, to_float_arrays
from .explain import build_explanations

APP_VERSION = "2.0.0"
BACKEND_VERSION = "2.0.0"

# case file numbering: TL-YYYY-NNNNNN, sequential per day, persisted across restarts
_CASE_COUNTER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "case_counter.json")


def _dhash(pil_image, hash_size: int = 8) -> str:
    """Difference hash (dHash) for perceptual duplicate detection."""
    import numpy as np
    img = pil_image.convert("L").resize((hash_size + 1, hash_size), resample=2)
    px = np.asarray(img, dtype=np.int16)
    diff = px[:, 1:] > px[:, :-1]
    return "".join("1" if b else "0" for b in diff.flatten())


def _next_case_id() -> str:
    """Allocate the next forensic case number (TL-2026-000001 style)."""
    import datetime as _dt
    import json as _json
    today = _dt.date.today().isoformat()
    os.makedirs(os.path.dirname(_CASE_COUNTER_PATH), exist_ok=True)
    try:
        with open(_CASE_COUNTER_PATH, "r") as fh:
            data = _json.load(fh)
    except Exception:
        data = {}
    count = int(data.get(today, 0)) + 1
    data[today] = count
    try:
        with open(_CASE_COUNTER_PATH, "w") as fh:
            _json.dump(data, fh)
    except Exception:
        pass
    return f"TL-{today[:4]}-{count:06d}"

# stage order mirrors the visible analysis pipeline in the UI
ANALYZERS = [
    MetadataAnalyzer(),
    StatisticsAnalyzer(),
    CompressionAnalyzer(),
    NoiseAnalyzer(),
    FrequencyAnalyzer(),
    PixelForensicsAnalyzer(),
    AIArtifactAnalyzer(),
    ManipulationAnalyzer(),
    QualityAnalyzer(),
    ConsistencyAnalyzer(),
]

STAGE_IDS = ["metadata", "statistics", "compression", "noise", "frequency",
             "pixel", "ai_artifacts", "manipulation", "quality", "consistency"]

# lazily-built global registry (models/registry.json)
_REGISTRY = None


def get_registry():
    global _REGISTRY
    if _REGISTRY is None:
        try:
            _REGISTRY = get_global_registry()
        except Exception:
            _REGISTRY = None
    return _REGISTRY


def _find_nonfinite(obj: Any, path: str = "$") -> List[str]:
    """Paths of every non-finite float (NaN or +/-inf) in a structure."""
    out: List[str] = []
    if isinstance(obj, float) and not math.isfinite(obj):
        out.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_find_nonfinite(v, f"{path}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.extend(_find_nonfinite(v, f"{path}[{i}]"))
    return out


def _sanitize_diagnostics(container: Any, path: str = "$") -> List[str]:
    """Replace non-finite floats with None in place; return their paths.

    Used for diagnostic-only fields (analyzer technical/visual payloads).
    Null means "measurement unavailable" - it never invents a number.
    """
    bad: List[str] = []
    if isinstance(container, dict):
        for k, v in list(container.items()):
            p = f"{path}.{k}"
            if isinstance(v, float) and not math.isfinite(v):
                container[k] = None
                bad.append(p)
            else:
                bad.extend(_sanitize_diagnostics(v, p))
    elif isinstance(container, list):
        for i, v in enumerate(container):
            p = f"{path}[{i}]"
            if isinstance(v, float) and not math.isfinite(v):
                container[i] = None
                bad.append(p)
            else:
                bad.extend(_sanitize_diagnostics(v, p))
    return bad


def _fallback_verdict_block(config: Dict[str, Any], results: List[AnalyzerResult],
                            failure: str, nonfinite_fields: List[str]) -> Dict[str, Any]:
    """The honest verdict-block representation when no valid verdict exists.

    Verdict is INCONCLUSIVE, probabilities are null (unknown - never a
    fabricated 50 or 0), and the reason names what went wrong. This is
    the single representation used both when ensemble.combine raises
    and when the finite-value boundary finds an invalid verdict-feeding
    value, so downstream logic can never retain a confident verdict
    built on invalid numbers.
    """
    quality = next((r for r in results if r.name == "quality"), None)
    insufficient = False
    if quality is not None:
        try:
            pixels = float(quality.technical.get("width", 0) or 0) * float(
                quality.technical.get("height", 1) or 1)
        except (TypeError, ValueError):
            pixels = 0
        insufficient = pixels < config.get("quality", {}).get("min_analysis_pixels", 16384)
    if nonfinite_fields:
        reason = ("Verdict withheld: verdict-feeding value(s) were non-finite "
                  f"({', '.join(nonfinite_fields)}). No confident verdict can "
                  "be supported from invalid numeric values.")
    else:
        reason = (f"Verdict computation failed ({failure}); no valid verdict "
                  "could be produced. No confident verdict is reported.")
    return {
        "verdict": "INCONCLUSIVE",
        "verdict_explanation": VERDICT_EXPLANATIONS.get("INCONCLUSIVE", ""),
        "verdict_reason": reason,
        "ai_probability": None,
        "manipulation_probability": None,
        "authenticity_probability": None,
        "confidence": 0.0,
        "confidence_label": "LOW",
        "confidence_penalties": [],
        "signal_disagreement": None,
        "signals_agree": False,
        "insufficient_evidence": insufficient,
        "probability_bands": config.get("bands", {}).get("ai_probability_bands", []),
        "why": [],
        "strongest_evidence": [],
        "evidence_against": [],
        "uncertain_factors": [],
        "detector": {"detector_name": "No trained model configured", "detector_version": "n/a",
                     "model_type": "none", "probability_ai": None, "confidence": 0.0,
                     "model_status": "not_configured", "available": False,
                     "limitations": "No trained AI-detection model is connected."},
        "detectors": [],
        "ml_available": False,
        "ml_ai_probability": None,
        "forensic_ai_probability": None,
        "provenance": {"ai_score": None, "confidence": None,
                       "note": "Verdict computation failed; provenance unavailable."},
        "ml_disagreement": None,
        "ml_ood_status": "UNKNOWN",
        "uncertainty": None,
        "false_positive_risk": None,
        "hybrid_weights": {"forensics": 0.0, "ml": 0.0, "provenance": 0.0},
        "weights_used": {},
        "thresholds_used": {},
        "finite_validation": {"nonfinite_fields": nonfinite_fields,
                              "failure": failure,
                              "action": "verdict block replaced with INCONCLUSIVE fallback"},
        "disclaimer": ("Probability estimates, not definitive proof. No detector can guarantee "
                       "whether an image is AI-generated or authentic."),
    }


def _apply_finite_boundary(report: Dict[str, Any], results: List[AnalyzerResult],
                           config: Dict[str, Any]) -> Dict[str, Any]:
    """Final finite-value gate before a report is returned (V5.1.8.62).

    - Diagnostic-only non-finite values (signal technical/visuals) become
      null with a recorded warning on that signal.
    - Any non-finite value in the verdict block replaces the whole block
      with the INCONCLUSIVE fallback (probabilities null, reason explicit),
      and the explanations are replaced with a matching minimal set so no
      stale confident explanation can survive.
    - Stack traces are stripped from signal technical payloads: failure
      reasons are preserved, internals are not exposed to API clients.
    Must run before the v4 schema and report hash are built.
    """
    for s in report.get("signals", []):
        for key in ("ai_score", "manip_score", "confidence"):
            v = s.get(key)
            if isinstance(v, float) and not math.isfinite(v):
                # Unreachable via BaseAnalyzer.run (clamp raises -> UNABLE),
                # kept as defense in depth, following the existing UNABLE
                # convention (50/50/0 = no information).
                s["status"] = "unable"
                s["ai_score"] = 50.0
                s["manip_score"] = 50.0
                s["confidence"] = 0.0
                s.setdefault("warnings", []).append(
                    "Signal scores contained a non-finite value; signal marked unable.")
                break
        tech = s.get("technical")
        if isinstance(tech, dict):
            tech.pop("traceback", None)
            bad = _sanitize_diagnostics(tech, f"signals.{s.get('name')}.technical")
            if bad:
                s.setdefault("warnings", []).append(
                    "Non-finite diagnostic value(s) recorded as unavailable (null): "
                    + ", ".join(bad))
        vis = s.get("visuals")
        if isinstance(vis, dict):
            bad = _sanitize_diagnostics(vis, f"signals.{s.get('name')}.visuals")
            if bad:
                s.setdefault("warnings", []).append(
                    "Non-finite visual value(s) recorded as unavailable (null): "
                    + ", ".join(bad))
    ens = report.get("ensemble", {})
    bad = _find_nonfinite(ens)
    if bad:
        report["ensemble"] = _fallback_verdict_block(config, results, "non-finite value", bad)
        report["explanations"] = {
            "basic": ("Verdict withheld: verdict-feeding values were non-finite, "
                      "so no confident verdict can be supported."),
            "technical": ("The ensemble verdict block contained non-finite numeric "
                          f"value(s) ({', '.join(bad)}); probabilities are reported "
                          "as unknown (null), not estimated."),
            "forensic": "No confident verdict can be supported from invalid numeric values.",
            "note": ("Generated deterministically from computed evidence. "
                     "No language model was involved in scoring."),
        }
    return report


def _safe_explanations(verdict_block: Dict[str, Any], results: List[AnalyzerResult]) -> Dict[str, str]:
    """Build explanations, falling back to a withheld-verdict text.

    Needed because the explanation builders format probabilities
    directly: a fallback verdict block (probabilities null) would
    otherwise crash report construction before the finite boundary
    runs. The boundary overwrites this with a specific reason when
    it fires; this wrapper only guarantees construction never fails.
    """
    try:
        return build_explanations(verdict_block, results)
    except Exception:
        return {
            "basic": "Verdict withheld: no valid verdict could be produced.",
            "technical": "Verdict computation failed; probabilities are unknown (null), not estimated.",
            "forensic": "No confident verdict can be supported from invalid numeric values.",
            "note": ("Generated deterministically from computed evidence. "
                     "No language model was involved in scoring."),
        }


def build_context(filename: str, mime_type: str, data: bytes) -> AnalysisContext:
    pil_image, info = decode_image(data)
    rgb, gray, small_gray = to_float_arrays(pil_image)
    ctx = AnalysisContext(
        image_id=uuid.uuid4().hex[:12],
        filename=filename,
        mime_type=mime_type,
        file_size=len(data),
        file_bytes=data,
        pil_image=pil_image,
        gray=gray,
        rgb=rgb,
        small_gray=small_gray,
        extra={"decode_info": info, "downscaled": info.get("downscaled", False),
               "has_alpha": info.get("has_alpha", False)},
    )
    return ctx


def _config_hash(config: dict) -> str:
    import json
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def run_analysis(filename: str, mime_type: str, data: bytes,
                 detector: AIDetectorProvider | None = None,
                 detectors: Optional[List[AIDetectorProvider]] = None,
                 config: dict | None = None,
                 include_visuals: bool = True) -> Dict[str, Any]:
    """Run the full forensic pipeline. Returns a JSON-serializable report dict."""
    config = config or load_config()
    started = time.perf_counter()
    ctx = build_context(filename, mime_type, data)
    image_sha256 = hashlib.sha256(data).hexdigest()

    results: List[AnalyzerResult] = []
    stage_timings: List[Dict[str, Any]] = []
    for analyzer in ANALYZERS:
        r = analyzer.run(ctx)
        results.append(r)
        stage_timings.append({"stage": r.name, "display": r.display_name,
                              "status": r.status.value,
                              "duration_ms": round(r.duration_ms, 1)})

    # detectors: explicit list wins; else single detector; else registry
    if detectors is None:
        if detector is not None:
            detectors = [detector]
        else:
            reg = get_registry()
            detectors = reg.get_ready_detectors() if reg else []
            if not detectors:
                detectors = [NullDetectorProvider()]

    ensemble = EnsembleEngine(config, detectors=detectors)
    # stash analyzer outputs on ctx so feature-based detectors (stacking model)
    # can consume them without re-running the analyzers
    ctx.extra["analyzer_results"] = [r.to_dict() for r in results]
    try:
        verdict_block = ensemble.combine(ctx, results)
    except Exception as exc:
        # V5.1.8.62: a verdict computation that fails (e.g. a non-finite
        # value reaching the ensemble) must never become a crash or a
        # confident verdict. Withhold the verdict explicitly.
        verdict_block = _fallback_verdict_block(config, results, type(exc).__name__, [])

    total_ms = (time.perf_counter() - started) * 1000

    # preview thumbnail
    thumb = ctx.pil_image.copy()
    thumb.thumbnail((480, 480))
    preview_uri = png_data_uri(thumb)

    w, h = ctx.pil_image.size
    det_infos = verdict_block.get("detectors", [])
    case_id = _next_case_id()
    # --- generator attribution (V4, spec §15): separate from AI detection ---
    # No model is trained/evaluated for attribution → ATTRIBUTION_UNAVAILABLE.
    # Never infer "AI detected" => "Midjourney".
    generator_attribution = {
        "status": "ATTRIBUTION_UNAVAILABLE",
        "label": None,
        "confidence": 0.0,
        "detail": ("No generator-attribution model is registered. AI detection "
                   "does not imply any specific generator."),
    }

    report = {
        "case_id": case_id,
        "analysis_id": ctx.image_id,
        "generator_attribution": generator_attribution,
        "filename": filename,
        "mime_type": mime_type,
        "file_size": len(data),
        "width": w,
        "height": h,
        "preview": preview_uri,
        "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline_ms": round(total_ms, 1),
        "stages": stage_timings,
        "signals": [r.to_dict() for r in results],
        "ensemble": verdict_block,
        "educational": _educational_notes(),
        "explanations": _safe_explanations(verdict_block, results),
        "reproducibility": {
            "case_id": case_id,
            "analysis_id": ctx.image_id,
            "app_version": APP_VERSION,
            "backend_version": BACKEND_VERSION,
            "config_version": config.get("version", "unknown"),
            "config_sha256": _config_hash(config),
            "detectors": [
                {"name": d.get("detector_name"), "version": d.get("detector_version"),
                 "model_type": d.get("model_type"), "status": d.get("model_status"),
                 "calibration": d.get("calibration")}
                for d in det_infos
            ],
            "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "image_sha256": image_sha256,
            "perceptual_hash": _dhash(ctx.pil_image),
            "image_dimensions": [w, h],
            "pipeline_ms": round(total_ms, 1),
            "note": "Original image bytes are not stored. Re-running this image through "
                    "the same app/backend/config/detector versions reproduces this report.",
        },
        "limitations": [
            "No detector can guarantee whether an image is AI-generated or authentic.",
            "Compression, resizing, screenshots and metadata stripping degrade forensic signals.",
            "Absence of metadata is not evidence of AI generation; metadata can be fabricated.",
            "ELA highlights potentially suspicious regions; it does not prove manipulation.",
            "Heuristic AI-artifact signals are not a trained model.",
            "Model outputs (where present) are calibrated estimates, not certainties.",
            "Do not use this analysis as the sole basis for legal, academic, employment, "
            "financial or other high-stakes decisions.",
        ],
    }
    # --- V5.1.8.62 finite-value boundary: must run BEFORE the v4 schema
    # (which copies ensemble/signals) and the report hash are built.
    report = _apply_finite_boundary(report, results, config)

    # --- V4 result schema (spec §39): strict separation ---
    v4_schema = {
        "observations": [
            {"analyzer": s["name"], "ai_score": s["ai_score"],
             "manip_score": s["manip_score"], "confidence": s["confidence"],
             "status": s["status"],
             "evidence": [{"text": e["text"], "direction": e["direction"],
                           "detail": e.get("detail", "")} for e in s.get("evidence", [])]}
            for s in report["signals"]
        ],
        "model_inference": report["ensemble"].get("detectors", []),
        "provenance": {
            "c2pa": report["signals"][0].get("technical", {}).get("c2pa_status")
                    if report["signals"] else "unknown",
            "ensemble_provenance": report["ensemble"].get("provenance", {}),
        },
        "uncertainty": {
            "breakdown": report["ensemble"].get("uncertainty"),
            "ood_status": report["ensemble"].get("ml_ood_status", "UNKNOWN"),
            "false_positive_risk": report["ensemble"].get("false_positive_risk"),
            "confidence": report["ensemble"].get("confidence"),
            "confidence_label": report["ensemble"].get("confidence_label"),
        },
        "ensemble": report["ensemble"],
        "interpretation": {
            "verdict": report["ensemble"].get("verdict"),
            "verdict_reason": report["ensemble"].get("verdict_reason"),
            "explanation": report.get("explanation", {}),
        },
        "limitations": report.get("limitations", []),
    }
    report["v4"] = v4_schema

    # report hash: SHA-256 over canonical JSON excluding the hash itself.
    # Computed LAST so it covers every key (including v4 schema).
    import json as _json
    canonical = _json.dumps({k: v for k, v in report.items() if k != "report_hash"},
                            sort_keys=True, separators=(",", ":"))
    report["report_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    if not include_visuals:
        report["preview"] = ""
        for s in report["signals"]:
            s["visuals"] = {}

    return report


def _educational_notes() -> Dict[str, Dict[str, str]]:
    return {
        "metadata": {
            "what": "Metadata is data embedded in the file: camera model, timestamps, software tags.",
            "why": "It gives provenance context - which camera or software touched the file.",
            "limits": "It can be removed, stripped by apps, or completely fabricated.",
        },
        "noise": {
            "what": "Every camera sensor adds a subtle, consistent noise pattern to photos.",
            "why": "Synthetic images often lack this texture or show unnaturally uniform noise.",
            "limits": "Denoising, compression and low light change noise dramatically.",
        },
        "frequency": {
            "what": "The 2D Fourier transform reveals repeating patterns invisible to the eye.",
            "why": "Some generators leave periodic fingerprints from their upsampling steps.",
            "limits": "Newer generators are cleaner; regular real textures can mimic peaks.",
        },
        "ela": {
            "what": "Error Level Analysis re-saves the image and amplifies compression differences.",
            "why": "Regions edited separately often respond differently than their surroundings.",
            "limits": "High-contrast edges light up innocently; ELA never proves manipulation.",
        },
        "compression": {
            "what": "JPEG quantization tables describe how aggressively the image was compressed.",
            "why": "Non-standard tables hint the image was re-saved, as happens after editing.",
            "limits": "Many legitimate pipelines re-encode images (messaging apps, CDNs).",
        },
    }
