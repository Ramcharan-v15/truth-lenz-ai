"""
Truth Lenz AI - Analyzer framework.

Every analyzer returns a structured AnalyzerResult. Scores are always on a
0-100 scale and are PROBABILISTIC SIGNALS, never proof:

  * ai_score    - how strongly this signal suggests AI generation
  * manip_score - how strongly this signal suggests manipulation/editing
  * confidence  - how reliable this signal is for THIS image (0-100)

A score of 50 means "no information either way". Scores below 50 lean
toward authentic/camera-capture; above 50 lean toward synthetic/manipulated.
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SignalStatus(str, Enum):
    OK = "ok"            # analyzer ran successfully
    WARNING = "warning"  # ran, but with caveats
    UNABLE = "unable"    # could not run on this image


class EvidenceDirection(str, Enum):
    SUPPORTS_AI = "supports_ai"                # evidence consistent with AI generation
    SUPPORTS_AUTHENTIC = "supports_authentic"  # evidence consistent with camera capture
    SUPPORTS_MANIP = "supports_manip"          # evidence consistent with manipulation
    NEUTRAL = "neutral"                        # context only
    UNCERTAIN = "uncertain"                    # ambiguous


@dataclass
class Evidence:
    text: str
    direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "direction": self.direction.value, "detail": self.detail}


@dataclass
class AnalyzerResult:
    name: str                    # machine name, e.g. "metadata"
    display_name: str            # human label, e.g. "Metadata Forensics"
    ai_score: float              # 0-100, higher = more AI-like
    manip_score: float           # 0-100, higher = more manipulation-like
    confidence: float            # 0-100, reliability of this signal for this image
    status: SignalStatus = SignalStatus.OK
    evidence: List[Evidence] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    technical: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    # Visual payloads (base64 PNG data URIs or chart data) this analyzer produced
    visuals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "ai_score": round(self.ai_score, 1),
            "manip_score": round(self.manip_score, 1),
            "confidence": round(self.confidence, 1),
            "status": self.status.value,
            "evidence": [e.to_dict() for e in self.evidence],
            "warnings": self.warnings,
            "technical": self.technical,
            "duration_ms": round(self.duration_ms, 1),
            "visuals": self.visuals,
        }


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Bound a score to [lo, hi].

    V5.1.8.62: non-finite inputs are REJECTED, never silently mapped to
    a bound. min()/max() map NaN to the upper bound (all NaN comparisons
    are False), so clamping NaN used to manufacture a maximally
    confident score. Raising here lets the analyzer wrapper record the
    signal as UNABLE instead.
    """
    v = float(v)
    if not math.isfinite(v):
        raise ValueError("clamp: non-finite value cannot be bounded")
    return max(lo, min(hi, v))


class BaseAnalyzer(ABC):
    """All forensic analyzers inherit from this."""

    name: str = "base"
    display_name: str = "Base Analyzer"

    @abstractmethod
    def analyze(self, ctx: "AnalysisContext") -> AnalyzerResult:
        ...

    def run(self, ctx: "AnalysisContext") -> AnalyzerResult:
        started = time.perf_counter()
        try:
            result = self.analyze(ctx)
            # V5.1.8.62: clamping is INSIDE the try block. Previously it
            # ran after the except, so a NaN score reaching here would
            # either crash the pipeline or (before clamp rejected
            # non-finite values) become a maximal score. Now a
            # non-finite score fails this signal cleanly into UNABLE.
            result.ai_score = clamp(result.ai_score)
            result.manip_score = clamp(result.manip_score)
            result.confidence = clamp(result.confidence)
        except Exception as exc:  # never let one analyzer kill the pipeline
            result = AnalyzerResult(
                name=self.name,
                display_name=self.display_name,
                ai_score=50.0,
                manip_score=50.0,
                confidence=0.0,
                status=SignalStatus.UNABLE,
                warnings=[f"Analyzer failed: {type(exc).__name__}"],
                # V5.1.8.62: the failure reason is preserved, but no
                # stack trace is ever attached to a result that reaches
                # API clients.
                technical={"error": str(exc)[:300]},
            )
        result.duration_ms = (time.perf_counter() - started) * 1000
        return result


# ---------------------------------------------------------------------------
# Shared analysis context: the safely-decoded image plus derived artefacts
# that multiple analyzers reuse (grayscale, downscaled copies, EXIF, ...).
# ---------------------------------------------------------------------------
@dataclass
class AnalysisContext:
    image_id: str
    filename: str
    mime_type: str
    file_size: int
    file_bytes: bytes
    pil_image: Any = None          # PIL.Image, RGB
    gray: Any = None               # numpy float32 grayscale 0..1
    rgb: Any = None                # numpy float32 RGB 0..1
    small_gray: Any = None         # downscaled grayscale for expensive ops
    exif: Dict[str, Any] = field(default_factory=dict)
    png_text: Dict[str, str] = field(default_factory=dict)
    icc_present: bool = False
    xmp_present: bool = False
    c2pa_present: bool = False
    c2pa_info: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AI detector provider abstraction (spec section 34/35).
# A trained model is ONE signal, never the whole verdict.
# ---------------------------------------------------------------------------
class DetectorResult:
    def __init__(self, ai_probability: Optional[float], confidence: float,
                 model_name: str, model_version: str, limitations: str,
                 available: bool, detail: str = ""):
        self.ai_probability = ai_probability
        self.confidence = confidence
        self.model_name = model_name
        self.model_version = model_version
        self.limitations = limitations
        self.available = available
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ai_probability": self.ai_probability,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "limitations": self.limitations,
            "available": self.available,
            "detail": self.detail,
        }


class ModelStatus(str, Enum):
    NOT_CONFIGURED = "not_configured"      # no model registered / found
    DEPENDENCY_MISSING = "dependency_missing"  # backend framework not installed
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"                        # registered but failed to load/run
    DISABLED = "disabled"                  # explicitly disabled by operator


@dataclass
class AIDetectorResult:
    """Formal result contract for every ML detector (Phase 1).

    RULE 5: raw model output is NEVER presented as a calibrated probability.
    `probability_ai` must be the calibrated value; `raw_score` is the
    uncalibrated model output. If no calibration was applied, `calibration`
    must say so explicitly and confidence is capped by the ensemble.
    """
    detector_name: str
    detector_version: str
    model_type: str = "unknown"            # e.g. "pytorch", "onnx", "external-api", "heuristic"
    probability_ai: Optional[float] = None        # calibrated P(AI-generated), 0-100
    probability_authentic: Optional[float] = None # calibrated P(camera capture), 0-100
    raw_score: Optional[float] = None             # uncalibrated model output (logit/prob)
    calibration: str = "none"                     # e.g. "temperature_scaling@T=1.8"
    confidence: float = 0.0                       # detector's self-reported reliability
    supported_formats: List[str] = field(default_factory=lambda: ["jpeg", "png", "webp"])
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    limitations: str = ""
    inference_time_ms: float = 0.0
    model_status: ModelStatus = ModelStatus.READY
    attention_heatmap: Optional[str] = None  # base64 PNG data URI (optional)
    generator_label: Optional[str] = None   # optional generator-family label
    detail: str = ""
    demonstration: bool = False  # True = demonstration-grade model; contributes
                                 # probability + uncertainty but has no verdict veto

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "detector_name": self.detector_name,
            "detector_version": self.detector_version,
            "model_type": self.model_type,
            "probability_ai": self.probability_ai,
            "probability_authentic": self.probability_authentic,
            "raw_score": self.raw_score,
            "calibration": self.calibration,
            "confidence": round(self.confidence, 1),
            "supported_formats": self.supported_formats,
            "preprocessing": self.preprocessing,
            "evidence": self.evidence,
            "limitations": self.limitations,
            "inference_time_ms": round(self.inference_time_ms, 1),
            "model_status": self.model_status.value,
            "generator_label": self.generator_label,
            "detail": self.detail,
            "demonstration": self.demonstration,
            "has_attention_heatmap": self.attention_heatmap is not None,
        }
        if self.attention_heatmap:
            d["attention_heatmap"] = self.attention_heatmap
        return d

    @property
    def available(self) -> bool:
        return (self.model_status == ModelStatus.READY
                and self.probability_ai is not None)


class AIDetectorProvider(ABC):
    """Interface for pluggable AI-generation detectors.

    Every ML detector must implement this interface. The ensemble consumes
    AIDetectorResult only - never framework-specific objects.
    """

    @abstractmethod
    def detect(self, ctx: AnalysisContext) -> AIDetectorResult: ...

    @abstractmethod
    def get_model_name(self) -> str: ...

    @abstractmethod
    def get_version(self) -> str: ...

    @abstractmethod
    def get_model_type(self) -> str:
        """Framework/backend kind: 'pytorch' | 'onnx' | 'tensorflow' | 'external-api' | ..."""
        ...

    @abstractmethod
    def get_limitations(self) -> str: ...

    def get_status(self) -> ModelStatus:
        return ModelStatus.READY

    def get_supported_formats(self) -> List[str]:
        return ["jpeg", "png", "webp"]

    def get_preprocessing_info(self) -> Dict[str, Any]:
        return {}


class NullDetectorProvider(AIDetectorProvider):
    """Transparent fallback used when no trained model is configured.

    Clearly labelled as unavailable - never fakes a model result.
    """

    def get_model_name(self) -> str:
        return "No trained model configured"

    def get_version(self) -> str:
        return "n/a"

    def get_model_type(self) -> str:
        return "none"

    def get_status(self) -> ModelStatus:
        return ModelStatus.NOT_CONFIGURED

    def get_limitations(self) -> str:
        return ("No trained AI-detection model is connected. The verdict is "
                "produced purely from local forensic signals. Register a model "
                "in models/registry.json to add it as one ensemble signal.")

    def detect(self, ctx: AnalysisContext) -> AIDetectorResult:
        return AIDetectorResult(
            detector_name=self.get_model_name(),
            detector_version=self.get_version(),
            model_type=self.get_model_type(),
            probability_ai=None,
            probability_authentic=None,
            confidence=0.0,
            limitations=self.get_limitations(),
            model_status=ModelStatus.NOT_CONFIGURED,
            detail="fallback: no model; not used in ensemble",
        )
