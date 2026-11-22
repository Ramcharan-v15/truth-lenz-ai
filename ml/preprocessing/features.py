"""Feature extraction: analyzer outputs -> fixed feature vector.

The stacking model learns from the SAME forensic signals the ensemble uses.
Features are deterministic functions of analyzer outputs — no hidden state.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

# (analyzer_name, technical_key, scale, default)
TECHNICAL_FEATURES = [
    ("noise", "global_noise_std", 0.05, 0.01),
    ("frequency", "spectral_slope", -5.0, -2.0),      # negative slope; scale by /-5
    ("frequency", "spectral_peak_count", 10.0, 0.0),
    ("compression", "estimated_jpeg_quality", 100.0, 90.0),
    ("pixel", "ela_mean", 50.0, 10.0),
    ("statistics", "luminance_entropy_bits", 8.0, 7.0),
    ("statistics", "contrast_rms", 80.0, 40.0),
    ("quality", "megapixels", 4.0, 0.25),
    ("manipulation", "max_ncc", 1.0, 0.0),
    ("ai_artifacts", "duplicate_block_ratio", 1.0, 0.0),
]

ANALYZER_ORDER = ["metadata", "statistics", "compression", "noise", "frequency",
                  "pixel", "ai_artifacts", "manipulation", "quality", "consistency"]


def feature_names() -> List[str]:
    names = []
    for a in ANALYZER_ORDER:
        names += [f"{a}.ai", f"{a}.manip", f"{a}.conf"]
    for a, k, _, _ in TECHNICAL_FEATURES:
        names += [f"{a}.{k}"]
    return names


def extract_features(signals: List[Dict[str, Any]]) -> List[float]:
    """Build the feature vector from analyzer result dicts (as in report['signals'])."""
    by_name = {s["name"]: s for s in signals}
    feats: List[float] = []
    for a in ANALYZER_ORDER:
        s = by_name.get(a, {})
        for key, default in (("ai_score", 50.0), ("manip_score", 50.0), ("confidence", 0.0)):
            # V5.1.8.62: a non-finite score must fail feature extraction
            # (the detector then reports ERROR / unavailable) rather than
            # flowing into the model as a fabricated value.
            v = float(s.get(key, default))
            if not math.isfinite(v):
                raise ValueError(f"non-finite analyzer score for {a}.{key}")
            feats.append(v / 100.0)
    for a, k, scale, default in TECHNICAL_FEATURES:
        tech = by_name.get(a, {}).get("technical", {})
        v = tech.get(k, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = default
        # normalize: divide by scale; spectral_slope is negative so flip sign
        if k == "spectral_slope":
            v = -v / abs(scale)
        else:
            v = v / scale
        # V5.1.8.62: previously max(-3, min(3, NaN)) silently became 3.0
        # (an extreme feature value). A non-finite technical value now
        # fails extraction instead of being clamped to an extreme.
        if not math.isfinite(v):
            raise ValueError(f"non-finite technical feature {a}.{k}")
        feats.append(max(-3.0, min(3.0, v)))
    return feats


def label_for(category: str) -> int:
    """Binary AI-generation label for training categories."""
    return 1 if category in ("ai_generated", "ai_edited") else 0
