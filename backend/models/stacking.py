"""Forensic stacking detector (V3).

A REAL trained model: L2-regularized logistic regression on forensic analyzer
outputs, trained on synthetic procedural data (see ml/datasets/DATASET.md).

Honest labeling is part of the contract: this is a DEMONSTRATION model.
It is registered with full metadata, calibrated probabilities, and a model
card. It joins the ensemble as one signal — never the verdict.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..analyzers.base import (AIDetectorProvider, AIDetectorResult,
                              AnalysisContext, ModelStatus)
from ..utils.security import sha256_file


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class ForensicStackingDetector(AIDetectorProvider):
    """AIDetectorProvider backed by tl-forensic-stacking weights (JSON)."""

    model_type = "stacking-logreg"

    def __init__(self, name: str = "tl-forensic-stacking",
                 version: str = "1.0.0",
                 model_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        self._name = name
        self._version = version
        self.model_path = model_path
        self.config = config or {}
        self._status = ModelStatus.NOT_CONFIGURED
        self._detail = "not loaded"
        self._model: Optional[Dict[str, Any]] = None
        self._sha256: Optional[str] = None

    # -- provider contract ---------------------------------------------
    def get_model_name(self) -> str:
        return self._name

    def get_version(self) -> str:
        return self._version

    def get_model_type(self) -> str:
        return self.model_type

    def get_status(self) -> ModelStatus:
        return self._status

    def get_supported_formats(self) -> List[str]:
        return ["jpeg", "png", "webp", "bmp"]

    def get_preprocessing_info(self) -> Dict[str, Any]:
        return {"method": "forensic feature stacking",
                "features": (self._model or {}).get("feature_names", []),
                "note": "No pixel preprocessing; features come from analyzer outputs."}

    def get_limitations(self) -> str:
        # Exam-day fix (2026-10-29): the honest_label in
        # models/tl-stacking-v1.json already ends with "not validated
        # on real-world imagery", so appending that phrase again
        # duplicated it in GET /api/models output. Append only the
        # sentence that is not already present, with correct
        # punctuation for both the labelled and fallback cases.
        base = ((self._model or {}).get("honest_label")
                or "Demonstration model trained on synthetic data.")
        if base.endswith("."):
            return base + " One ensemble signal only."
        return base + ". One ensemble signal only."

    # -- lifecycle -------------------------------------------------------
    def load(self) -> None:
        try:
            if not self.model_path or not os.path.isfile(self.model_path):
                self._status = ModelStatus.NOT_CONFIGURED
                self._detail = f"model file not found: {self.model_path}"
                return
            with open(self.model_path, "r") as fh:
                self._model = json.load(fh)
            self._sha256 = sha256_file(self.model_path)
            # sanity: weights match the model's own feature list
            if len(self._model["weights"]) != len(self._model.get("feature_names", [])):
                raise ValueError("weight count does not match model feature list")
            self._status = ModelStatus.READY
            self._detail = "ready"
        except Exception as exc:
            self._status = ModelStatus.ERROR
            self._detail = f"load failed: {type(exc).__name__}: {exc}"

    # -- inference --------------------------------------------------------
    def detect(self, ctx: AnalysisContext) -> AIDetectorResult:
        if self._status != ModelStatus.READY or not self._model:
            return AIDetectorResult(
                detector_name=self._name, detector_version=self._version,
                model_type=self.model_type, confidence=0.0,
                limitations=self.get_limitations(),
                model_status=self._status,
                detail=f"model not ready ({self._detail})")
        started = time.perf_counter()
        try:
            from ml.preprocessing.features import extract_features, feature_names
            signals = ctx.extra.get("analyzer_results") or []
            if not signals:
                raise RuntimeError("analyzer results not available on context")
            full = np.array(extract_features(signals), dtype=np.float64)
            # apply the training-time variance-threshold mask
            kept = self._model.get("feature_names") or []
            if kept and len(kept) != len(full):
                all_names = feature_names()
                mask = [n in kept for n in all_names]
                feats = full[np.array(mask)]
            else:
                feats = full
            n_exp = len(self._model["weights"])
            if len(feats) != n_exp:
                raise RuntimeError(
                    f"feature count {len(feats)} != trained {n_exp}")
            mu = np.array(self._model["standardization"]["mean"])
            sigma = np.array(self._model["standardization"]["std"])
            # V5.1.8.62: a zero or non-finite standardization parameter
            # would produce inf/NaN z-scores and a meaningless
            # probability. Treat the model as not ready (ERROR) instead
            # of computing with it. Caught by the except below.
            if (mu.shape != sigma.shape
                    or not np.all(np.isfinite(mu))
                    or not np.all(np.isfinite(sigma))
                    or np.any(sigma == 0)):
                raise ValueError("invalid standardization parameters: model not ready")
            # OOD guard: extreme z-scores mean the input is far from the
            # training distribution — report honestly instead of exploding
            z = (feats - mu) / sigma
            max_z = float(np.abs(z).max())
            # V4: 4-level OOD status (spec §16)
            if max_z <= 3.0:
                ood_status = "IN_DISTRIBUTION"
            elif max_z <= 6.0:
                ood_status = "MILD_SHIFT"
            else:
                ood_status = "OUT_OF_DISTRIBUTION"
            xs = z
            logit = float(xs @ np.array(self._model["weights"]) + self._model["bias"])
            raw_p = _sigmoid(logit)
            cal = self._model.get("calibration", {})
            T = cal.get("params", {}).get("temperature", 1.0)
            cal_p = _sigmoid(logit / T)
            cal_desc = (f"{cal.get('method', 'none')}"
                        f"(T={cal.get('params', {}).get('temperature', 'n/a')})"
                        if cal.get("method") else "none (uncalibrated)")
            elapsed = (time.perf_counter() - started) * 1000
            # confidence reflects OOD distance: far from training distribution
            # means the calibrated number is less trustworthy
            ood = ood_status == "OUT_OF_DISTRIBUTION"
            conf = 40.0 if ood else (55.0 if ood_status == "MILD_SHIFT" else 70.0)
            demo = bool(self.config.get("demonstration", True))  # default True: safe
            ev = (f"Stacking model logit {logit:.3f} -> raw P(AI) {raw_p * 100:.1f}% "
                  f"-> calibrated P(AI) {cal_p * 100:.1f}%")
            if ood:
                ev += (f" [note: input features far from training distribution "
                       f"(max |z|={max_z:.1f}); treat probability with extra caution]")
            return AIDetectorResult(
                detector_name=self._name,
                detector_version=self._version,
                model_type=self.model_type,
                probability_ai=round(cal_p * 100, 1),
                probability_authentic=round((1 - cal_p) * 100, 1),
                raw_score=round(raw_p, 4),
                calibration=cal_desc,
                confidence=conf,  # reduced when input is out-of-distribution
                preprocessing={
                    "method": "forensic feature stacking",
                    "n_features": len(feats),
                    "standardization": "z-score (train split)",
                    "model_sha256": (self._sha256 or "")[:16],
                    "max_abs_z": round(max_z, 2),
                    "ood_status": ood_status,
                },
                evidence=[ev],
                limitations=self.get_limitations(),
                inference_time_ms=elapsed,
                model_status=ModelStatus.READY,
                demonstration=demo,
                detail=f"weights: {self._name} v{self._version}; {cal_desc}",
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            return AIDetectorResult(
                detector_name=self._name, detector_version=self._version,
                model_type=self.model_type, confidence=0.0,
                limitations=self.get_limitations(),
                inference_time_ms=elapsed,
                model_status=ModelStatus.ERROR,
                detail=f"inference failed: {type(exc).__name__}: {str(exc)[:200]}")
