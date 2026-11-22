"""Probability calibration layer (Phase 9).

RULE 5: raw model output is NEVER presented as a calibrated probability.
Each calibrator maps raw scores -> calibrated P(AI) in 0..1 and records its
method + parameters. Until calibration is documented, the ensemble caps the
detector's influence and labels it uncalibrated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CalibrationInfo:
    method: str                      # 'temperature_scaling' | 'platt' | 'isotonic' | 'none'
    params: Dict[str, Any]
    fitted_on: str = ""              # description of calibration data
    version: str = "1.0"

    def describe(self) -> str:
        if self.method == "none":
            return "none (uncalibrated)"
        ps = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.method}({ps})"


class BaseCalibrator:
    method = "none"

    def calibrate(self, raw: float) -> float:
        raise NotImplementedError

    def info(self) -> CalibrationInfo:
        return CalibrationInfo(method=self.method, params={})


class IdentityCalibrator(BaseCalibrator):
    """Explicitly uncalibrated: passes the score through, labelled as such."""

    def calibrate(self, raw: float) -> float:
        return float(np.clip(raw, 0.0, 1.0))

    def info(self) -> CalibrationInfo:
        return CalibrationInfo(method="none", params={},
                               fitted_on="n/a - raw scores used directly")


class TemperatureScaling(BaseCalibrator):
    """p_cal = sigmoid(logit / T). Expects raw to be a logit or probability."""
    method = "temperature_scaling"

    def __init__(self, temperature: float = 1.0, from_logits: bool = True):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.from_logits = from_logits

    def calibrate(self, raw: float) -> float:
        x = float(raw)
        if not self.from_logits:
            # convert probability -> logit, guard edges
            x = float(np.clip(x, 1e-6, 1 - 1e-6))
            x = float(np.log(x / (1 - x)))
        p = 1.0 / (1.0 + np.exp(-x / self.temperature))
        return float(np.clip(p, 0.0, 1.0))

    def info(self) -> CalibrationInfo:
        return CalibrationInfo(method=self.method,
                               params={"temperature": self.temperature,
                                       "from_logits": self.from_logits})


class PlattScaling(BaseCalibrator):
    """p_cal = sigmoid(a * raw + b)."""
    method = "platt"

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a, self.b = float(a), float(b)

    def calibrate(self, raw: float) -> float:
        p = 1.0 / (1.0 + np.exp(-(self.a * float(raw) + self.b)))
        return float(np.clip(p, 0.0, 1.0))

    def info(self) -> CalibrationInfo:
        return CalibrationInfo(method=self.method, params={"a": self.a, "b": self.b})


class IsotonicCalibrator(BaseCalibrator):
    """Piecewise-constant isotonic mapping fitted on (raw, label) pairs."""
    method = "isotonic"

    def __init__(self, xs: Optional[List[float]] = None, ys: Optional[List[float]] = None):
        self.xs = sorted(xs or [0.0, 1.0])
        self.ys = list(ys or [0.0, 1.0])
        if len(self.xs) != len(self.ys):
            raise ValueError("xs and ys must have equal length")

    def calibrate(self, raw: float) -> float:
        x = float(np.clip(raw, 0.0, 1.0))
        return float(np.clip(np.interp(x, self.xs, self.ys), 0.0, 1.0))

    def info(self) -> CalibrationInfo:
        return CalibrationInfo(method=self.method,
                               params={"knots": len(self.xs)})


def calibrator_from_config(cfg: Optional[Dict[str, Any]]) -> BaseCalibrator:
    """Build a calibrator from a config dict. None/missing -> explicit Identity."""
    if not cfg:
        return IdentityCalibrator()
    method = cfg.get("method", "none")
    params = cfg.get("params", {})
    if method == "temperature_scaling":
        return TemperatureScaling(**params)
    if method == "platt":
        return PlattScaling(**params)
    if method == "isotonic":
        return IsotonicCalibrator(**params)
    return IdentityCalibrator()
