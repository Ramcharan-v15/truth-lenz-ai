"""Model backend adapters (Phase 3).

Each adapter exposes the SAME AIDetectorResult contract regardless of
framework. Missing optional dependencies degrade to DEPENDENCY_MISSING -
never to fake results.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..analyzers.base import (AIDetectorProvider, AIDetectorResult,
                              AnalysisContext, ModelStatus)
from .calibration import BaseCalibrator, IdentityCalibrator, calibrator_from_config
from .preprocessing import StandardPreprocessor


class BaseModelAdapter(AIDetectorProvider):
    """Shared plumbing: preprocessing, calibration, timing, result building."""

    model_type = "base"

    def __init__(self, name: str, version: str, model_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        self._name = name
        self._version = version
        self.model_path = model_path
        self.config = config or {}
        self.preprocessor = StandardPreprocessor(self.config.get("preprocessing"))
        self.calibrator: BaseCalibrator = calibrator_from_config(self.config.get("calibration"))
        self._status = ModelStatus.NOT_CONFIGURED
        self._status_detail = "not initialized"
        self.supports_tiling = bool(self.config.get("supports_tiling", False))

    # -- AIDetectorProvider contract -------------------------------------
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
        return self.preprocessor.describe()

    def get_limitations(self) -> str:
        return self.config.get(
            "limitations",
            "Model limitations not documented by the operator. Treat outputs cautiously.")

    # -- to be implemented by backends ------------------------------------
    def _load_model(self) -> None:
        raise NotImplementedError

    def _infer(self, tensor) -> float:
        """Return RAW model score (probability or logit of AI class)."""
        raise NotImplementedError

    # -- shared detect() ---------------------------------------------------
    def load(self) -> None:
        try:
            self._status = ModelStatus.LOADING
            self._load_model()
            self._status = ModelStatus.READY
            self._status_detail = "ready"
        except ImportError as exc:
            self._status = ModelStatus.DEPENDENCY_MISSING
            self._status_detail = f"missing dependency: {exc}"
        except Exception as exc:
            self._status = ModelStatus.ERROR
            self._status_detail = f"load failed: {type(exc).__name__}: {exc}"

    def _unavailable(self, reason: str) -> AIDetectorResult:
        return AIDetectorResult(
            detector_name=self._name, detector_version=self._version,
            model_type=self.model_type, confidence=0.0,
            limitations=self.get_limitations(), model_status=self._status,
            detail=reason)

    def detect(self, ctx: AnalysisContext) -> AIDetectorResult:
        if self._status != ModelStatus.READY:
            return self._unavailable(f"model not ready ({self._status_detail})")
        started = time.perf_counter()
        try:
            tensor, prec_rec = self.preprocessor.process(ctx.pil_image)
            raw = float(self._infer(tensor))
            calibrated = self.calibrator.calibrate(raw)
            cal_info = self.calibrator.info()
            # uncalibrated outputs get capped influence downstream
            conf = float(self.config.get("confidence", 70.0))
            if cal_info.method == "none":
                conf = min(conf, 60.0)
            elapsed = (time.perf_counter() - started) * 1000
            return AIDetectorResult(
                detector_name=self._name, detector_version=self._version,
                model_type=self.model_type,
                probability_ai=round(calibrated * 100, 1),
                probability_authentic=round((1 - calibrated) * 100, 1),
                raw_score=round(raw, 4),
                calibration=cal_info.describe(),
                confidence=conf,
                supported_formats=self.get_supported_formats(),
                preprocessing=prec_rec.to_dict(),
                evidence=[f"Model raw score {raw:.4f} -> calibrated P(AI) {calibrated * 100:.1f}%"],
                limitations=self.get_limitations(),
                inference_time_ms=elapsed,
                model_status=ModelStatus.READY,
                detail=f"calibration: {cal_info.describe()}",
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


class TorchDetectorProvider(BaseModelAdapter):
    """PyTorch (.pt/.pth) image classifier adapter. Expects a model that maps
    a CHW float tensor -> single AI-class score."""
    model_type = "pytorch"

    def _load_model(self) -> None:
        import torch  # noqa
        if not self.model_path:
            raise RuntimeError("no model_path configured")
        # weights-only, no pickle execution of arbitrary code paths beyond torch.load
        self._torch = torch
        self._model = torch.load(self.model_path, map_location="cpu", weights_only=True)
        if hasattr(self._model, "eval"):
            self._model.eval()

    def _infer(self, tensor) -> float:
        import torch
        import numpy as np
        t = torch.from_numpy(np.asarray(tensor)).unsqueeze(0)
        with torch.no_grad():
            out = self._model(t)
        import numpy as _np
        v = _np.asarray(out).ravel()[0]
        return float(v)


class ONNXDetectorProvider(BaseModelAdapter):
    """ONNX Runtime (.onnx) adapter."""
    model_type = "onnx"

    def _load_model(self) -> None:
        import onnxruntime  # noqa
        if not self.model_path:
            raise RuntimeError("no model_path configured")
        import onnxruntime as ort
        self._session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def _infer(self, tensor) -> float:
        import numpy as np
        arr = np.asarray(tensor, dtype=np.float32)[None, ...]
        out = self._session.run(None, {self._input_name: arr})
        return float(np.asarray(out[0]).ravel()[0])


class ExternalAPIDetectorProvider(BaseModelAdapter):
    """Remote AI-detection API adapter.

    Sends the ORIGINAL image bytes (never the preprocessed tensor) to a
    configured HTTPS endpoint via multipart POST, then extracts the AI
    probability using a configurable JSON path. SSRF protections apply
    (see backend/utils/security.py).
    """

    model_type = "external-api"

    def _load_model(self) -> None:
        cfg = self.config.get("api", {})
        if not cfg.get("endpoint"):
            raise RuntimeError("no api.endpoint configured")
        from ..utils.security import validate_outbound_url
        validate_outbound_url(cfg["endpoint"])  # raises on SSRF risk
        self._endpoint = cfg["endpoint"]
        self._api_key_env = cfg.get("api_key_env", "")
        self._score_path = cfg.get("score_json_path", "ai_probability")
        self._timeout = float(cfg.get("timeout_s", 20))

    def _extract(self, payload: dict, path: str):
        cur: Any = payload
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                raise KeyError(f"score path '{path}' not found in API response")
        return float(cur)

    def detect(self, ctx: AnalysisContext) -> AIDetectorResult:
        # Override: external APIs want original bytes, not our tensor.
        if self._status != ModelStatus.READY:
            return self._unavailable(f"model not ready ({self._status_detail})")
        import io as _io
        import json as _json
        import os
        import urllib.request
        started = time.perf_counter()
        try:
            boundary = "----truthlenz-api"
            body = (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
                f"filename=\"image.jpg\"\r\nContent-Type: {ctx.mime_type}\r\n\r\n"
            ).encode() + ctx.file_bytes + f"\r\n--{boundary}--\r\n".encode()
            headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
            if self._api_key_env and os.environ.get(self._api_key_env):
                headers["Authorization"] = f"Bearer {os.environ[self._api_key_env]}"
            req = urllib.request.Request(self._endpoint, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = _json.load(resp)
            raw = self._extract(payload, self._score_path)
            # API may return 0-1 or 0-100; normalize
            if raw > 1.0:
                raw = raw / 100.0
            calibrated = self.calibrator.calibrate(raw)
            cal_info = self.calibrator.info()
            elapsed = (time.perf_counter() - started) * 1000
            _, prec_rec = self.preprocessor.process(ctx.pil_image)
            return AIDetectorResult(
                detector_name=self._name, detector_version=self._version,
                model_type=self.model_type,
                probability_ai=round(calibrated * 100, 1),
                probability_authentic=round((1 - calibrated) * 100, 1),
                raw_score=round(raw, 4),
                calibration=cal_info.describe(),
                confidence=float(self.config.get("confidence", 65.0)),
                preprocessing={"note": "remote inference; local preprocessing record for reference",
                               **prec_rec.to_dict()},
                evidence=[f"Remote model raw score {raw:.4f} -> calibrated P(AI) {calibrated * 100:.1f}%"],
                limitations=self.get_limitations() + " The image leaves this server when sent to the remote API.",
                inference_time_ms=elapsed,
                model_status=ModelStatus.READY,
                detail=f"endpoint: {self._endpoint}",
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            return AIDetectorResult(
                detector_name=self._name, detector_version=self._version,
                model_type=self.model_type, confidence=0.0,
                limitations=self.get_limitations(),
                inference_time_ms=elapsed,
                model_status=ModelStatus.ERROR,
                detail=f"remote inference failed: {type(exc).__name__}: {str(exc)[:200]}")
