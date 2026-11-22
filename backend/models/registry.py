"""Model registry (Phase 2).

Loads detector definitions from a configurable local model directory:

    models/
      README.md
      registry.json        # list of model entries

If no model is registered/found, detectors report MODEL_NOT_CONFIGURED
(equivalently ModelStatus.NOT_CONFIGURED) and the ensemble proceeds with
forensic signals only. NOTHING is faked.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..analyzers.base import AIDetectorProvider, ModelStatus, NullDetectorProvider
from ..utils.security import sha256_file, validate_model_path
from .adapters import ExternalAPIDetectorProvider, ONNXDetectorProvider, TorchDetectorProvider
from .stacking import ForensicStackingDetector

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

ADAPTERS = {
    "pytorch": TorchDetectorProvider,
    "onnx": ONNXDetectorProvider,
    "external-api": ExternalAPIDetectorProvider,
    "stacking": ForensicStackingDetector,
}

# V4: model lifecycle statuses (spec §32)
MODEL_STATUSES = {"DEMONSTRATION", "RESEARCH", "VALIDATED", "PRODUCTION", "DEPRECATED"}


def _lifecycle_status(entry: Dict[str, Any]) -> str:
    """Resolve lifecycle status: explicit config > demonstration flag > default."""
    cfg = entry.get("config", {}) or {}
    s = cfg.get("lifecycle_status")
    if s in MODEL_STATUSES:
        return s
    if cfg.get("demonstration"):
        return "DEMONSTRATION"
    return "RESEARCH"


class ModelRegistry:
    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = os.path.abspath(models_dir or DEFAULT_MODELS_DIR)
        self._providers: Dict[str, AIDetectorProvider] = {}
        self._errors: Dict[str, str] = {}

    # -- registry API ------------------------------------------------------
    def register(self, provider: AIDetectorProvider) -> None:
        self._providers[provider.get_model_name()] = provider

    def load(self, name: str) -> AIDetectorProvider:
        """Load (initialize) a registered provider by name."""
        provider = self._providers.get(name)
        if provider is None:
            raise KeyError(f"model '{name}' is not registered")
        load = getattr(provider, "load", None)
        if callable(load):
            load()
        return provider

    def unload(self, name: str) -> bool:
        """Drop a provider from the registry (frees references)."""
        return self._providers.pop(name, None) is not None

    def list_models(self) -> List[Dict[str, Any]]:
        out = []
        for name, p in self._providers.items():
            entry = {
                "name": name,
                "version": p.get_version(),
                "model_type": p.get_model_type(),
                "status": p.get_status().value,
                "limitations": p.get_limitations(),
                "supported_formats": p.get_supported_formats(),
                "preprocessing": p.get_preprocessing_info(),
            }
            # V3 record fields (Phase 8) + V4 lifecycle (spec §32)
            cfg = getattr(p, "config", {}) or {}
            for key in ("license", "input_size", "dataset_version", "training_data",
                        "intended_use", "lifecycle_status", "validation_status",
                        "benchmark_ids", "training_seed"):
                if key in cfg:
                    entry[key] = cfg[key]
            if "lifecycle_status" not in entry:
                # resolve from provider's registry entry
                entry["lifecycle_status"] = "DEMONSTRATION" if cfg.get("demonstration") else "RESEARCH"
            cal = getattr(p, "_model", {}) or {}
            if isinstance(cal, dict) and cal.get("calibration"):
                entry["calibration"] = cal["calibration"].get("method")
                entry["calibration_detail"] = cal["calibration"]
            out.append(entry)
        return out

    def get_status(self) -> Dict[str, Any]:
        models = self.list_models()
        ready = [m for m in models if m["status"] == ModelStatus.READY.value]
        return {
            "models_dir": self.models_dir,
            "total": len(models),
            "ready": len(ready),
            "models": models,
            "errors": self._errors,
            "ml_available": len(ready) > 0,
        }

    def get_ready_detectors(self) -> List[AIDetectorProvider]:
        return [p for p in self._providers.values()
                if p.get_status() == ModelStatus.READY]

    # -- loading from disk ---------------------------------------------------
    def load_from_directory(self, models_dir: Optional[str] = None) -> "ModelRegistry":
        """Read registry.json and instantiate adapters. Never raises for a
        missing/empty registry - that simply means no models configured."""
        if models_dir:
            self.models_dir = os.path.abspath(models_dir)
        reg_path = os.path.join(self.models_dir, "registry.json")
        if not os.path.isfile(reg_path):
            return self  # nothing configured: honest empty state
        try:
            with open(reg_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            self._errors["_registry"] = f"could not parse registry.json: {exc}"
            return self
        for entry in data.get("models", []):
            if not entry.get("enabled", True):
                continue
            try:
                self._register_entry(entry)
            except Exception as exc:
                name = entry.get("name", "?")
                self._errors[name] = f"{type(exc).__name__}: {exc}"
        return self

    def _register_entry(self, entry: Dict[str, Any]) -> None:
        backend = entry.get("backend", "onnx")
        adapter_cls = ADAPTERS.get(backend)
        if adapter_cls is None:
            raise ValueError(f"unknown backend '{backend}'")
        model_file = entry.get("model_file")
        model_path = None
        if model_file:
            model_path = validate_model_path(model_file, self.models_dir)
        provider = adapter_cls(
            name=entry.get("name", f"unnamed-{backend}"),
            version=entry.get("version", "unknown"),
            model_path=model_path,
            config=entry.get("config", {}),
        )
        # record file hash for reproducibility when a file is used
        if model_path:
            try:
                entry["_sha256"] = sha256_file(model_path)
            except Exception:
                pass
        self.register(provider)
        provider.load()


def get_global_registry(models_dir: Optional[str] = None) -> ModelRegistry:
    """Build (and load) the registry from the models directory."""
    return ModelRegistry(models_dir).load_from_directory()
