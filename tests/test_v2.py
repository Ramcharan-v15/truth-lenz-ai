"""V2 tests: detector interface, registry, calibration, preprocessing, tiling,
multi-model ensemble, benchmark metrics, robustness, C2PA states, generator
classifier, LLM guardrails, security, reports, and new API endpoints.

All 31 original tests must keep passing - this file only adds coverage.
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.analyzers.base import (
    AIDetectorProvider, AIDetectorResult, ModelStatus, NullDetectorProvider,
)
from backend.services.pipeline import build_context, run_analysis

DATA = os.path.join(os.path.dirname(__file__), "..", "test_data")


def img_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def make_image(w: int = 256, h: int = 256) -> bytes:
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class FixedDetector(AIDetectorProvider):
    """Test double: returns a fixed calibrated probability."""

    def __init__(self, prob: float, confidence: float = 80.0,
                 calibration: str = "temperature_scaling(T=1.5)"):
        self._prob = prob
        self._conf = confidence
        self._cal = calibration

    def get_model_name(self): return "test-fixed-detector"
    def get_version(self): return "0.1-test"
    def get_model_type(self): return "test-double"
    def get_limitations(self): return "test double - not a real model"

    def detect(self, ctx):
        return AIDetectorResult(
            detector_name=self.get_model_name(),
            detector_version=self.get_version(),
            model_type=self.get_model_type(),
            probability_ai=self._prob,
            probability_authentic=100 - self._prob,
            raw_score=self._prob / 100,
            calibration=self._cal,
            confidence=self._conf,
            limitations=self.get_limitations(),
            model_status=ModelStatus.READY,
        )


# ------------------------------------------------- Phase 1: formal interface
class TestDetectorInterface:
    def test_result_contract_fields(self):
        r = AIDetectorResult(detector_name="x", detector_version="1")
        d = r.to_dict()
        for key in ("detector_name", "detector_version", "model_type",
                    "probability_ai", "probability_authentic", "raw_score",
                    "calibration", "confidence", "supported_formats",
                    "preprocessing", "evidence", "limitations",
                    "inference_time_ms", "model_status"):
            assert key in d, f"missing field {key}"

    def test_null_detector_honest(self):
        n = NullDetectorProvider()
        assert n.get_status() == ModelStatus.NOT_CONFIGURED
        ctx = build_context("t.jpg", "image/jpeg", make_image())
        r = n.detect(ctx)
        assert r.probability_ai is None
        assert not r.available
        assert r.model_status == ModelStatus.NOT_CONFIGURED

    def test_every_detector_identifies_name_and_version(self):
        for det in (NullDetectorProvider(), FixedDetector(50)):
            assert det.get_model_name() and det.get_version()
            assert det.get_model_type()

    def test_raw_vs_calibrated_separate(self):
        det = FixedDetector(72.0)
        ctx = build_context("t.jpg", "image/jpeg", make_image())
        r = det.detect(ctx)
        assert r.raw_score != r.probability_ai  # distinct fields
        assert r.calibration != "none"


# ------------------------------------------------- Phase 2+3: registry/adapters
class TestRegistry:
    def test_empty_registry_honest(self):
        from backend.models.registry import ModelRegistry
        reg = ModelRegistry(models_dir="/tmp/tl-nonexistent-models")
        reg.load_from_directory()
        st = reg.get_status()
        assert st["total"] == 0 and st["ready"] == 0
        assert st["ml_available"] is False

    def test_register_unload_list(self):
        from backend.models.registry import ModelRegistry
        reg = ModelRegistry(models_dir="/tmp/tl-nonexistent-models")
        det = FixedDetector(60)
        reg.register(det)
        assert len(reg.list_models()) == 1
        assert reg.unload("test-fixed-detector") is True
        assert reg.list_models() == []

    def test_invalid_model_file_recorded_not_raised(self):
        import json, tempfile
        from backend.models.registry import ModelRegistry
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "registry.json"), "w") as fh:
                json.dump({"models": [{
                    "name": "bad", "version": "1", "backend": "onnx",
                    "model_file": "../../../etc/passwd", "enabled": True}]}, fh)
            reg = ModelRegistry(models_dir=td).load_from_directory()
            assert reg.get_status()["total"] == 0
            assert "bad" in reg.get_status()["errors"]  # path traversal blocked

    def test_missing_dependency_degrades_gracefully(self):
        from backend.models.adapters import ONNXDetectorProvider, TorchDetectorProvider
        for cls in (ONNXDetectorProvider, TorchDetectorProvider):
            p = cls(name="t", version="1", model_path=None, config={})
            p.load()
            assert p.get_status() in (ModelStatus.DEPENDENCY_MISSING, ModelStatus.ERROR)
            ctx = build_context("t.jpg", "image/jpeg", make_image())
            r = p.detect(ctx)
            assert r.probability_ai is None  # never faked
            assert not r.available


# ------------------------------------------------- Phase 4: preprocessing
class TestPreprocessing:
    def test_records_every_step(self):
        from backend.models.preprocessing import StandardPreprocessor
        pp = StandardPreprocessor({"target_size": (224, 224), "crop_strategy": "center",
                                   "normalization": "imagenet"})
        img = Image.new("RGB", (640, 480), (10, 20, 30))
        tensor, rec = pp.process(img)
        assert tensor.shape == (3, 224, 224)
        assert rec.original_dimensions == (640, 480)
        assert rec.model_input_dimensions == (224, 224)
        steps = [s["step"] for s in rec.steps]
        assert "color_convert" not in steps  # already RGB
        assert "resize" in steps and "center_crop" in steps and "normalize" in steps
        d = rec.to_dict()
        assert d["normalization"] == "imagenet"

    def test_no_silent_alteration(self):
        from backend.models.preprocessing import StandardPreprocessor
        pp = StandardPreprocessor()
        img = Image.new("RGBA", (100, 80), (1, 2, 3, 255))
        _, rec = pp.process(img)
        assert rec.original_mode == "RGBA"
        assert any(s["step"] == "color_convert" for s in rec.steps)


# ------------------------------------------------- Phase 5: tiling
class TestTiling:
    def test_tiles_cover_image(self):
        from backend.models.tiling import generate_tiles
        img = Image.new("RGB", (600, 400), (5, 5, 5))
        tiles = generate_tiles(img, tile_size=256, overlap=0.25)
        assert len(tiles) > 1
        # every tile inside bounds
        for t in tiles:
            x0, y0, x1, y1 = t["box"]
            assert 0 <= x0 < x1 <= 600 and 0 <= y0 < y1 <= 400

    def test_aggregation_returns_map(self):
        from backend.models.tiling import aggregate_tile_predictions, generate_tiles
        img = Image.new("RGB", (300, 300), (9, 9, 9))
        tiles = generate_tiles(img, tile_size=128, overlap=0.0)
        mean_p, uri, info = aggregate_tile_predictions(img, tiles, lambda im: 0.7)
        assert abs(mean_p - 0.7) < 1e-9
        assert uri.startswith("data:image/png;base64,")
        assert info["tile_count"] == len(tiles)


# ------------------------------------------------- Phase 9: calibration
class TestCalibration:
    def test_identity_labels_uncalibrated(self):
        from backend.models.calibration import calibrator_from_config
        c = calibrator_from_config(None)
        assert c.info().method == "none"
        assert "uncalibrated" in c.info().describe()
        assert c.calibrate(0.7) == pytest.approx(0.7)

    def test_temperature_scaling(self):
        from backend.models.calibration import TemperatureScaling
        c = TemperatureScaling(temperature=2.0, from_logits=False)
        # T>1 softens: 0.9 -> closer to 0.5
        assert c.calibrate(0.9) < 0.9
        assert 0.0 <= c.calibrate(0.2) <= 1.0

    def test_platt_scaling(self):
        from backend.models.calibration import PlattScaling
        c = PlattScaling(a=1.0, b=0.0)
        assert c.calibrate(0.0) == pytest.approx(0.5)

    def test_isotonic_monotone(self):
        from backend.models.calibration import IsotonicCalibrator
        c = IsotonicCalibrator(xs=[0.0, 0.5, 1.0], ys=[0.1, 0.4, 0.9])
        assert c.calibrate(0.25) < c.calibrate(0.75)


# ------------------------------------------------- Phase 7: generator classifier
class TestGeneratorClassifier:
    def test_unavailable_honest(self):
        from backend.models.generator import UnavailableGeneratorClassifier
        g = UnavailableGeneratorClassifier()
        pred = g.classify_generator(None)
        assert pred.label == "unknown"
        assert "unavailable" in pred.detail.lower()
        assert "trained" in g.get_attribution_limitations().lower()


# ------------------------------------------------- Phase 8: multi-model ensemble
class TestHybridEnsemble:
    def _ctx_results(self, data: bytes):
        from backend.services.pipeline import ANALYZERS
        ctx = build_context("t.jpg", "image/jpeg", data)
        return ctx, [a.run(ctx) for a in ANALYZERS]

    def test_ml_does_not_dominate(self):
        from backend.analyzers.ensemble import EnsembleEngine
        from backend.utils.image_io import load_config
        # authentic photo: forensics lean authentic; rogue ML claims 95% AI
        data = img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg")
        ctx, results = self._ctx_results(data)
        eng = EnsembleEngine(load_config(), detectors=[FixedDetector(95.0, 90.0)])
        out = eng.combine(ctx, results)
        # ML pulled it up but could not single-handedly dictate the verdict
        assert out["ai_probability"] < 95.0
        assert out["ml_available"] is True
        assert out["ml_ai_probability"] == pytest.approx(95.0)

    def test_disagreement_forces_inconclusive(self):
        from backend.analyzers.ensemble import EnsembleEngine
        from backend.utils.image_io import load_config
        # SD-signature image: forensics say AI; detector says ~not AI, confidently
        data = img_bytes(f"{DATA}/ai_generated/sd_output.png")
        ctx, results = self._ctx_results(data)
        eng = EnsembleEngine(load_config(), detectors=[FixedDetector(5.0, 85.0)])
        out = eng.combine(ctx, results)
        assert out["verdict"] == "INCONCLUSIVE"
        assert out["ml_disagreement"] > 25
        assert any("disagree" in p for p in out["confidence_penalties"])

    def test_no_detector_unchanged_behavior(self):
        # V3: a demonstration model IS registered. This test verifies the
        # no-detector path by running the pipeline with detectors disabled.
        data = img_bytes(f"{DATA}/ai_generated/sd_output.png")
        rep = run_analysis("sd.png", "image/png", data, detectors=[])
        assert rep["ensemble"]["verdict"] == "LIKELY_AI_GENERATED"
        assert rep["ensemble"]["ml_available"] is False


# ------------------------------------------------- Phase 10: benchmark metrics
class TestBenchmarkMetrics:
    def test_perfect_scores(self):
        from backend.benchmark.metrics import full_evaluation
        y_true = [0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9]
        m = full_evaluation(y_true, y_prob)
        assert m["accuracy"] == 1.0 and m["f1"] == 1.0
        assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
        assert m["tp"] == 2 and m["tn"] == 2

    def test_worst_scores(self):
        from backend.benchmark.metrics import full_evaluation
        m = full_evaluation([0, 0, 1, 1], [0.9, 0.8, 0.1, 0.2])
        assert m["accuracy"] == 0.0
        assert m["false_positive_rate"] == 1.0
        assert m["roc_auc"] == 0.0

    def test_calibration_error_sane(self):
        from backend.benchmark.metrics import expected_calibration_error
        ece = expected_calibration_error([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2])
        assert 0.0 <= ece["ece"] <= 0.3
        assert ece["n_bins"] == 10

    def test_single_class_roc_nan(self):
        from backend.benchmark.metrics import roc_auc
        import math
        assert math.isnan(roc_auc([0, 0], [0.1, 0.2]))


# ------------------------------------------------- Phase 12: robustness
class TestRobustness:
    def test_transforms_run_and_score(self):
        from backend.benchmark import robustness
        from backend.services.pipeline import run_analysis
        data = make_image()
        out = robustness.evaluate_robustness(run_analysis, data, "t.jpg",
                                             transforms=robustness.TRANSFORMS[:3])
        assert out["overall_robustness_score"] >= 0
        assert len(out["transforms"]) == 3
        assert all("robustness_score" in t for t in out["transforms"])
        assert "original" in out and "verdict" in out["original"]

    def test_all_transform_functions_valid(self):
        from backend.benchmark import robustness
        img = Image.new("RGB", (200, 200), (50, 60, 70))
        for t in robustness.TRANSFORMS:
            out = t["fn"](img)
            assert isinstance(out, Image.Image)
            assert out.size[0] > 0 and out.size[1] > 0


# ------------------------------------------------- Phase 13: C2PA states
class TestC2PA:
    def test_missing_for_plain_jpeg(self):
        from backend.analyzers.metadata import _inspect_c2pa
        data = img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg")
        status, info = _inspect_c2pa(data, "JPEG")
        assert status == "missing"
        assert info == {}

    def test_unsupported_for_bmp(self):
        from backend.analyzers.metadata import _inspect_c2pa
        buf = io.BytesIO()
        Image.new("RGB", (32, 32)).save(buf, format="BMP")
        status, _ = _inspect_c2pa(buf.getvalue(), "BMP")
        assert status == "unsupported"

    def test_detects_manifest_markers(self):
        from backend.analyzers.metadata import _inspect_c2pa
        blob = b"\xff\xd8\xff" + b"jumb" + b"c2pa_manifest" + b'"issuer":"TestIssuer"' + b"\x00" * 100
        status, info = _inspect_c2pa(blob, "JPEG")
        assert status == "unvalidated"
        assert info.get("issuer") == "TestIssuer"

    def test_missing_never_ai_evidence(self):
        data = img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg")
        rep = run_analysis("c.jpg", "image/jpeg", data)
        md = next(s for s in rep["signals"] if s["name"] == "metadata")
        assert md["technical"]["c2pa_status"] == "missing"
        # no evidence entry should claim AI from missing C2PA
        assert not any("c2pa" in e["text"].lower() and "ai" in e["text"].lower()
                       for e in md["evidence"])


# ------------------------------------------------- Phase 19: reproducibility
class TestReproducibility:
    def test_repro_block_complete(self):
        rep = run_analysis("t.jpg", "image/jpeg", make_image())
        r = rep["reproducibility"]
        assert r["app_version"] and r["backend_version"]
        assert r["config_version"] and len(r["config_sha256"]) == 16
        assert len(r["image_sha256"]) == 64
        assert r["analysis_id"] == rep["analysis_id"]

    def test_report_hash_stable(self):
        import hashlib, json
        rep = run_analysis("t.jpg", "image/jpeg", make_image())
        canonical = json.dumps({k: v for k, v in rep.items() if k != "report_hash"},
                               sort_keys=True, separators=(",", ":"))
        assert hashlib.sha256(canonical.encode()).hexdigest() == rep["report_hash"]


# ------------------------------------------------- Phase 21: LLM guardrails
class TestLLMGuardrails:
    def test_disabled_explainer(self):
        from backend.llm.interface import get_explainer
        ex = get_explainer()
        assert ex.get_provider_name() == "not configured"
        with pytest.raises(RuntimeError):
            ex.explain({}, "basic")

    def test_evidence_packet_no_invention(self):
        from backend.utils.security import serialize_evidence_for_llm, validate_llm_output
        rep = run_analysis("t.jpg", "image/jpeg", make_image())
        packet = serialize_evidence_for_llm(rep)
        assert "EVIDENCE PACKET" in packet
        assert "Do not invent" in packet
        # invented number flagged
        warns = validate_llm_output("The image is 99% AI-generated.", rep)
        assert any("99%" in w for w in warns)
        # real number passes
        ai = int(round(rep["ensemble"]["ai_probability"]))
        warns2 = validate_llm_output(f"AI probability is {ai}%.", rep)
        assert warns2 == []


# ------------------------------------------------- Phase 20: security
class TestSecurity:
    def test_model_path_traversal_blocked(self):
        from backend.utils.security import validate_model_path
        with pytest.raises(ValueError):
            validate_model_path("../../../etc/passwd", "/tmp/tl-models")

    def test_model_extension_blocked(self):
        import tempfile
        from backend.utils.security import validate_model_path
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "evil.exe")
            open(p, "w").write("x")
            with pytest.raises(ValueError):
                validate_model_path("evil.exe", td)

    def test_ssrf_private_blocked(self):
        from backend.utils.security import validate_outbound_url
        with pytest.raises(ValueError):
            validate_outbound_url("http://127.0.0.1:8000/v1/x")
        with pytest.raises(ValueError):
            validate_outbound_url("http://10.0.0.5/x")
        with pytest.raises(ValueError):
            validate_outbound_url("ftp://example.com/x")

    def test_no_traceback_in_api_errors(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        c = TestClient(app, raise_server_exceptions=False)
        r = c.post("/api/analyze", files={"file": ("x.bin", b"\x00" * 100, "application/octet-stream")})
        assert r.status_code in (400, 415)
        assert "Traceback" not in r.text


# ------------------------------------------------- new API endpoints
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


class TestV2API:
    def test_models_endpoint(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        body = r.json()
        # V3: demonstration model registered and active
        assert body["ml_available"] is True
        assert body["total"] >= 1
        names = [m["name"] for m in body["models"]]
        assert "tl-forensic-stacking" in names

    def test_research_endpoint(self, client):
        data = img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg")
        ar = client.post("/api/analyze?save=true", files={"file": ("c.jpg", data, "image/jpeg")})
        aid = ar.json()["analysis_id"]
        r = client.get(f"/api/research/{aid}")
        assert r.status_code == 200
        body = r.json()
        assert body["report_hash"]
        assert "signals" in body and "reproducibility" in body
        client.delete(f"/api/history/{aid}")

    def test_explain_endpoint(self, client):
        data = img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg")
        r = client.post("/api/explain?level=basic", files={"file": ("c.jpg", data, "image/jpeg")})
        assert r.status_code == 200
        body = r.json()
        assert body["deterministic_explanation"]
        assert body["llm"]["configured"] is False

    def test_benchmark_endpoints(self, client):
        r = client.get("/api/benchmark/results?dataset=sample")
        # sample results.json exists from earlier smoke run; either way must not 500
        assert r.status_code in (200, 404)

    def test_robustness_endpoint(self, client):
        data = img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg")
        r = client.post("/api/robustness", files={"file": ("c.jpg", data, "image/jpeg")})
        assert r.status_code == 200
        assert "overall_robustness_score" in r.json()
