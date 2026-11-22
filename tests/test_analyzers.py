"""Unit + integration tests for Truth Lenz AI forensic analyzers.

All test images are synthetic (see generate_test_data.py) - no copyrighted
material. Run: .venv/bin/python -m pytest tests/ -x -q
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.analyzers.base import SignalStatus
from backend.services import pipeline
from backend.services.pipeline import build_context, run_analysis
from backend.utils.image_io import decode_image, load_config

DATA = os.path.join(os.path.dirname(__file__), "..", "test_data")


def img_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


@pytest.fixture(scope="session", autouse=True)
def _dataset():
    sys.path.insert(0, os.path.dirname(__file__))
    import generate_test_data
    generate_test_data.main()


# ---------------------------------------------------------------- file validation
class TestValidation:
    def test_rejects_non_image(self):
        with pytest.raises(ValueError):
            decode_image(b"this is not an image at all" * 100)

    def test_rejects_truncated(self):
        data = img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg")
        with pytest.raises(ValueError):
            decode_image(data[:200])

    def test_decode_ok(self):
        data = img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg")
        im, info = decode_image(data)
        assert im.size == (512, 512)
        assert im.mode == "RGB"

    def test_downscales_huge(self):
        big = Image.new("RGB", (3000, 2000), (10, 20, 30))
        buf = io.BytesIO()
        big.save(buf, format="JPEG", quality=80)
        im, info = decode_image(buf.getvalue())
        assert max(im.size) <= 2048 and info["downscaled"]


# ---------------------------------------------------------------- metadata
class TestMetadata:
    def _run(self, path):
        from backend.analyzers.metadata import MetadataAnalyzer
        data = img_bytes(path)
        ctx = build_context("t.jpg", "image/jpeg", data)
        return MetadataAnalyzer().run(ctx)

    def test_ai_signature_detected(self):
        r = self._run(f"{DATA}/ai_generated/sd_output.png")
        assert r.ai_score >= 90
        assert any("AI generator signature" in e.text or "generation parameters" in e.text
                   for e in r.evidence)

    def test_camera_exif_supports_authentic(self):
        r = self._run(f"{DATA}/authentic_camera/canon_photo.jpg")
        assert any(e.direction.value == "supports_authentic" for e in r.evidence)
        assert r.ai_score <= 50

    def test_editing_software_flagged(self):
        r = self._run(f"{DATA}/edited/photoshop_export.jpg")
        assert r.manip_score >= 70

    def test_missing_metadata_not_ai_evidence(self):
        """Missing metadata must NEVER count as AI evidence."""
        r = self._run(f"{DATA}/screenshots/ui_screenshot.png")
        assert r.ai_score <= 55, "missing metadata treated as AI evidence!"


# ---------------------------------------------------------------- statistics / histograms
class TestStatistics:
    def test_histograms_present(self):
        from backend.analyzers.statistics import StatisticsAnalyzer
        data = img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg")
        ctx = build_context("t.jpg", "image/jpeg", data)
        r = StatisticsAnalyzer().run(ctx)
        h = r.visuals["histograms"]
        assert set(h) == {"r", "g", "b", "luminance"}
        assert len(h["r"]["counts"]) == 64
        assert r.technical["luminance_entropy_bits"] > 4


# ---------------------------------------------------------------- compression
class TestCompression:
    def test_heavy_jpeg_warned(self):
        from backend.analyzers.compression import CompressionAnalyzer
        data = img_bytes(f"{DATA}/compressed/heavy_q30.jpg")
        ctx = build_context("t.jpg", "image/jpeg", data)
        r = CompressionAnalyzer().run(ctx)
        q = r.technical.get("estimated_jpeg_quality")
        assert q is not None and q <= 60
        assert any("Heavy" in w for w in r.warnings)

    def test_png_lossless_noted(self):
        from backend.analyzers.compression import CompressionAnalyzer
        data = img_bytes(f"{DATA}/screenshots/ui_screenshot.png")
        ctx = build_context("t.png", "image/png", data)
        r = CompressionAnalyzer().run(ctx)
        assert r.technical.get("lossless") is True


# ---------------------------------------------------------------- noise
class TestNoise:
    def test_flat_region_estimation(self):
        from backend.analyzers.noise import NoiseAnalyzer
        data = img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg")
        ctx = build_context("t.jpg", "image/jpeg", data)
        r = NoiseAnalyzer().run(ctx)
        # synthetic photo-like image has real noise -> camera-like
        assert r.technical["noise_std"] > 0.001
        assert r.status == SignalStatus.OK


# ---------------------------------------------------------------- frequency
class TestFrequency:
    def test_spectrum_visual(self):
        from backend.analyzers.frequency import FrequencyAnalyzer
        data = img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg")
        ctx = build_context("t.jpg", "image/jpeg", data)
        r = FrequencyAnalyzer().run(ctx)
        assert r.visuals["spectrum"].startswith("data:image/png;base64,")
        assert "spectral_slope" in r.technical
        # natural-ish slope range
        assert -7.0 < r.technical["spectral_slope"] < -1.0

    def test_smooth_synth_steeper_slope(self):
        from backend.analyzers.frequency import FrequencyAnalyzer
        d1 = img_bytes(f"{DATA}/ai_generated/smooth_synth.jpg")
        d2 = img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg")
        r1 = FrequencyAnalyzer().run(build_context("a.jpg", "image/jpeg", d1))
        r2 = FrequencyAnalyzer().run(build_context("b.jpg", "image/jpeg", d2))
        assert r1.technical["spectral_slope"] < r2.technical["spectral_slope"]


# ---------------------------------------------------------------- pixel / ELA
class TestPixel:
    def test_ela_visual(self):
        from backend.analyzers.pixel_forensics import PixelForensicsAnalyzer
        data = img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg")
        ctx = build_context("t.jpg", "image/jpeg", data)
        r = PixelForensicsAnalyzer().run(ctx)
        assert r.visuals["ela"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------- manipulation
class TestManipulation:
    def test_copymove_detected(self):
        from backend.analyzers.manipulation import ManipulationAnalyzer
        data = img_bytes(f"{DATA}/edited/copymove_clone.jpg")
        ctx = build_context("t.jpg", "image/jpeg", data)
        r = ManipulationAnalyzer().run(ctx)
        assert r.manip_score >= 75, f"copy-move missed: {r.manip_score}"
        assert r.visuals["manipulation_overlay"].startswith("data:image/png;base64,")

    def test_clean_image_no_copymove(self):
        from backend.analyzers.manipulation import ManipulationAnalyzer
        data = img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg")
        ctx = build_context("t.jpg", "image/jpeg", data)
        r = ManipulationAnalyzer().run(ctx)
        assert r.manip_score < 60, f"false positive: {r.manip_score}"


# ---------------------------------------------------------------- quality / screenshot
class TestQuality:
    def test_screenshot_detected(self):
        from backend.analyzers.quality import QualityAnalyzer
        data = img_bytes(f"{DATA}/screenshots/ui_screenshot.png")
        ctx = build_context("t.png", "image/png", data)
        r = QualityAnalyzer().run(ctx)
        assert r.technical["screenshot_likely"] is True

    def test_quality_tiers(self):
        from backend.analyzers.quality import QualityAnalyzer
        data = img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg")
        ctx = build_context("t.jpg", "image/jpeg", data)
        r = QualityAnalyzer().run(ctx)
        assert r.technical["quality_tier"] in ("Excellent", "Good", "Fair", "Poor")

    def test_blur_detection_scale(self):
        """Laplacian variance must be on 0-255 scale; sharp photos must not be 'blurry'."""
        import numpy as np
        from scipy.ndimage import gaussian_filter
        from backend.analyzers.quality import QualityAnalyzer
        data = img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg")
        ctx = build_context("t.jpg", "image/jpeg", data)
        r = QualityAnalyzer().run(ctx)
        assert r.technical["blur_laplacian_variance"] > 60, "sharp test image flagged blurry"
        # heavily blurred variant must score low
        arr = (ctx.rgb * 255).astype(float)
        blurred = gaussian_filter(arr, 8)
        ctx2 = build_context("t.jpg", "image/jpeg", data)
        ctx2.rgb = (blurred / 255).astype(np.float32)
        ctx2.gray = (0.299 * ctx2.rgb[..., 0] + 0.587 * ctx2.rgb[..., 1]
                     + 0.114 * ctx2.rgb[..., 2]).astype(np.float32)
        r2 = QualityAnalyzer().run(ctx2)
        assert r2.technical["blur_laplacian_variance"] < 60, "blurred image not detected"


# ---------------------------------------------------------------- ensemble
class TestEnsemble:
    def _verdict(self, path, mime):
        return run_analysis("t", mime, img_bytes(path))["ensemble"]

    def test_ai_signature_verdict(self):
        e = self._verdict(f"{DATA}/ai_generated/sd_output.png", "image/png")
        assert e["verdict"] == "LIKELY_AI_GENERATED"
        assert e["ai_probability"] >= 60

    def test_copymove_verdict(self):
        e = self._verdict(f"{DATA}/edited/copymove_clone.jpg", "image/jpeg")
        assert e["verdict"] == "LIKELY_MANIPULATED"

    def test_probabilities_bounded(self):
        e = self._verdict(f"{DATA}/authentic_camera/canon_photo.jpg", "image/jpeg")
        for k in ("ai_probability", "manipulation_probability", "authenticity_probability"):
            assert 1 <= e[k] <= 99, "never claim 100% certainty"
        assert e["verdict"] in ("LIKELY_AUTHENTIC", "LIKELY_AI_GENERATED",
                                "LIKELY_MANIPULATED", "INCONCLUSIVE")

    def test_confidence_separate_from_probability(self):
        e = self._verdict(f"{DATA}/compressed/heavy_q30.jpg", "image/jpeg")
        assert e["confidence"] < 70  # heavy compression must reduce confidence
        assert any("compression" in p.lower() for p in e["confidence_penalties"])

    def test_config_weights_documented(self):
        cfg = load_config()
        assert "weights" in cfg and "verdict_thresholds" in cfg
        assert cfg["weights"]["metadata"]["ai"] > 1.0


# ---------------------------------------------------------------- API
class TestAPI:
    @pytest.fixture(scope="module")
    def client(self):
        from fastapi.testclient import TestClient
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from backend.main import app
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"

    def test_analyze(self, client):
        data = img_bytes(f"{DATA}/ai_generated/sd_output.png")
        r = client.post("/api/analyze", files={"file": ("sd.png", data, "image/png")})
        assert r.status_code == 200
        j = r.json()
        assert j["ensemble"]["verdict"] == "LIKELY_AI_GENERATED"
        assert len(j["signals"]) == 10

    def test_reject_bad_file(self, client):
        r = client.post("/api/analyze", files={"file": ("x.txt", b"hello", "text/plain")})
        assert r.status_code in (400, 415)

    def test_reject_oversize(self, client):
        big = b"\xff\xd8\xff" + b"\x00" * (26 * 1024 * 1024)
        r = client.post("/api/analyze", files={"file": ("big.jpg", big, "image/jpeg")})
        assert r.status_code == 413

    def test_batch(self, client):
        files = [
            ("files", ("a.jpg", img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg"), "image/jpeg")),
            ("files", ("b.png", img_bytes(f"{DATA}/ai_generated/sd_output.png"), "image/png")),
        ]
        r = client.post("/api/batch", files=files)
        assert r.status_code == 200
        assert r.json()["count"] == 2

    def test_compare(self, client):
        files = {
            "file_a": ("a.jpg", img_bytes(f"{DATA}/authentic_camera/canon_photo.jpg"), "image/jpeg"),
            "file_b": ("b.png", img_bytes(f"{DATA}/ai_generated/sd_output.png"), "image/png"),
        }
        r = client.post("/api/compare", files=files)
        assert r.status_code == 200
        assert "comparison" in r.json()
