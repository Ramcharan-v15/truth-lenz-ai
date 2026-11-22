"""Teacher Demo tests — no inference, no pilot images, no REAL test set.

Covers:
- Teacher upload validation helper (magic bytes, empty, oversized,
  unsupported, extension spoofing) using synthetic Pillow fixtures
  generated in tmp_path only.
- GET /api/models metadata contract actually used by the teacher demo:
  it must report the current demonstration model honestly, with no
  prediction, probability, or confidence fields, and must not run
  inference (no /api/analyze call is made in this file).
- Unknown /api/* routes return JSON 404 (Stage 3), which the teacher
  demo maps to ANALYSIS_UNAVAILABLE, not to a verdict.

Explicitly NOT tested here: detector inference, training, evaluation,
pilot images, frozen REAL test images, and the existing /api/analyze
pipeline. Those are out of scope for this task and are not invoked.
"""

import io
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.teacher_demo_validation import (
    TEACHER_MAX_BYTES,
    detect_teacher_format,
    validate_teacher_upload_bytes,
)


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (8, 6), color=(10, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (8, 6), color=(200, 60, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def _webp_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (8, 6), color=(30, 200, 90)).save(buf, format="WEBP")
    return buf.getvalue()


def test_detect_teacher_format_jpeg_png_webp():
    assert detect_teacher_format(_jpeg_bytes()) == "image/jpeg"
    assert detect_teacher_format(_png_bytes()) == "image/png"
    assert detect_teacher_format(_webp_bytes()) == "image/webp"


def test_validate_valid_images_return_detected_mime():
    assert validate_teacher_upload_bytes(_png_bytes(), "photo.png") == "image/png"
    assert validate_teacher_upload_bytes(_jpeg_bytes(), "photo.jpg") == "image/jpeg"
    assert validate_teacher_upload_bytes(_webp_bytes(), "photo.webp") == "image/webp"


def test_validate_rejects_empty():
    with pytest.raises(ValueError, match="Empty"):
        validate_teacher_upload_bytes(b"", "empty.png")


def test_validate_rejects_unsupported_text_and_gif():
    with pytest.raises(ValueError, match="Unsupported"):
        validate_teacher_upload_bytes(b"hello, this is not an image", "note.png")
    # GIF header: not in the teacher demo's allowed set.
    with pytest.raises(ValueError, match="Unsupported"):
        validate_teacher_upload_bytes(b"GIF89a" + b"\x00" * 20, "anim.gif")


def test_validate_rejects_extension_spoofing():
    # .png name, text contents: magic bytes decide, extension does not.
    with pytest.raises(ValueError, match="Unsupported"):
        validate_teacher_upload_bytes(b"not an image at all", "spoof.png")


def test_validate_magic_wins_over_extension():
    # PNG bytes with a .jpg name: accepted as PNG, because magic bytes
    # are authoritative and the extension alone is never trusted.
    assert validate_teacher_upload_bytes(_png_bytes(), "photo.jpg") == "image/png"


def test_validate_rejects_oversized_with_small_limit():
    # Use a small limit parameter so the test does not allocate 25 MB.
    with pytest.raises(ValueError, match="too large"):
        validate_teacher_upload_bytes(_png_bytes(), "big.png", max_bytes=10)


def test_teacher_max_bytes_matches_backend_limit():
    # Teacher demo limit matches the backend per-file limit (25 MB).
    assert TEACHER_MAX_BYTES == 25 * 1024 * 1024


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from backend.main import app

    return TestClient(app, raise_server_exceptions=False)


def test_models_endpoint_reports_demonstration_model_honestly(client):
    """GET /api/models is metadata only: no inference, no prediction.

    The current registry must report tl-forensic-stacking as a
    demonstration / synthetic-only model. If that ever changes, this
    test will fail closed and the teacher demo's honesty claim must
    be re-checked before any classification is shown.
    """
    r = client.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body
    assert "models_dir" not in body, "server filesystem path must not leak"
    models = body["models"]
    assert len(models) >= 1
    stacking = [m for m in models if m["name"] == "tl-forensic-stacking"]
    assert stacking, "expected tl-forensic-stacking in registry"
    m = stacking[0]
    assert m["version"] == "1.0.0"
    # Honest demonstration status, in at least one of these fields.
    text = " ".join(str(v) for v in m.values()).lower()
    assert "demonstration" in text or "synthetic" in text
    assert m.get("lifecycle_status") in ("DEMONSTRATION", None) or "demonstration" in text
    # No prediction fields on a metadata endpoint.
    for forbidden in ("probability_ai", "probability", "confidence", "verdict", "ai_probability"):
        assert forbidden not in body
        assert forbidden not in m


def test_models_endpoint_has_no_prediction_fields(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    payload = r.text.lower()
    assert "likely_ai_generated" not in payload
    assert "likely_authentic" not in payload


def test_unknown_api_route_is_json_404_not_a_verdict(client):
    """Stage 3 JSON 404: teacher demo maps this to ANALYSIS_UNAVAILABLE."""
    r = client.get("/api/teacher-demo-does-not-exist")
    assert r.status_code == 404
    # Must be JSON, not SPA HTML, and must not contain a verdict.
    assert "text/html" not in r.headers.get("content-type", "")
    assert "likely_" not in r.text.lower()
