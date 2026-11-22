"""Numerical-integrity regression tests (V5.1.8.62).

Covers the V5.1.8.61 reproduction matrix (black / white / solid
non-representable color / gradient / seeded noise at 64, 256, 1024),
1x1 and flat JPEG/WebP cases, clamp() rejection, the exact
mirror_symmetry_corr root cause, injected non-finite values, strict
JSON for API responses and persisted reports, failed-persistence
isolation, and unchanged behavior on ordinary valid images.

All state is isolated by tests/conftest.py (runtime writes redirect
to tmp_path); no test writes to real project databases, counters,
registries, or manifests.
"""
import copy
import io
import json
import math
import random
import sqlite3

import pytest
from PIL import Image

from backend.analyzers.base import AnalyzerResult, clamp
from backend.services import pipeline, storage
from backend.services.reporting import to_json


# ---------------------------------------------------------------- helpers

def _reject_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")


def _strict_loads(raw: bytes):
    """Parse JSON, rejecting NaN/Infinity constants outright."""
    return json.loads(raw.decode(), parse_constant=_reject_constant)


def _find_nonfinite(obj, path="$", out=None):
    if out is None:
        out = []
    if isinstance(obj, float) and not math.isfinite(obj):
        out.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _find_nonfinite(v, f"{path}.{k}", out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _find_nonfinite(v, f"{path}[{i}]", out)
    return out


def _assert_finite_report(report):
    bad = _find_nonfinite(report)
    assert bad == [], f"non-finite values in report: {bad}"


def _make_image(kind, size):
    if kind == "black":
        return Image.new("RGB", (size, size), (0, 0, 0))
    if kind == "white":
        return Image.new("RGB", (size, size), (255, 255, 255))
    if kind == "color":
        # V5.1.8.61 root-cause color: gray value not exactly representable
        return Image.new("RGB", (size, size), (40, 90, 140))
    if kind == "gradient":
        img = Image.new("RGB", (size, size))
        px = img.load()
        for x in range(size):
            v = round(x / max(1, size - 1) * 255)
            for y in range(size):
                px[x, y] = (v, v, v)
        return img
    if kind == "noise":
        rng = random.Random(20261024)
        img = Image.new("RGB", (size, size))
        img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                     for _ in range(size * size)])
        return img
    raise ValueError(kind)


def _png_bytes(kind, size):
    buf = io.BytesIO()
    _make_image(kind, size).save(buf, format="PNG")
    return buf.getvalue()


def _ai_artifacts_signal(report):
    return next(s for s in report["signals"] if s["name"] == "ai_artifacts")


def _signal(report, name):
    return next(s for s in report["signals"] if s["name"] == name)


# ---------------------------------------------------------------- clamp

def test_clamp_rejects_nonfinite():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            clamp(bad)
        with pytest.raises(ValueError):
            clamp(bad, 1, 99)


def test_clamp_finite_unchanged():
    assert clamp(50.0) == 50.0
    assert clamp(-5.0) == 0.0
    assert clamp(150.0) == 100.0
    assert clamp(99.9, 1, 99) == 99.0
    assert clamp(0.5, 1, 99) == 1.0


# ---------------------------------------------------------------- symmetry root cause

def test_mirror_symmetry_constant_color_is_unavailable_not_zero():
    # The exact V5.1.8.61 trigger: solid (40, 90, 140) at every size.
    for size in (64, 256, 1024):
        ctx = pipeline.build_context(f"color_{size}.png", "image/png", _png_bytes("color", size))
        analyzer = next(a for a in pipeline.ANALYZERS if a.name == "ai_artifacts")
        result = analyzer.run(ctx)
        assert result.technical["mirror_symmetry_corr"] is None
        assert any("unmeasurable" in w for w in result.warnings)
        assert result.ai_score == result.ai_score  # finite, no NaN score


def test_mirror_symmetry_constant_black_white_unavailable():
    for kind in ("black", "white"):
        ctx = pipeline.build_context(f"{kind}.png", "image/png", _png_bytes(kind, 256))
        analyzer = next(a for a in pipeline.ANALYZERS if a.name == "ai_artifacts")
        result = analyzer.run(ctx)
        # Correlation on a constant half is undefined, not zero.
        assert result.technical["mirror_symmetry_corr"] is None


def test_mirror_symmetry_nonconstant_is_finite():
    for kind in ("gradient", "noise"):
        ctx = pipeline.build_context(f"{kind}.png", "image/png", _png_bytes(kind, 256))
        analyzer = next(a for a in pipeline.ANALYZERS if a.name == "ai_artifacts")
        result = analyzer.run(ctx)
        sym = result.technical["mirror_symmetry_corr"]
        assert sym is None or (math.isfinite(sym) and -1.0 <= sym <= 1.0)


# ---------------------------------------------------------------- 15-cell matrix

@pytest.mark.parametrize("kind", ["black", "white", "color", "gradient", "noise"])
@pytest.mark.parametrize("size", [64, 256, 1024])
def test_degenerate_matrix_finite_and_strict(kind, size):
    report = pipeline.run_analysis(f"{kind}_{size}.png", "image/png", _png_bytes(kind, size))
    assert report["ensemble"]["verdict"] in (
        "LIKELY_AI_GENERATED", "LIKELY_MANIPULATED", "LIKELY_AUTHENTIC", "INCONCLUSIVE")
    _assert_finite_report(report)
    # Strict serialization must succeed (this is what used to 500).
    json.dumps(report, allow_nan=False)
    # No stack trace ever leaks through a signal payload.
    for s in report["signals"]:
        assert "traceback" not in s.get("technical", {})
    sym = _ai_artifacts_signal(report)["technical"]["mirror_symmetry_corr"]
    if kind in ("black", "white", "color"):
        assert sym is None
    else:
        assert sym is None or math.isfinite(sym)


def test_color_cells_previously_500ing_now_inconclusive():
    # The three cells that failed strict JSON in V5.1.8.61.
    for size in (64, 256, 1024):
        report = pipeline.run_analysis(f"color_{size}.png", "image/png", _png_bytes("color", size))
        assert report["ensemble"]["verdict"] == "INCONCLUSIVE"
        _assert_finite_report(report)


def test_1x1_image():
    report = pipeline.run_analysis("tiny.png", "image/png", _png_bytes("color", 1))
    assert report["ensemble"]["verdict"] == "INCONCLUSIVE"
    assert report["ensemble"]["insufficient_evidence"] is True
    _assert_finite_report(report)
    json.dumps(report, allow_nan=False)


def test_flat_jpeg_and_webp():
    for fmt, mime in (("JPEG", "image/jpeg"), ("WEBP", "image/webp")):
        buf = io.BytesIO()
        _make_image("color", 256).save(buf, format=fmt, quality=95)
        report = pipeline.run_analysis(f"flat.{fmt.lower()}", mime, buf.getvalue())
        _assert_finite_report(report)
        json.dumps(report, allow_nan=False)


# ---------------------------------------------------------------- API responses

@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from backend.api import routes
    from backend.main import app

    routes._RATE.clear()
    return TestClient(app)


def test_api_color_png_strict_body(client):
    # The exact input that 500'd in V5.1.8.60/61 through the API path.
    resp = client.post("/api/analyze?save=false", files={"file": ("color.png", _png_bytes("color", 256), "image/png")})
    assert resp.status_code == 200
    body = _strict_loads(resp.content)
    assert body["ensemble"]["verdict"] == "INCONCLUSIVE"
    assert "NaN" not in resp.text and "Infinity" not in resp.text
    _assert_finite_report(body)


def test_api_flat_jpeg_and_1x1_strict_body(client):
    buf = io.BytesIO()
    _make_image("color", 256).save(buf, format="JPEG", quality=95)
    resp = client.post("/api/analyze?save=false", files={"file": ("flat.jpg", buf.getvalue(), "image/jpeg")})
    assert resp.status_code == 200
    _strict_loads(resp.content)
    assert "NaN" not in resp.text

    resp = client.post("/api/analyze?save=false", files={"file": ("tiny.png", _png_bytes("color", 1), "image/png")})
    assert resp.status_code == 200
    body = _strict_loads(resp.content)
    assert body["ensemble"]["verdict"] == "INCONCLUSIVE"


# ---------------------------------------------------------------- injected non-finite values

def test_injected_nan_score_becomes_unable(monkeypatch):
    target = next(a for a in pipeline.ANALYZERS if a.name == "noise")
    orig = target.analyze

    def bad(ctx):
        r = orig(ctx)
        r.ai_score = float("nan")
        return r

    monkeypatch.setattr(target, "analyze", bad)
    report = pipeline.run_analysis("noise.png", "image/png", _png_bytes("noise", 256))
    sig = _signal(report, "noise")
    assert sig["status"] == "unable"
    assert "error" in sig["technical"]
    assert "traceback" not in sig["technical"]
    _assert_finite_report(report)
    json.dumps(report, allow_nan=False)


def test_injected_nan_diagnostic_becomes_null_with_warning(monkeypatch):
    target = next(a for a in pipeline.ANALYZERS if a.name == "noise")
    orig = target.analyze

    def bad(ctx):
        r = orig(ctx)
        r.technical["probe_nan"] = float("nan")
        return r

    monkeypatch.setattr(target, "analyze", bad)
    report = pipeline.run_analysis("noise.png", "image/png", _png_bytes("noise", 256))
    sig = _signal(report, "noise")
    assert sig["technical"]["probe_nan"] is None
    assert any("probe_nan" in w for w in sig["warnings"])
    _assert_finite_report(report)
    json.dumps(report, allow_nan=False)


def test_injected_nan_feature_disables_ml_not_verdict(monkeypatch):
    # NaN in a stacking feature must fail the detector (ERROR /
    # unavailable), not clamp to an extreme and not crash the pipeline.
    target = next(a for a in pipeline.ANALYZERS if a.name == "statistics")
    orig = target.analyze

    def bad(ctx):
        r = orig(ctx)
        r.technical["luminance_entropy_bits"] = float("nan")
        return r

    monkeypatch.setattr(target, "analyze", bad)
    report = pipeline.run_analysis("noise.png", "image/png", _png_bytes("noise", 256))
    _assert_finite_report(report)
    json.dumps(report, allow_nan=False)
    assert report["ensemble"]["ml_available"] is False
    assert report["ensemble"]["ml_ai_probability"] is None


def test_all_scores_nan_cannot_manufacture_confidence(monkeypatch):
    for a in pipeline.ANALYZERS:
        def bad(ctx, a=a):
            return AnalyzerResult(name=a.name, display_name=a.display_name,
                                  ai_score=float("nan"), manip_score=float("nan"),
                                  confidence=float("nan"))
        monkeypatch.setattr(a, "analyze", bad)
    report = pipeline.run_analysis("noise.png", "image/png", _png_bytes("noise", 256))
    assert all(s["status"] == "unable" for s in report["signals"])
    _assert_finite_report(report)
    json.dumps(report, allow_nan=False)
    # The verdict itself must not be confident: with every forensic
    # signal unable, no decisive support exists, so the ensemble stays
    # inconclusive even though the demonstration ML model still emits
    # a probability from default features (pre-existing behavior, and
    # the ML value alone cannot carry a verdict).
    assert report["ensemble"]["verdict"] == "INCONCLUSIVE"
    for key in ("ai_probability", "manipulation_probability", "authenticity_probability"):
        v = report["ensemble"][key]
        assert v is None or math.isfinite(v)


# ---------------------------------------------------------------- persistence

def test_save_rejects_nan_and_writes_nothing():
    report = pipeline.run_analysis("noise.png", "image/png", _png_bytes("noise", 256))
    bad = copy.deepcopy(report)
    bad["signals"][0]["technical"]["probe"] = float("nan")
    with pytest.raises(ValueError):
        storage.save_analysis(bad)
    # No partial or malformed row: the redirected tmp DB is either absent
    # (never connected) or contains zero rows.
    import os

    if os.path.exists(storage.DB_PATH):
        conn = sqlite3.connect(storage.DB_PATH)
        try:
            assert conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0] == 0
        finally:
            conn.close()
    # Positive control: the valid report saves, and the stored JSON is strict.
    rid = storage.save_analysis(report)
    conn = sqlite3.connect(storage.DB_PATH)
    try:
        row = conn.execute("SELECT full_json FROM analyses WHERE id = ?", (rid,)).fetchone()
    finally:
        conn.close()
    assert row is not None
    _strict_loads(row[0].encode())


def test_reporting_to_json_strict():
    report = pipeline.run_analysis("noise.png", "image/png", _png_bytes("noise", 256))
    _strict_loads(to_json(report).encode())
    bad = copy.deepcopy(report)
    bad["ensemble"]["ai_probability"] = float("nan")
    with pytest.raises(ValueError):
        to_json(bad)


# ---------------------------------------------------------------- intact ordinary behavior

def test_ordinary_images_unchanged():
    # Pre-fix baselines captured 2026-10-24 (V5.1.8.62) before any edit.
    import os

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data")
    cases = [
        ("authentic_camera/canon_photo.jpg", "image/jpeg",
         "INCONCLUSIVE", 32.0, 47.7, 64.3, 47.3, 0.1),
        ("ai_generated/sd_output.png", "image/png",
         "LIKELY_AI_GENERATED", 65.5, 48.0, 46.5, 41.7, 10.9),
    ]
    for rel, mime, verdict, ai, manip, auth, conf, ml in cases:
        with open(os.path.join(root, rel), "rb") as fh:
            data = fh.read()
        report = pipeline.run_analysis(os.path.basename(rel), mime, data)
        ens = report["ensemble"]
        assert ens["verdict"] == verdict
        assert ens["ai_probability"] == ai
        assert ens["manipulation_probability"] == manip
        assert ens["authenticity_probability"] == auth
        assert ens["confidence"] == conf
        assert ens["ml_ai_probability"] == ml
        _assert_finite_report(report)
