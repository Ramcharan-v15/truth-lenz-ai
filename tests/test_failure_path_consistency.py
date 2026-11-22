"""Failure-path consistency regression tests (V5.1.8.64).

Covers the V5.1.8.63 residuals, now implemented:
- compare / robustness / blind-reveal handling of null verdict
  probabilities (null stays null, never zero, explicitly explained);
- ML suppression when no forensic signal is usable (ML unavailable,
  probability null, reason recorded, raw detector entry preserved).

All test-generated state is isolated by tests/conftest.py. The single
exception is blind reveal's leakage report, whose path is hardcoded to
validation/results/ in the implementation: these tests delete their own
leakage report afterwards (distinctive EXP-TEST-64-* ids) so the working
tree is left unchanged.
"""
import copy
import io
import json
import os
import random

import pytest
from PIL import Image

from backend.analyzers.base import AnalyzerResult
from backend.services import pipeline


# ---------------------------------------------------------------- helpers

def _make_image(kind, size):
    if kind == "color":
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


def _fallback_shaped(report):
    """A report as the numerical fallback produces it: real signals, but
    verdict probabilities withheld (null)."""
    bad = copy.deepcopy(report)
    bad["ensemble"] = {
        "verdict": "INCONCLUSIVE",
        "ai_probability": None,
        "manipulation_probability": None,
        "authenticity_probability": None,
        "confidence": 0.0,
        "ml_ai_probability": None,
        "ml_available": False,
    }
    return bad


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from backend.api import routes
    from backend.main import app

    routes._RATE.clear()
    return TestClient(app)


def _strict_loads(raw: bytes):
    def _reject(value):
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(raw.decode(), parse_constant=_reject)


# ---------------------------------------------------------------- compare

def _compare_via_api(client, monkeypatch, null_sides):
    """Run POST /api/compare with run_analysis wrapped so the requested
    sides (0 = A, 1 = B) return fallback-shaped reports."""
    from backend.api import routes

    calls = {"n": 0}
    real = pipeline.run_analysis

    def wrapped(filename, mime, data, **kwargs):
        rep = real(filename, mime, data, **kwargs)
        if calls["n"] in null_sides:
            rep = _fallback_shaped(rep)
        calls["n"] += 1
        return rep

    monkeypatch.setattr(routes.pipeline, "run_analysis", wrapped)
    return client.post("/api/compare", files={
        "file_a": ("a.png", _png_bytes("gradient", 256), "image/png"),
        "file_b": ("b.png", _png_bytes("noise", 256), "image/png"),
    })


def test_compare_control_numeric_deltas(client):
    # Canon vs synth intact baselines (V5.1.8.62): deltas stay numeric.
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "test_data")
    with open(os.path.join(root, "authentic_camera/canon_photo.jpg"), "rb") as fh:
        a = fh.read()
    with open(os.path.join(root, "ai_generated/sd_output.png"), "rb") as fh:
        b = fh.read()
    resp = client.post("/api/compare", files={
        "file_a": ("canon.jpg", a, "image/jpeg"),
        "file_b": ("synth.png", b, "image/png")})
    assert resp.status_code == 200
    comp = resp.json()["comparison"]
    assert comp["ai_delta"] == round(32.0 - 65.5, 1)
    assert comp["manip_delta"] == round(47.7 - 48.0, 1)
    assert comp["comparison_note"] is None


def test_compare_one_sided_null(client, monkeypatch):
    resp = _compare_via_api(client, monkeypatch, null_sides={1})
    assert resp.status_code == 200
    body = _strict_loads(resp.content)
    comp = body["comparison"]
    # Null stays null: never zero, never a crash.
    assert comp["ai_delta"] is None
    assert comp["manip_delta"] is None
    assert comp["ai_delta"] != 0 and comp["manip_delta"] != 0
    # A outcome preserved as its real (confident) verdict; B withheld.
    assert comp["verdict_a"] == "LIKELY_MANIPULATED"
    assert comp["verdict_b"] == "INCONCLUSIVE"
    assert comp["comparison_note"] and "unavailable" in comp["comparison_note"]
    # Both image outcomes are still present in full.
    assert body["image_a"]["ensemble"]["ai_probability"] is not None
    assert body["image_b"]["ensemble"]["ai_probability"] is None


def test_compare_two_sided_null(client, monkeypatch):
    resp = _compare_via_api(client, monkeypatch, null_sides={0, 1})
    assert resp.status_code == 200
    comp = resp.json()["comparison"]
    assert comp["ai_delta"] is None
    assert comp["manip_delta"] is None
    assert comp["verdict_a"] == "INCONCLUSIVE"
    assert comp["verdict_b"] == "INCONCLUSIVE"
    assert comp["comparison_note"] and "not zero" in comp["comparison_note"]


# ---------------------------------------------------------------- robustness

def _robust_run_fn(mode):
    """Fake run_fn: real analysis, with fallback shaping per mode.
    mode: 'control' | 'base_null' | 'transform_null'."""
    calls = {"n": 0}

    def fn(filename, mime, data):
        rep = pipeline.run_analysis(filename, mime, data)
        n = calls["n"]
        calls["n"] += 1
        if mode == "base_null" and n == 0:
            return _fallback_shaped(rep)
        if mode == "transform_null" and n == 2:
            return _fallback_shaped(rep)
        return rep

    return fn


def test_robustness_control():
    from backend.benchmark.robustness import TRANSFORMS, evaluate_robustness

    out = evaluate_robustness(_robust_run_fn("control"), _png_bytes("noise", 256),
                              "noise.png", transforms=TRANSFORMS[:2])
    assert out["overall_robustness_score"] is not None
    assert out["overall_robustness_score"] >= 0
    assert out["transforms_measured"] == 2
    assert out["transforms_unavailable"] == 0
    assert out["numerical_evidence"] == "available"


def test_robustness_base_null_is_unavailable_not_brittle():
    from backend.benchmark.robustness import TRANSFORMS, evaluate_robustness

    out = evaluate_robustness(_robust_run_fn("base_null"), _png_bytes("noise", 256),
                              "noise.png", transforms=TRANSFORMS[:2])
    # Not a 0.0 "brittle" score and not an error: explicitly unavailable.
    assert out["overall_robustness_score"] is None
    assert out["transforms"] == []
    assert out["numerical_evidence"] == "unavailable"
    assert "withheld" in out["unavailable_reason"]
    assert out["original"]["ai_probability"] is None


def test_robustness_transform_null_excluded_from_aggregate():
    from backend.benchmark.robustness import TRANSFORMS, evaluate_robustness

    out = evaluate_robustness(_robust_run_fn("transform_null"), _png_bytes("noise", 256),
                              "noise.png", transforms=TRANSFORMS[:2])
    assert len(out["transforms"]) == 2
    measured, unavailable = out["transforms"]
    assert measured["robustness_score"] is not None
    assert unavailable["robustness_score"] is None
    assert unavailable["delta_ai"] is None
    assert unavailable["delta_manipulation"] is None
    assert unavailable["numerical_evidence"] == "unavailable"
    # Aggregate uses measured rows only; no 0.0 invented for the gap.
    assert out["overall_robustness_score"] == measured["robustness_score"]
    assert out["transforms_measured"] == 1
    assert out["transforms_unavailable"] == 1
    assert out["numerical_evidence"] == "partial"


# ---------------------------------------------------------------- blind reveal

def _craft_experiment(eid, predictions):
    from backend.validation import _save_record

    rec = {
        "experiment_id": eid,
        "kind": "blind",
        "status": "COMPLETED",
        "status_history": [],
        "reproducibility": {"dataset_id": "test", "manifest_path": ""},
        "predictions": predictions,
        "n_predictions": len(predictions),
    }
    _save_record(rec)
    return rec


def _leak_path(eid):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "validation", "results", f"{eid}_LEAKAGE_REPORT.json")


def test_reveal_marks_withheld_predictions():
    from backend.validation.blind_protocol import load_experiment, reveal_and_score

    eid = "EXP-TEST-64-WITHHELD"
    _craft_experiment(eid, [
        {"image_id": "img_a", "filename": "a.png", "sha256": "a" * 64,
         "verdict": "LIKELY_AI_GENERATED", "ai_probability": 85.0,
         "manipulation_probability": 10.0, "confidence": 70.0,
         "ood_status": "UNKNOWN", "model_probability_ai": 80.0, "case_id": "c1"},
        {"image_id": "img_b", "filename": "b.png", "sha256": "b" * 64,
         "verdict": "INCONCLUSIVE", "ai_probability": None,
         "manipulation_probability": None, "confidence": 0.0,
         "ood_status": "UNKNOWN", "model_probability_ai": None, "case_id": "c2"},
        {"image_id": "img_c", "filename": "c.png", "sha256": "c" * 64,
         "verdict": "LIKELY_AUTHENTIC", "ai_probability": 20.0,
         "manipulation_probability": 15.0, "confidence": 70.0,
         "ood_status": "UNKNOWN", "model_probability_ai": 25.0, "case_id": "c3"},
    ])
    try:
        out = reveal_and_score(eid, {"img_a": "ai_generated", "img_b": "authentic",
                                     "img_c": "authentic"})
        metrics = out["metrics"]
        assert metrics["n_predictions"] == 3
        assert metrics["n_scored"] == 3
        assert metrics["n_withheld_probability"] == 1
        assert metrics["withheld_image_ids"] == ["img_b"]
        assert metrics["withheld_reason"] and "withheld" in metrics["withheld_reason"]
        # Binary metrics use only the two probability-bearing predictions.
        assert metrics["binary"].get("status") != "INSUFFICIENT_SAMPLE"
        # Verdict counts preserved: img_b still counts as inconclusive.
        assert metrics["confusion"]["authentic"]["inconclusive"] == 1
        # Stored record not corrupted: the withheld prediction keeps its null.
        rec = load_experiment(eid)
        withheld = next(p for p in rec["predictions"] if p["image_id"] == "img_b")
        assert withheld["ai_probability"] is None
    finally:
        if os.path.exists(_leak_path(eid)):
            os.remove(_leak_path(eid))


def test_reveal_decisive_withheld_probability_does_not_crash():
    from backend.validation.blind_protocol import reveal_and_score

    eid = "EXP-TEST-64-DECISIVE-NULL"
    _craft_experiment(eid, [
        {"image_id": "img_x", "filename": "x.png", "sha256": "d" * 64,
         "verdict": "LIKELY_AI_GENERATED", "ai_probability": None,
         "manipulation_probability": None, "confidence": 0.0,
         "ood_status": "UNKNOWN", "model_probability_ai": None, "case_id": "c9"},
        {"image_id": "img_y", "filename": "y.png", "sha256": "e" * 64,
         "verdict": "LIKELY_AUTHENTIC", "ai_probability": 20.0,
         "manipulation_probability": 15.0, "confidence": 70.0,
         "ood_status": "UNKNOWN", "model_probability_ai": 25.0, "case_id": "c8"},
    ])
    try:
        # Old code raised TypeError on None / 100.0 here.
        out = reveal_and_score(eid, {"img_x": "ai_generated", "img_y": "authentic"})
        metrics = out["metrics"]
        assert metrics["n_withheld_probability"] == 1
        assert metrics["withheld_image_ids"] == ["img_x"]
        # No fabricated score and no crash: with only one probability-bearing
        # prediction, binary metrics are insufficient by design (not by error).
        assert metrics["binary"]["status"] == "INSUFFICIENT_SAMPLE"
    finally:
        if os.path.exists(_leak_path(eid)):
            os.remove(_leak_path(eid))


# ---------------------------------------------------------------- ML suppression

def _inject_nan_scores(monkeypatch, names):
    for a in pipeline.ANALYZERS:
        if a.name not in names:
            continue
        orig = a.analyze

        def bad(ctx, orig=orig):
            r = orig(ctx)
            r.ai_score = float("nan")
            r.manip_score = float("nan")
            r.confidence = float("nan")
            return r

        monkeypatch.setattr(a, "analyze", bad)


def test_ml_suppressed_when_no_usable_signals(monkeypatch):
    _inject_nan_scores(monkeypatch, {a.name for a in pipeline.ANALYZERS})
    report = pipeline.run_analysis("noise.png", "image/png", _png_bytes("noise", 256))
    ens = report["ensemble"]
    assert all(s["status"] == "unable" for s in report["signals"])
    # Usable probability withheld...
    assert ens["ml_available"] is False
    assert ens["ml_ai_probability"] is None
    assert ens["ml_suppressed_reason"] and "No usable forensic signals" in ens["ml_suppressed_reason"]
    assert ens["verdict"] == "INCONCLUSIVE"
    # ...and no ML percentage in the verdict-facing explanation...
    assert "ML detector P(AI)" not in report["explanations"]["technical"]
    # ...while the forensic explanation names the withholding explicitly...
    assert "ML probability withheld" in report["explanations"]["forensic"]
    # ...and the raw detector execution is still recorded as provenance.
    dets = [d for d in ens["detectors"] if d.get("probability_ai") is not None]
    assert dets, "raw detector result must remain recorded"
    assert dets[0]["model_status"] == "ready"


def test_ml_not_suppressed_when_some_signals_usable(monkeypatch):
    _inject_nan_scores(monkeypatch, {"noise"})
    report = pipeline.run_analysis("noise.png", "image/png", _png_bytes("noise", 256))
    ens = report["ensemble"]
    assert ens["ml_available"] is True
    assert ens["ml_ai_probability"] is not None
    assert ens["ml_suppressed_reason"] is None


def test_ml_baselines_unchanged_with_usable_signals():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "test_data")
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
        assert ens["ml_available"] is True
        assert ens["ml_ai_probability"] == ml
        assert ens["ml_suppressed_reason"] is None
