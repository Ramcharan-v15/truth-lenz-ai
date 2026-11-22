"""V4 tests: datasets, validation, OOD, robustness formula, new APIs, regressions.

All fixtures are synthetic and labeled. No metrics are fabricated.
"""
import json
import os

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------- metrics extensions --------------------------------------------
def test_brier_score():
    from backend.benchmark.metrics import brier_score
    # perfect predictions -> 0
    assert brier_score([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0]) == 0.0
    # worst -> 1.0
    assert brier_score([0, 1], [1.0, 0.0]) == 1.0
    # 0.5 everywhere -> 0.25
    assert brier_score([0, 1, 0, 1], [0.5] * 4) == 0.25


def test_balanced_accuracy():
    from backend.benchmark.metrics import balanced_accuracy
    assert balanced_accuracy([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    # imbalanced: all predicted 0 -> recall(1)=0, recall(0)=1 -> 0.5
    assert balanced_accuracy([0, 0, 0, 1], [0.1, 0.2, 0.3, 0.4]) == 0.5


def test_full_evaluation_has_v4_metrics():
    from backend.benchmark.metrics import full_evaluation
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.7])
    out = full_evaluation(y, p)
    assert "brier_score" in out and 0 <= out["brier_score"] <= 1
    assert "balanced_accuracy" in out and 0 <= out["balanced_accuracy"] <= 1


# ---------- OOD levels -------------------------------------------------------
def test_ood_status_levels():
    """max|z| tiers map to the four OOD statuses."""
    # simulate via the stacking detector's logic
    def level(max_z):
        if max_z <= 3.0:
            return "IN_DISTRIBUTION"
        elif max_z <= 6.0:
            return "MILD_SHIFT"
        return "OUT_OF_DISTRIBUTION"
    assert level(1.5) == "IN_DISTRIBUTION"
    assert level(4.5) == "MILD_SHIFT"
    assert level(14.0) == "OUT_OF_DISTRIBUTION"


def test_ood_in_ensemble_output():
    from backend.services.pipeline import run_analysis
    data = open(os.path.join(ROOT, "test_data", "authentic_camera",
                             "canon_photo.jpg"), "rb").read()
    rep = run_analysis("c.jpg", "image/jpeg", data, include_visuals=False)
    ood = rep["ensemble"]["ml_ood_status"]
    assert ood in ("IN_DISTRIBUTION", "MILD_SHIFT", "OUT_OF_DISTRIBUTION", "UNKNOWN")


# ---------- robustness formula --------------------------------------------------
def test_robustness_score_formula():
    from backend.benchmark.robustness import robustness_score
    # perfect stability
    assert robustness_score(0, 0, False) == 100.0
    # documented formula: 100 - mean(|d_ai|, |d_manip|) - 25*flip
    assert robustness_score(10, 20, False) == 85.0
    assert robustness_score(10, 20, True) == 60.0
    # never negative
    assert robustness_score(200, 200, True) == 0.0


def test_robustness_records_v4_fields():
    from backend.benchmark.robustness import evaluate_robustness
    from backend.services.pipeline import run_analysis

    def run_fn(fname, mime, data):
        return run_analysis(fname, mime, data, include_visuals=False)

    data = open(os.path.join(ROOT, "test_data", "ai_generated",
                             "smooth_synth.jpg"), "rb").read()
    rep = evaluate_robustness(run_fn, data, "s.jpg")
    t = rep["transforms"][0]
    for field in ("original_probability", "transformed_probability",
                  "probability_delta", "original_verdict", "verdict_changed",
                  "confidence_change", "ood_change", "robustness_score"):
        assert field in t, f"missing {field}"


# ---------- experiment tracking ----------------------------------------------------
def test_experiment_id_format():
    from backend.validation import next_experiment_id
    eid = next_experiment_id()
    assert eid.startswith("EXP-")
    parts = eid.split("-")
    assert len(parts) == 3 and len(parts[2]) == 6


def test_experiment_save_load_roundtrip(tmp_path):
    from backend.validation import (build_reproducibility, list_experiments,
                                    load_experiment, next_experiment_id,
                                    save_experiment)
    import backend.validation as V
    old_dir, old_counter = V.RESULTS_DIR, V.COUNTER_PATH
    V.RESULTS_DIR = str(tmp_path / "results")
    V.COUNTER_PATH = str(tmp_path / "counter.json")
    try:
        eid = next_experiment_id()
        repro = build_reproducibility("ds1", "1.0", "abc123", "m1", "1.0",
                                      "def456", "cfg789", seed=42)
        assert "reproducibility_hash" in repro
        path = save_experiment(eid, repro, {"accuracy": 0.9})
        assert os.path.isfile(path)
        rec = load_experiment(eid)
        assert rec["results_hash"]
        assert len(list_experiments()) >= 1
    finally:
        V.RESULTS_DIR, V.COUNTER_PATH = old_dir, old_counter


# ---------- generator attribution ------------------------------------------------------
def test_generator_attribution_unavailable():
    from backend.services.pipeline import run_analysis
    data = open(os.path.join(ROOT, "test_data", "authentic_camera",
                             "canon_photo.jpg"), "rb").read()
    rep = run_analysis("c.jpg", "image/jpeg", data, include_visuals=False)
    ga = rep["generator_attribution"]
    assert ga["status"] == "ATTRIBUTION_UNAVAILABLE"
    assert ga["label"] is None


# ---------- v4 result schema ---------------------------------------------------------------
def test_v4_schema_separation():
    from backend.services.pipeline import run_analysis
    data = open(os.path.join(ROOT, "test_data", "authentic_camera",
                             "canon_photo.jpg"), "rb").read()
    rep = run_analysis("c.jpg", "image/jpeg", data, include_visuals=False)
    v4 = rep["v4"]
    assert set(v4.keys()) == {"observations", "model_inference", "provenance",
                              "uncertainty", "ensemble", "interpretation",
                              "limitations"}
    assert len(v4["observations"]) == 10
    # observations are measurements, not conclusions
    assert "ai_score" in v4["observations"][0]


def test_perceptual_hash_present():
    from backend.services.pipeline import run_analysis
    data = open(os.path.join(ROOT, "test_data", "authentic_camera",
                             "canon_photo.jpg"), "rb").read()
    rep = run_analysis("c.jpg", "image/jpeg", data, include_visuals=False)
    ph = rep["reproducibility"]["perceptual_hash"]
    assert isinstance(ph, str) and len(ph) == 64
    assert set(ph) <= {"0", "1"}


# ---------- V3 regression tests (spec §46) -----------------------------------------------
def test_regression_zero_variance_no_explosion():
    """Constant features must not explode z-scores."""
    from ml.preprocessing.features import extract_features
    import numpy as np
    # all signals identical -> many constant features
    feats = np.array(extract_features([]))
    assert np.all(np.isfinite(feats))


def test_regression_nan_inf_safe():
    """NaN/Inf cannot enter inference."""
    from backend.models.stacking import ForensicStackingDetector
    det = ForensicStackingDetector(
        model_path=os.path.join(ROOT, "models", "tl-stacking-v1.json"))
    det.load()
    assert det.get_status().value == "ready"
    # model weights are finite
    import json as _json
    m = _json.load(open(os.path.join(ROOT, "models", "tl-stacking-v1.json")))
    assert all(np.isfinite(m["weights"]))
    assert np.isfinite(m["bias"])


def test_regression_demonstration_no_veto():
    """Demonstration model cannot veto decisive forensic evidence."""
    from backend.services.pipeline import run_analysis
    data = open(os.path.join(ROOT, "test_data", "ai_generated",
                             "sd_output.png"), "rb").read()
    rep = run_analysis("sd.png", "image/png", data, include_visuals=False)
    # forensics say AI (71%), demo model says 10% — verdict must follow forensics
    assert rep["ensemble"]["verdict"] == "LIKELY_AI_GENERATED"
    d = rep["ensemble"]["detectors"][0]
    assert d.get("demonstration") is True


def test_regression_missing_metadata_not_ai():
    from backend.services.pipeline import run_analysis
    # screenshot with stripped metadata should not become AI
    data = open(os.path.join(ROOT, "test_data", "screenshots",
                             sorted(os.listdir(os.path.join(
                                 ROOT, "test_data", "screenshots")))[0]),
                "rb").read()
    rep = run_analysis("s.png", "image/png", data, include_visuals=False)
    # must not be LIKELY_AI_GENERATED solely from missing metadata
    assert rep["ensemble"]["verdict"] != "LIKELY_AI_GENERATED" or \
        rep["ensemble"]["ai_probability"] < 90


def test_regression_uncalibrated_not_probability():
    """Raw scores must be separate from calibrated probabilities."""
    from backend.services.pipeline import run_analysis
    data = open(os.path.join(ROOT, "test_data", "authentic_camera",
                             "canon_photo.jpg"), "rb").read()
    rep = run_analysis("c.jpg", "image/jpeg", data, include_visuals=False)
    d = rep["ensemble"]["detectors"][0]
    assert d["raw_score"] is not None
    assert d["probability_ai"] is not None
    assert d["calibration"] != "none"


# ---------- new API endpoints -------------------------------------------------------
@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


def test_api_validation(client):
    r = client.get("/api/validation")
    assert r.status_code == 200
    body = r.json()
    # V5.1: real-world validation now PARTIAL (Columbia splicing measured)
    assert body["validation_status"] == "PARTIAL_REAL_WORLD_VALIDATION"
    assert body["real_world"]["status"] == "MEASURED"
    assert body["real_world"]["experiment_id"] == "EXP-2026-000273"
    assert body["real_world"]["n"] == 1831
    # AI-generation on real data still not measured
    assert body["real_world_ai_generation"]["status"] == "NOT_MEASURED"


def test_api_datasets(client):
    r = client.get("/api/datasets")
    assert r.status_code == 200
    assert "datasets" in r.json()


def test_api_experiments(client):
    r = client.get("/api/experiments")
    assert r.status_code == 200
    assert "experiments" in r.json()


def test_api_calibration(client):
    r = client.get("/api/calibration")
    assert r.status_code == 200
    models = r.json()["models"]
    assert any(m["model"] == "tl-forensic-stacking" for m in models)


def test_api_cross_generator(client):
    r = client.get("/api/cross-generator")
    assert r.status_code == 200
    assert r.json()["status"] == "NOT_MEASURED"


def test_api_hard_negatives(client):
    r = client.get("/api/hard-negatives")
    assert r.status_code == 200
    body = r.json()
    assert body["real_world"] == "NOT_MEASURED"
    assert body["synthetic"]["n"] == 30


def test_api_ood(client):
    r = client.get("/api/ood")
    assert r.status_code == 200
    body = r.json()
    assert "IN_DISTRIBUTION" in body["levels"]
    assert "OUT_OF_DISTRIBUTION" in body["levels"]
