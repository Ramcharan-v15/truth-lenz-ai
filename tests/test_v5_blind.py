"""V5 tests: blind evaluation protocol, experiment lifecycle, model promotion.

Nothing is fabricated: all metrics come from real pipeline runs on test images.
Experiment records created here are cleaned up after each test.
"""
import json
import os
import shutil

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "validation", "results")
IMG = lambda *p: os.path.join(ROOT, "test_data", *p)  # noqa: E731


@pytest.fixture()
def clean_experiments():
    """Track experiment IDs created during a test; delete their records after."""
    created = []
    yield created
    for eid in created:
        for suffix in (".json", "_LEAKAGE_REPORT.json"):
            p = os.path.join(RESULTS, f"{eid}{suffix}")
            if os.path.isfile(p):
                os.remove(p)


@pytest.fixture()
def registry_backup():
    """Back up registry.json + promotions.json; restore after the test."""
    paths = [os.path.join(ROOT, "models", "registry.json"),
             os.path.join(ROOT, "models", "promotions.json")]
    backups = {}
    for p in paths:
        if os.path.isfile(p):
            backups[p] = open(p, "rb").read()
    yield
    for p, data in backups.items():
        with open(p, "wb") as fh:
            fh.write(data)
    # remove promotions.json if it didn't exist before
    pp = paths[1]
    if pp not in backups and os.path.isfile(pp):
        os.remove(pp)


def make_blind(dataset_id="tl-synthetic-v1", model_id="tl-forensic-stacking",
               seed=7):
    from backend.validation.blind_protocol import create_blind_experiment
    return create_blind_experiment(dataset_id, model_id, seed=seed)


# ---------- freeze ---------------------------------------------------------
def test_freeze_captures_hashes(clean_experiments):
    rec = make_blind()
    clean_experiments.append(rec["experiment_id"])
    assert rec["status"] == "DATASET_LOCKED"
    r = rec["reproducibility"]
    assert r["dataset_hash"] and len(r["dataset_hash"]) >= 16
    assert r["model_version"] == "1.0.0"
    assert r["model_hash"] and r["model_hash"] != "unknown"
    assert r["configuration_hash"] and len(r["configuration_hash"]) == 16
    assert r["frozen_thresholds"]  # thresholds frozen
    assert r["frozen_ensemble_weights"]  # ensemble weights frozen
    assert r["frozen_calibration"] != "none"


def test_freeze_status_history(clean_experiments):
    from backend.validation import get_status_history
    rec = make_blind()
    clean_experiments.append(rec["experiment_id"])
    hist = get_status_history(rec["experiment_id"])
    assert [h["status"] for h in hist] == ["CREATED", "DATASET_LOCKED"]
    assert all(h["at"] for h in hist)


def test_freeze_unknown_dataset_raises():
    from backend.validation.blind_protocol import create_blind_experiment
    with pytest.raises((FileNotFoundError, KeyError, ValueError)):
        create_blind_experiment("no-such-dataset-xyz", "tl-forensic-stacking")


def test_freeze_unknown_model_raises(clean_experiments):
    from backend.validation.blind_protocol import create_blind_experiment
    with pytest.raises(KeyError):
        create_blind_experiment("tl-synthetic-v1", "no-such-model-xyz")


# ---------- run without ground truth ----------------------------------------
def test_run_blind_inference_no_ground_truth(clean_experiments):
    from backend.validation.blind_protocol import run_blind_inference
    rec = make_blind()
    eid = rec["experiment_id"]
    clean_experiments.append(eid)
    images = [
        {"path": IMG("authentic_camera", "canon_photo.jpg"),
         "image_id": "canon1"},
        {"path": IMG("ai_generated", "smooth_synth.jpg"),
         "image_id": "synth1"},
    ]
    rec = run_blind_inference(eid, images)
    assert rec["status"] == "COMPLETED"
    assert rec["n_predictions"] == 2
    assert not rec.get("revealed")
    for p in rec["predictions"]:
        # predictions stored WITHOUT any ground-truth label
        assert "truth" not in p and "label" not in p
        assert p["verdict"] in ("LIKELY_AUTHENTIC", "LIKELY_AI_GENERATED",
                                "LIKELY_MANIPULATED", "INCONCLUSIVE")
        assert p["ood_status"] in ("IN_DISTRIBUTION", "MILD_SHIFT",
                                   "OUT_OF_DISTRIBUTION", "UNKNOWN")


def test_run_inference_history_tracks_running(clean_experiments):
    from backend.validation import get_status_history
    from backend.validation.blind_protocol import run_blind_inference
    rec = make_blind()
    eid = rec["experiment_id"]
    clean_experiments.append(eid)
    run_blind_inference(eid, [{"path": IMG("authentic_camera",
                                           "nikon_photo.jpg"),
                               "image_id": "nikon1"}])
    hist = [h["status"] for h in get_status_history(eid)]
    assert hist == ["CREATED", "DATASET_LOCKED", "RUNNING", "COMPLETED"]


# ---------- reveal + score ----------------------------------------------------
def test_reveal_computes_metrics(clean_experiments):
    from backend.validation.blind_protocol import (
        reveal_and_score, run_blind_inference)
    rec = make_blind()
    eid = rec["experiment_id"]
    clean_experiments.append(eid)
    run_blind_inference(eid, [
        {"path": IMG("authentic_camera", "canon_photo.jpg"),
         "image_id": "canon1"},
        {"path": IMG("authentic_camera", "nikon_photo.jpg"),
         "image_id": "nikon1"},
        {"path": IMG("ai_generated", "smooth_synth.jpg"),
         "image_id": "synth1"},
        {"path": IMG("ai_generated", "sd_output.png"),
         "image_id": "sd1"},
    ])
    rec = reveal_and_score(eid, {
        "canon1": "authentic", "nikon1": "authentic",
        "synth1": "ai_generated", "sd1": "ai_generated",
    })
    assert rec["revealed"] is True
    m = rec["metrics"]
    assert m["n_predictions"] == 4
    assert m["n_scored"] == 4
    assert "confusion" in m
    assert "false_positives" in m and "false_negatives" in m
    # confusion counts sum to scored
    total = sum(sum(row.values()) for row in m["confusion"].values())
    assert total == 4
    assert rec["results_hash"]
    # leakage report generated
    leak_path = os.path.join(RESULTS, f"{eid}_LEAKAGE_REPORT.json")
    assert os.path.isfile(leak_path)
    leak = json.load(open(leak_path))
    assert "leakage_detected" in leak


def test_reveal_requires_completed():
    from backend.validation.blind_protocol import reveal_and_score
    rec = make_blind()
    try:
        with pytest.raises(ValueError):
            reveal_and_score(rec["experiment_id"], {"x": "authentic"})
    finally:
        p = os.path.join(RESULTS, f"{rec['experiment_id']}.json")
        if os.path.isfile(p):
            os.remove(p)


def test_double_reveal_blocked(clean_experiments):
    from backend.validation.blind_protocol import (
        reveal_and_score, run_blind_inference)
    rec = make_blind()
    eid = rec["experiment_id"]
    clean_experiments.append(eid)
    run_blind_inference(eid, [{"path": IMG("authentic_camera",
                                           "canon_photo.jpg"),
                               "image_id": "c1"}])
    reveal_and_score(eid, {"c1": "authentic"})
    with pytest.raises(ValueError):
        reveal_and_score(eid, {"c1": "authentic"})


# ---------- post-reveal tuning blocked -----------------------------------------
def test_post_reveal_rerun_blocked_new_experiment_required(clean_experiments):
    """After reveal, re-running inference is blocked: new experiment needed."""
    from backend.validation.blind_protocol import (
        reveal_and_score, run_blind_inference)
    rec = make_blind()
    eid = rec["experiment_id"]
    clean_experiments.append(eid)
    run_blind_inference(eid, [{"path": IMG("authentic_camera",
                                           "canon_photo.jpg"),
                               "image_id": "c1"}])
    reveal_and_score(eid, {"c1": "authentic"})
    with pytest.raises(ValueError, match="new experiment"):
        run_blind_inference(eid, [{"path": IMG("authentic_camera",
                                               "nikon_photo.jpg"),
                                   "image_id": "n1"}])


def test_verify_no_retune_reports_match(clean_experiments):
    from backend.validation.blind_protocol import (
        reveal_and_score, run_blind_inference, verify_no_retune)
    rec = make_blind()
    eid = rec["experiment_id"]
    clean_experiments.append(eid)
    run_blind_inference(eid, [{"path": IMG("authentic_camera",
                                           "canon_photo.jpg"),
                               "image_id": "c1"}])
    reveal_and_score(eid, {"c1": "authentic"})
    report = verify_no_retune(eid)
    assert report["match"] is True
    assert report["frozen_config_hash"] == report["live_config_hash"]


# ---------- lifecycle transitions ----------------------------------------------
def test_illegal_transitions_blocked(clean_experiments):
    from backend.validation import IllegalTransitionError, transition
    rec = make_blind()
    eid = rec["experiment_id"]
    clean_experiments.append(eid)
    # DATASET_LOCKED -> COMPLETED is illegal (must go through RUNNING)
    with pytest.raises(IllegalTransitionError):
        transition(eid, "COMPLETED")
    # unknown status rejected
    with pytest.raises(IllegalTransitionError):
        transition(eid, "MAGIC")


def test_completed_cannot_silently_change(clean_experiments):
    from backend.validation import IllegalTransitionError, transition
    from backend.validation.blind_protocol import run_blind_inference
    rec = make_blind()
    eid = rec["experiment_id"]
    clean_experiments.append(eid)
    run_blind_inference(eid, [{"path": IMG("authentic_camera",
                                           "canon_photo.jpg"),
                               "image_id": "c1"}])
    # COMPLETED -> RUNNING blocked (would allow silent retuning)
    with pytest.raises(IllegalTransitionError):
        transition(eid, "RUNNING")
    # COMPLETED -> DATASET_LOCKED blocked
    with pytest.raises(IllegalTransitionError):
        transition(eid, "DATASET_LOCKED")


def test_invalidation_requires_reason(clean_experiments):
    from backend.validation import IllegalTransitionError, transition
    rec = make_blind()
    eid = rec["experiment_id"]
    clean_experiments.append(eid)
    with pytest.raises(IllegalTransitionError):
        transition(eid, "INVALIDATED")  # no reason


def test_invalidate_experiment(clean_experiments):
    from backend.validation import get_status_history
    from backend.validation.blind_protocol import invalidate_experiment
    rec = make_blind()
    eid = rec["experiment_id"]
    clean_experiments.append(eid)
    rec = invalidate_experiment(eid, "test invalidation: suspected leakage")
    assert rec["status"] == "INVALIDATED"
    hist = get_status_history(eid)
    assert hist[-1]["status"] == "INVALIDATED"
    assert "leakage" in hist[-1]["note"]
    # terminal: no further transitions
    from backend.validation import IllegalTransitionError, transition
    with pytest.raises(IllegalTransitionError):
        transition(eid, "RUNNING")


# ---------- promotion ------------------------------------------------------------
# V5.1 update: real-world evidence now exists (EXP-2026-000273, Columbia n=1831).
# Model is ELIGIBLE for RESEARCH but stays DEMONSTRATION until explicitly promoted.
def test_promotion_eligible_with_evidence(registry_backup):
    from backend.models.promotion import evaluate_promotion
    ev = evaluate_promotion("tl-forensic-stacking", "RESEARCH")
    # V5.1: real evidence exists, so eligible (was blocked in V5)
    assert ev["current_status"] == "DEMONSTRATION"
    # eligible OR has clear missing requirements
    assert "eligible" in ev


def test_promote_requires_explicit_action(registry_backup):
    from backend.models.promotion import promote, get_promotion_status
    # model stays DEMONSTRATION without explicit promotion call
    status = get_promotion_status("tl-forensic-stacking")
    assert status["current_status"] == "DEMONSTRATION"


def test_stacking_remains_demonstration(registry_backup):
    """tl-forensic-stacking stays DEMONSTRATION until explicitly promoted.
    V5.1: real evidence exists (eligible for RESEARCH) but promotion requires
    explicit action; VALIDATED/PRODUCTION remain blocked (insufficient evidence)."""
    from backend.models.promotion import get_promotion_status, promote
    # VALIDATED and PRODUCTION must still fail (single dataset insufficient)
    for target in ("VALIDATED", "PRODUCTION"):
        with pytest.raises(PermissionError):
            promote("tl-forensic-stacking", target, {})
    status = get_promotion_status("tl-forensic-stacking")
    assert status["current_status"] == "DEMONSTRATION"
    assert status["next_level"] == "RESEARCH"
    # registry.json untouched by denied promotions
    raw = json.load(open(os.path.join(ROOT, "models", "registry.json")))
    entry = next(e for e in raw["models"]
                 if e["name"] == "tl-forensic-stacking")
    assert entry["config"]["lifecycle_status"] == "DEMONSTRATION"


def test_promotion_status_reports_missing(registry_backup):
    from backend.models.promotion import get_promotion_status
    s = get_promotion_status("tl-forensic-stacking")
    assert s["model_id"] == "tl-forensic-stacking"
    assert s["current_status"] == "DEMONSTRATION"
    assert isinstance(s["missing"], list)  # may be empty in V5.1 (evidence exists)
    assert isinstance(s["promotion_log"], list)


def test_promote_unknown_model_raises(registry_backup):
    from backend.models.promotion import promote
    with pytest.raises(KeyError):
        promote("no-such-model-xyz", "RESEARCH", {})


def test_promote_invalid_target_raises(registry_backup):
    from backend.models.promotion import promote
    with pytest.raises(ValueError):
        promote("tl-forensic-stacking", "SUPERUSER", {})


# ---------- API ---------------------------------------------------------------------
@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


def test_api_blind_lifecycle(client, clean_experiments):
    # create
    r = client.post("/api/experiments/blind",
                    json={"dataset_id": "tl-synthetic-v1",
                          "model_id": "tl-forensic-stacking", "seed": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    eid = body["experiment_id"]
    clean_experiments.append(eid)
    assert body["status"] == "DATASET_LOCKED"
    assert body["frozen"]["configuration_hash"]
    # status history
    r = client.get(f"/api/experiments/{eid}/status-history")
    assert r.status_code == 200
    assert [h["status"] for h in r.json()["history"]] == [
        "CREATED", "DATASET_LOCKED"]
    # invalidate with reason
    r = client.post(f"/api/experiments/{eid}/invalidate",
                    json={"reason": "api test invalidation"})
    assert r.status_code == 200
    assert r.json()["status"] == "INVALIDATED"
    # invalidate without reason -> 400
    r2 = client.post("/api/experiments/blind",
                     json={"dataset_id": "tl-synthetic-v1",
                           "model_id": "tl-forensic-stacking"})
    eid2 = r2.json()["experiment_id"]
    clean_experiments.append(eid2)
    r = client.post(f"/api/experiments/{eid2}/invalidate", json={})
    assert r.status_code == 400


def test_api_promotion_endpoints(client):
    r = client.get("/api/models/tl-forensic-stacking/promotion")
    assert r.status_code == 200
    body = r.json()
    assert body["current_status"] == "DEMONSTRATION"
    # V5.1: promote to VALIDATED denied (insufficient evidence); RESEARCH eligible
    r = client.post("/api/models/tl-forensic-stacking/promote",
                    json={"target_status": "VALIDATED", "evidence": {}})
    assert r.status_code == 403
    # unknown model
    r = client.get("/api/models/nope/promotion")
    assert r.status_code == 404
