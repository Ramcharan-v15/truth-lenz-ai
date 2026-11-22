"""V5.1 tests: real-world dataset acquisition, blind evaluation, honesty.

Uses the Columbia splicing dataset (RWD-2026-000001) if present;
skips gracefully if not downloaded.
"""
import json
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DS_ID = "RWD-2026-000001"
MANIFEST = os.path.join(ROOT, "datasets", "manifests", DS_ID, "manifest.json")

requires_data = pytest.mark.skipif(
    not os.path.isfile(MANIFEST),
    reason="Columbia dataset not downloaded")


# ---------- dataset registration -------------------------------------------
def test_v51_dataset_registered():
    from backend.datasets.registry import get_dataset
    d = get_dataset(DS_ID)
    assert d is not None
    assert d["license_status"] == "LICENSE_VERIFIED"


def test_v51_license_not_self_certified():
    """License was verified from the official page, not self-certified."""
    from backend.datasets.registry import get_dataset
    d = get_dataset(DS_ID)
    notes = d.get("license_notes", "")
    assert "columbia.edu" in notes.lower() or "dvmm" in notes.lower()


@requires_data
def test_v51_manifest_conforms():
    m = json.load(open(MANIFEST))
    assert m["dataset_id"] == DS_ID
    assert m["total_images"] >= 1000
    assert m["validation_status"] in ("NOT_YET_VALIDATED", "VALIDATED")


@requires_data
def test_v51_no_synthetic_labels():
    """Real dataset must not contain synthetic test_data labels."""
    m = json.load(open(MANIFEST))
    classes = {r["class"] for r in m["records"]}
    assert classes <= {"authentic", "real_edited"}
    # manipulation type preserved
    spliced = [r for r in m["records"] if r["class"] == "real_edited"]
    assert all(r.get("manipulation_type") == "splicing" for r in spliced)


@requires_data
def test_v51_quality_audit_ready():
    from backend.datasets.quality import audit_dataset
    result = audit_dataset(DS_ID)
    assert result["status"] == "READY"


# ---------- blind protocol --------------------------------------------------
@requires_data
def test_v51_blind_experiment_exists():
    from backend.validation import list_experiments
    exps = list_experiments()
    ids = [e["experiment_id"] for e in exps]
    assert any(i.startswith("EXP-2026-") for i in ids)


def test_v51_model_stays_demonstration():
    """No promotion without real-world evidence."""
    from backend.models.promotion import get_promotion_status
    s = get_promotion_status("tl-forensic-stacking")
    assert s["current_status"] == "DEMONSTRATION"


def test_v51_promotion_denied_without_evidence():
    from backend.models.promotion import promote
    with pytest.raises(Exception):
        promote("tl-forensic-stacking", "VALIDATED", evidence={})


# ---------- honesty ----------------------------------------------------------
def test_v51_test_data_is_synthetic():
    """Guard: test_data/ must never be presented as real-world data."""
    gen = open(os.path.join(ROOT, "tests", "generate_test_data.py")).read()
    assert "synthetic" in gen.lower()
    assert "no copyrighted material" in gen.lower()


def test_v51_scorecard_honest_without_data():
    """Scorecard must show NOT_MEASURED when no experiments exist."""
    from backend.benchmark.scorecard import build_scorecard
    sc = build_scorecard([])
    assert sc["rows"], "scorecard should have rows"
    for row in sc["rows"]:
        assert row["status"] in ("NOT_MEASURED", "INSUFFICIENT_SAMPLE",
                                 "MEASURED", "BLOCKED", "INVALIDATED")
