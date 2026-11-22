"""V5 benchmark engine tests.

All fixtures are tiny synthetic images generated in tmp_path — clearly labeled
as test fixtures, never presented as real data. No metrics are fabricated:
every number is computed from the fixture inputs.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.benchmark.statistics import (  # noqa: E402
    bootstrap_ci, binomial_ci, sample_size_warning, full_report,
)
from backend.benchmark.generators import evaluate_generators  # noqa: E402
from backend.benchmark.failures import (  # noqa: E402
    analyze_false_positives, analyze_false_negatives, analyze_inconclusive,
    analyze_failures,
)
from backend.benchmark.hard_negatives import (  # noqa: E402
    register_hard_negative, load_registry, evaluate_hard_negatives, CATEGORIES,
)
from backend.benchmark.scorecard import build_scorecard  # noqa: E402
from backend.benchmark import engine  # noqa: E402
from backend import validation as _val  # noqa: E402


# ---------------------------------------------------------------- helpers

def make_image(path, color=(200, 30, 30)):
    from PIL import Image
    img = Image.new("RGB", (32, 32), color)
    img.save(path)


def stub_report(prob, verdict="LIKELY_AI_GENERATED", ood="IN_DISTRIBUTION",
                case_id="TL-TEST-0001"):
    def _run(filename, mime, data):
        return {
            "case_id": case_id,
            "ensemble": {
                "ai_probability": prob,
                "verdict": verdict,
                "verdict_reason": "fixture reason",
                "ml_ood_status": ood,
                "detectors": [],
            },
            "signals": [{"name": "noise", "ai_score": prob}],
            "width": 32, "height": 32,
        }
    return _run


def make_manifest(tmpdir, n_pos=4, n_neg=4, cls_pos="ai_generated",
                  cls_neg="authentic", split="test"):
    records = []
    for i in range(n_pos):
        p = os.path.join(tmpdir, f"pos{i}.png")
        make_image(p, (250, 10, 10))
        records.append({"image_id": f"pos{i}", "path": p, "class": cls_pos,
                        "split": split, "generator_family": "genA",
                        "source": "src1", "perceptual_hash": None})
    for i in range(n_neg):
        p = os.path.join(tmpdir, f"neg{i}.png")
        make_image(p, (10, 10, 250))
        records.append({"image_id": f"neg{i}", "path": p, "class": cls_neg,
                        "split": split, "generator_family": "genB",
                        "source": "src2", "perceptual_hash": None})
    return {"dataset_id": "fixture-ds", "version": "1.0",
            "dataset_hash": "fixturehash", "records": records}


@pytest.fixture()
def isolated_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(_val, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(_val, "COUNTER_PATH", str(tmp_path / "counter.json"))


# ---------------------------------------------------------------- statistics

def test_bootstrap_ci_deterministic():
    yt = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    yp = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.4])
    fn = lambda a, b: float(np.mean((b >= 0.5).astype(int) == a))
    c1 = bootstrap_ci(yt, yp, fn, n_boot=200, seed=42)
    c2 = bootstrap_ci(yt, yp, fn, n_boot=200, seed=42)
    assert c1 == c2
    assert c1["lo"] <= c1["hi"]
    assert 0.0 <= c1["lo"] <= 1.0 and 0.0 <= c1["hi"] <= 1.0
    assert c1["method"] == "bootstrap" and c1["seed"] == 42


def test_wilson_interval_known_value():
    # 60/100 at 95%: Wilson ≈ [0.502, 0.691]
    w = binomial_ci(60, 100)
    assert w["method"] == "wilson"
    assert abs(w["lo"] - 0.502) < 0.01
    assert abs(w["hi"] - 0.691) < 0.01


def test_wilson_n_zero():
    w = binomial_ci(0, 0)
    assert w["lo"] is None and w["hi"] is None


def test_sample_size_warnings():
    assert any("NOT MEASURED" in w for w in sample_size_warning(0))
    assert any("INSUFFICIENT SAMPLE SIZE" in w for w in sample_size_warning(10))
    assert any("INSUFFICIENT SAMPLE SIZE" in w for w in sample_size_warning(40))
    assert sample_size_warning(500) == []


def test_full_report_labels_measured():
    yt = np.array([0, 0, 1, 1, 0, 1, 1, 0] * 10)
    yp = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.4] * 10)
    rep = full_report(yt, yp, n_boot=50)
    assert rep["status"] == "MEASURED"
    for name, m in rep["metrics"].items():
        assert m["status"] == "MEASURED", name
    assert rep["confidence_intervals"]["accuracy_bootstrap"]["method"] == "bootstrap"
    assert rep["confidence_intervals"]["accuracy_wilson"]["method"] == "wilson"
    assert set(rep["confusion_matrix"]) == {"tp", "tn", "fp", "fn"}


def test_full_report_small_n_warns():
    yt = np.array([0, 1])
    yp = np.array([0.2, 0.8])
    rep = full_report(yt, yp, n_boot=20)
    assert any("INSUFFICIENT" in w for w in rep["warnings"])


# ---------------------------------------------------------------- generators

def _grow(fam, y_true, ai_prob, verdict="LIKELY_AI_GENERATED"):
    return [{"generator_family": fam, "y_true": t, "ai_prob": p,
             "y_pred": int(p >= 0.5), "verdict": verdict}
            for t, p in zip(y_true, ai_prob)]


def test_generator_degradation_math():
    rows = (_grow("genA", [1, 1, 1, 1], [0.9, 0.9, 0.9, 0.9]) +
            _grow("genB", [1, 1, 1, 1], [0.9, 0.9, 0.1, 0.1]))
    out = evaluate_generators(rows, seen_families=["genA"])
    assert out["seen"]["accuracy"] == 1.0
    assert out["unseen"]["accuracy"] == 0.5
    assert out["degradation"]["accuracy_drop"] == 0.5
    assert out["degradation"]["status"] == "MEASURED"


def test_generator_unmeasured_family_not_listed():
    rows = _grow("genA", [1, 1], [0.9, 0.9])
    out = evaluate_generators(rows, seen_families=["genA"])
    assert "genX" not in out["per_family"]
    assert out["families_evaluated"] == ["genA"]
    assert out["unseen"]["status"] == "NOT_MEASURED"


def test_generator_empty_rows():
    out = evaluate_generators([], seen_families=["genA"])
    assert out["status"] == "NOT_MEASURED"


# ---------------------------------------------------------------- failures

def _frow(case_id, y_true, y_pred, **kw):
    r = {"case_id": case_id, "y_true": y_true, "y_pred": y_pred,
         "ai_prob": 0.9 if y_pred else 0.1,
         "verdict": "LIKELY_AI_GENERATED" if y_pred else "LIKELY_AUTHENTIC",
         "verdict_reason": "fixture", "ood_status": "IN_DISTRIBUTION",
         "attributes": {"camera": "phone", "compression": "heavy",
                        "screenshot": "no"}}
    r.update(kw)
    return r


def test_fp_clustering_counts():
    rows = ([_frow(f"fp{i}", 0, 1) for i in range(6)] +
            [_frow(f"tn{i}", 0, 0) for i in range(6)])
    out = analyze_false_positives(rows)
    assert out["n_false_positives"] == 6
    assert out["false_positive_rate"] == 0.5
    cam = {c["value"]: c["count"] for c in out["clusters"]["camera"]}
    assert cam["phone"] == 6
    assert out["clusters"]["camera"][0]["example_case_ids"][0].startswith("fp")


def test_fn_clustering():
    rows = ([_frow(f"fn{i}", 1, 0, generator_family="genZ") for i in range(3)] +
            [_frow(f"tp{i}", 1, 1, generator_family="genZ") for i in range(3)])
    out = analyze_false_negatives(rows)
    assert out["n_false_negatives"] == 3
    assert out["false_negative_rate"] == 0.5
    gf = {c["value"]: c["count"] for c in out["clusters"]["generator_family"]}
    assert gf["genZ"] == 3


def test_inconclusive_rate_and_reasons():
    rows = ([_frow(f"i{i}", 0, 0) for i in range(2)] +
            [_frow("u1", 0, 0, verdict="INCONCLUSIVE",
                   verdict_reason="forensics vs ML disagree")] +
            [_frow("u2", 1, 1, verdict="INCONCLUSIVE",
                   verdict_reason="forensics vs ML disagree")])
    out = analyze_inconclusive(rows)
    assert out["n_inconclusive"] == 2
    assert out["inconclusive_rate"] == 0.5
    assert out["reason_clusters"][0]["count"] == 2


def test_failures_empty():
    out = analyze_failures([])
    assert out["status"] == "NOT_MEASURED"


# ---------------------------------------------------------------- hard negatives

def test_hn_register_and_load(tmp_path):
    rp = str(tmp_path / "registry.jsonl")
    e = register_hard_negative("ab" * 16, "night_mode",
                               {"source": "fixture"}, registry_path=rp)
    assert e["category"] == "night_mode"
    assert len(load_registry(rp)) == 1


def test_hn_register_bad_category(tmp_path):
    with pytest.raises(ValueError):
        register_hard_negative("ab" * 16, "not_a_category",
                               {"source": "x"},
                               registry_path=str(tmp_path / "r.jsonl"))


def test_hn_register_bad_provenance(tmp_path):
    with pytest.raises(ValueError):
        register_hard_negative("ab" * 16, "hdr", {},
                               registry_path=str(tmp_path / "r.jsonl"))


def test_hn_evaluate_fpr():
    rows = ([{"hard_negative": True, "y_true": 0, "y_pred": 1,
              "hard_negative_category": "hdr", "case_id": "c1"}] * 2 +
            [{"hard_negative": True, "y_true": 0, "y_pred": 0,
              "hard_negative_category": "hdr", "case_id": "c2"}] * 8)
    out = evaluate_hard_negatives(rows)
    assert out["status"] == "MEASURED"
    assert out["n"] == 10
    assert out["false_positive_rate"] == 0.2
    assert out["by_category"]["hdr"]["false_positive_rate"] == 0.2


def test_hn_evaluate_empty():
    assert evaluate_hard_negatives([])["status"] == "NOT_MEASURED"


def test_hn_categories_documented():
    assert "night_mode" in CATEGORIES and "heavy_compression" in CATEGORIES


# ---------------------------------------------------------------- engine

def test_engine_completes_and_groups(tmp_path, isolated_validation):
    man = make_manifest(str(tmp_path))
    rec = engine.run_benchmark(
        man, model_id="tl-forensic-stacking",
        split_config={"type": "standard", "eval_split": "test"},
        run_fn=stub_report(0.9),
        output_dir=str(tmp_path / "out"),
        dataset_type="synthetic", seed=42)
    assert rec["status"] == "completed"
    m = rec["metrics"]
    assert m["overall"]["status"] == "MEASURED"
    assert m["overall"]["n"] == 8
    assert "class" in m["groups"] and "generator_family" in m["groups"]
    assert os.path.isfile(os.path.join(tmp_path, "out", "LEAKAGE_REPORT.json"))
    assert m["groups"]["class"]["ai_generated"]["status"] == "MEASURED"


def test_engine_invalidated_on_leakage(tmp_path, isolated_validation):
    man = make_manifest(str(tmp_path), n_pos=2, n_neg=2)
    # force exact duplicate across splits
    man["records"][0]["sha256"] = "ff" * 32
    man["records"][0]["split"] = "train"
    man["records"][1]["sha256"] = "ff" * 32
    man["records"][1]["split"] = "test"
    rec = engine.run_benchmark(
        man, model_id="tl-forensic-stacking",
        split_config={"type": "standard", "eval_split": "test"},
        run_fn=stub_report(0.9),
        output_dir=str(tmp_path / "out2"),
        dataset_type="synthetic")
    assert rec["status"] == "invalidated"
    assert rec["metrics"]["status"] == "INVALIDATED"
    assert "LEAKAGE" in rec["metrics"]["reason"]


def test_engine_excludes_unknown_class(tmp_path, isolated_validation):
    man = make_manifest(str(tmp_path), n_pos=2, n_neg=2, cls_pos="unknown")
    rec = engine.run_benchmark(
        man, model_id="tl-forensic-stacking",
        split_config={"type": "standard", "eval_split": "test"},
        run_fn=stub_report(0.9),
        output_dir=str(tmp_path / "out3"),
        dataset_type="synthetic")
    m = rec["metrics"]
    assert m["n_excluded_unknown_class"] == 2
    assert m["overall"]["n"] == 2


# ---------------------------------------------------------------- scorecard

def test_scorecard_all_not_measured(isolated_validation):
    sc = build_scorecard([])
    assert sc["status"] == "MEASURED"
    assert len(sc["rows"]) == 13
    assert all(c["status"] == "NOT_MEASURED" for c in sc["rows"])
    assert sc["summary"]["n_not_measured"] == 13


def test_scorecard_reads_real_experiment(tmp_path, isolated_validation):
    man = make_manifest(str(tmp_path))
    rec = engine.run_benchmark(
        man, model_id="tl-forensic-stacking",
        split_config={"type": "standard", "eval_split": "test"},
        run_fn=stub_report(0.9),
        output_dir=str(tmp_path / "out4"),
        dataset_type="synthetic")
    sc = build_scorecard([rec["experiment_id"]])
    by_id = {c["id"]: c for c in sc["rows"]}
    # n=8 < 30 -> INSUFFICIENT_SAMPLE, but honestly derived from the artifact
    assert by_id["synthetic_heldout"]["status"] == "INSUFFICIENT_SAMPLE"
    assert by_id["synthetic_heldout"]["experiment_id"] == rec["experiment_id"]
    assert by_id["synthetic_heldout"]["n"] == 8
    assert by_id["real_ai"]["status"] == "NOT_MEASURED"
    assert by_id["blind_test"]["status"] == "NOT_MEASURED"


# ---------------------------------------------------------------- API

@pytest.fixture()
def client(isolated_validation):
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


def test_api_scorecard(client):
    r = client.get("/api/benchmark/scorecard")
    assert r.status_code == 200
    body = r.json()
    assert len(body["rows"]) == 13
    assert all(c["status"] == "NOT_MEASURED" for c in body["rows"])


def test_api_failures_invalidated(client, tmp_path):
    from backend import validation as v
    eid = v.next_experiment_id()
    repro = v.build_reproducibility("d", "1", "h", "m", "1", "h", "c", 1)
    v.save_experiment(eid, repro, {"status": "INVALIDATED",
                                  "reason": "DATA LEAKAGE"}, status="invalidated")
    r = client.get(f"/api/benchmark/failures/{eid}")
    assert r.status_code == 200
    assert r.json()["status"] == "INVALIDATED"


def test_api_generators_not_measured(client, tmp_path):
    from backend import validation as v
    eid = v.next_experiment_id()
    repro = v.build_reproducibility("d", "1", "h", "m", "1", "h", "c", 1)
    v.save_experiment(eid, repro, {"status": "MEASURED"})
    r = client.get(f"/api/benchmark/generators/{eid}")
    assert r.status_code == 200
    assert r.json()["status"] == "NOT_MEASURED"


def test_api_failures_404(client):
    r = client.get("/api/benchmark/failures/EXP-2099-999999")
    assert r.status_code == 404
