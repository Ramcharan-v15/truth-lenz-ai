"""V4 dataset subsystem tests.

All fixtures are tiny synthetic images generated in tmp_path — clearly labeled
as test fixtures, never presented as real data.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.validate import (  # noqa: E402
    validate_record, validate_manifest, validate_experiment, apply_defaults,
)
from datasets.leakage import (  # noqa: E402
    dhash, dhash_from_pixels, hamming_distance,
    exact_duplicates, perceptual_duplicates, check_split_integrity,
    canonical_records_hash,
)
from datasets.split import (  # noqa: E402
    standard_split, generator_heldout_split, source_heldout_split,
)
from datasets.blind import seal_blind_set, verify_blind_integrity  # noqa: E402

DATASETS = os.path.join(os.path.dirname(__file__), "..", "datasets")


def make_record(image_id, sha256=None, cls="authentic", split="none", **kw):
    import hashlib
    rec = {
        "image_id": image_id,
        "sha256": sha256 or hashlib.sha256(image_id.encode()).hexdigest(),
        "dataset_id": "test-ds",
        "class": cls,
        "split": split,
    }
    rec.update(kw)
    return rec


def make_image(path, seed=0, size=(32, 32)):
    """Deterministic tiny test image."""
    import random
    from PIL import Image
    rng = random.Random(seed)
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
    img.save(path)
    return path


# ------------------------------------------------------- schema tests ---
def test_valid_record_passes():
    assert validate_record(make_record("img1")) == []


def test_record_missing_required_fails():
    errs = validate_record({"image_id": "x"})
    assert any("sha256" in e for e in errs)
    assert any("dataset_id" in e for e in errs)
    assert any("class" in e for e in errs)


def test_record_bad_class_fails():
    r = make_record("img1", cls="not_a_class")
    errs = validate_record(r)
    assert any("class" in e for e in errs)


def test_record_bad_sha256_fails():
    r = make_record("img1", sha256="zzz")
    assert validate_record(r) != []


def test_apply_defaults_never_invents():
    r = apply_defaults(make_record("img1"))
    assert r["source_type"] == "unknown"
    assert r["generator_name"] == "unknown"
    assert r["license"] == "not_provided"
    # present values untouched
    r2 = apply_defaults(make_record("img1", source_type="camera"))
    assert r2["source_type"] == "camera"


def test_valid_manifest_passes():
    m = {
        "dataset_id": "d1", "version": "1.0",
        "created_at": "2026-10-06T00:00:00+00:00",
        "total_images": 0, "validation_status": "NOT_YET_VALIDATED",
    }
    assert validate_manifest(m) == []


def test_manifest_bad_status_fails():
    m = {"dataset_id": "d1", "version": "1.0", "created_at": "t",
         "total_images": 0, "validation_status": "SUPER_VALIDATED"}
    assert validate_manifest(m) != []


def test_experiment_schema():
    exp = {"experiment_id": "EXP-2026-000001", "dataset_id": "d",
           "dataset_version": "1.0", "model_id": "m", "model_version": "1.0",
           "seed": 42, "timestamp": "2026-10-06T00:00:00+00:00", "status": "completed"}
    assert validate_experiment(exp) == []
    bad = dict(exp, experiment_id="nope")
    assert validate_experiment(bad) != []


# ---------------------------------------------------------- dHash ------
def test_dhash_identical_images_distance_zero(tmp_path):
    p1 = make_image(str(tmp_path / "a.png"), seed=1)
    p2 = make_image(str(tmp_path / "b.png"), seed=1)
    assert hamming_distance(dhash(p1), dhash(p2)) == 0


def test_dhash_different_images_distance_positive(tmp_path):
    p1 = make_image(str(tmp_path / "a.png"), seed=1)
    p2 = make_image(str(tmp_path / "b.png"), seed=999)
    assert hamming_distance(dhash(p1), dhash(p2)) > 0


def test_dhash_from_pixels_known():
    # strictly decreasing across each row -> every comparison is 1
    px = [100 - 10 * (i % 9) for i in range(72)]
    h = dhash_from_pixels(px)
    assert h == "f" * 16
    assert hamming_distance(h, "0" * 16) == 64


# ------------------------------------------------- duplicate detection ---
def test_exact_duplicates_found():
    recs = [make_record("a", sha256="ab" * 32), make_record("b", sha256="ab" * 32),
            make_record("c", sha256="cd" * 32)]
    dups = exact_duplicates(recs)
    assert list(dups.keys()) == ["ab" * 32]
    assert set(dups["ab" * 32]) == {"a", "b"}


def test_perceptual_duplicates_found():
    h_all_ones, h_all_zero = "f" * 16, "0" * 16
    recs = [make_record("a", perceptual_hash=h_all_ones),
            make_record("b", perceptual_hash=h_all_zero),
            make_record("c")]  # no phash -> skipped
    res = perceptual_duplicates(recs, threshold=5)
    assert res["pairs"] == []  # distance 64 > 5
    assert res["skipped_no_phash"] == 1
    res2 = perceptual_duplicates(recs, threshold=64)
    assert len(res2["pairs"]) == 1


def test_split_integrity_clean():
    recs = [make_record(f"img{i}", split="train" if i < 5 else "test") for i in range(10)]
    rep = check_split_integrity(recs)
    assert rep["valid"] is True
    assert rep["n_blocked"] == 0


def test_split_integrity_blocks_exact_leak():
    recs = [make_record("a", sha256="ab" * 32, split="train"),
            make_record("b", sha256="ab" * 32, split="test")]
    rep = check_split_integrity(recs)
    assert rep["valid"] is False
    assert rep["n_blocked"] > 0
    assert "INVALID" in rep["verdict"]


def test_split_integrity_blocks_perceptual_leak():
    recs = [make_record("a", perceptual_hash="f" * 16, split="train"),
            make_record("b", perceptual_hash="f" * 16, split="test")]
    rep = check_split_integrity(recs, perceptual_threshold=5)
    assert rep["valid"] is False


# ------------------------------------------------------------ splits ---
def _labeled_records(n=20):
    recs = []
    for i in range(n):
        recs.append(make_record(
            f"img{i}", cls="authentic" if i % 2 == 0 else "ai_generated",
            generator_family="diffusion" if i % 2 == 0 else "gan",
            generator_name="genA" if i < n // 2 else "genB",
            source_type="camera" if i % 3 else "synthetic",
        ))
    return recs


def test_standard_split_deterministic_and_stratified():
    r1 = standard_split(_labeled_records(), seed=42)
    r2 = standard_split(_labeled_records(), seed=42)
    assert [r["image_id"] for r in r1["records"]] == [r["image_id"] for r in r2["records"]]
    assert [r["split"] for r in r1["records"]] == [r["split"] for r in r2["records"]]
    counts = r1["split_counts"]
    assert set(counts) == {"train", "validation", "test"}
    assert sum(counts.values()) == 20


def test_standard_split_ratios():
    recs = [make_record(f"img{i}", cls="authentic") for i in range(100)]
    res = standard_split(recs, ratios=(0.7, 0.15, 0.15), seed=1)
    assert res["split_counts"]["train"] == 70
    assert res["split_counts"]["validation"] == 15
    assert res["split_counts"]["test"] == 15


def test_generator_heldout_isolation():
    recs = _labeled_records(20)
    res = generator_heldout_split(recs, heldout_generators=["genB"], seed=42)
    assert res["heldout_isolation_ok"] is True
    test_gens = {r["generator_name"].lower() for r in res["records"] if r["split"] == "test"}
    train_gens = {r["generator_name"].lower() for r in res["records"] if r["split"] == "train"}
    assert test_gens == {"genb"}
    assert "genb" not in train_gens


def test_source_heldout_isolation():
    recs = _labeled_records(20)
    res = source_heldout_split(recs, heldout_sources=["synthetic"], seed=42)
    assert res["source_isolation_ok"] is True
    assert all(r["source_type"] != "synthetic"
               for r in res["records"] if r["split"] == "train")


# -------------------------------------------------------------- blind ---
def test_blind_seal_and_verify(tmp_path):
    recs = [make_record(f"img{i}", cls="authentic") for i in range(5)]
    out = str(tmp_path / "blind_test_manifest.json")
    manifest = seal_blind_set(recs, out, dataset_id="tl-blind-test")
    assert manifest["sealed"] is True
    assert all(r["split"] == "blind" for r in manifest["records"])
    assert manifest["total_images"] == 5
    assert "never use for training" in manifest["usage_restrictions"]
    result = verify_blind_integrity(out)
    assert result["valid"] is True


def test_blind_tamper_detected(tmp_path):
    recs = [make_record(f"img{i}") for i in range(3)]
    out = str(tmp_path / "blind.json")
    seal_blind_set(recs, out, dataset_id="tl-blind-test")
    m = json.load(open(out))
    m["records"][0]["class"] = "ai_generated"  # tamper
    result = verify_blind_integrity(m)
    assert result["valid"] is False


def test_blind_rejects_invalid_record(tmp_path):
    with pytest.raises(ValueError):
        seal_blind_set([{"image_id": "bad"}], str(tmp_path / "b.json"), dataset_id="x")


# ------------------------------------------------- synthetic import -----
def test_synthetic_manifest_exists_and_valid():
    p = os.path.join(DATASETS, "manifests", "synthetic", "manifest.json")
    assert os.path.isfile(p), "run datasets/import_synthetic.py first"
    m = json.load(open(p))
    assert validate_manifest(m) == []
    assert m["dataset_id"] == "tl-synthetic-v1"
    assert m["validation_status"] == "SYNTHETIC_ONLY"
    assert m["total_images"] == 780
    assert len(m["records"]) == 780
    # class mapping honored
    classes = {r["class"] for r in m["records"]}
    assert classes == {"authentic", "ai_generated", "real_edited", "ai_edited"}
    assert any(r.get("hard_negative") for r in m["records"])
    # every record validates
    for r in m["records"]:
        assert validate_record(r) == [], r["image_id"]
    # all marked synthetic, nothing invented as real
    assert all(r["source_type"] == "synthetic" for r in m["records"])


def test_synthetic_manifest_hash_stable():
    p = os.path.join(DATASETS, "manifests", "synthetic", "manifest.json")
    m = json.load(open(p))
    assert m["dataset_hash"] == canonical_records_hash(m["records"])


def test_template_manifests_honest():
    for sub in ("real_world", "generators", "hard_negatives", "robustness"):
        p = os.path.join(DATASETS, "manifests", sub, "manifest.json")
        m = json.load(open(p))
        assert m["total_images"] == 0
        assert m["validation_status"] == "NOT_YET_VALIDATED"
        assert m["records"] == []
