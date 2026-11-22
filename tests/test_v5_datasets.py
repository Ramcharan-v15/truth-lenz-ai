"""V5 dataset registry / licensing / ingestion tests.

All fixtures are tiny synthetic images generated in tmp_path — clearly labeled
as test fixtures, never presented as real data. The registry and manifest
roots are monkeypatched to tmp dirs so the real project tree is untouched.
"""
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import backend.datasets.registry as reg
import backend.datasets.ingest as ing
import backend.datasets.quality as qua
import backend.datasets.cards as cards


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    """Isolate registry + dataset roots to tmp."""
    monkeypatch.setattr(reg, "_REGISTRY_PATH", str(tmp_path / "registry.json"))
    monkeypatch.setattr(ing, "_DATASETS_ROOT", str(tmp_path / "datasets"))
    monkeypatch.setattr(qua, "_DATASETS_ROOT", str(tmp_path / "datasets"))
    monkeypatch.setattr(cards, "_CARDS_DIR", str(tmp_path / "cards"))
    return tmp_path


def make_jpg(path, color=(200, 50, 50), size=(32, 32)):
    from PIL import Image
    Image.new("RGB", size, color).save(path, "JPEG")


def make_png(path, color=(50, 200, 50), size=(32, 32)):
    from PIL import Image
    Image.new("RGB", size, color).save(path, "PNG")


# ------------------------------------------------------------- registry ---

def test_register_assigns_rwd_id(iso):
    did = reg.register_dataset({"name": "fixture-a", "source": "test"})
    assert did.startswith("RWD-")
    parts = did.split("-")
    assert len(parts) == 3 and len(parts[2]) == 6 and parts[1].isdigit()


def test_register_sequential_ids(iso):
    a = reg.register_dataset({"name": "fixture-a"})
    b = reg.register_dataset({"name": "fixture-b"})
    assert a != b
    assert int(b.split("-")[2]) == int(a.split("-")[2]) + 1


def test_register_requires_name(iso):
    with pytest.raises(ValueError):
        reg.register_dataset({"source": "test"})
    with pytest.raises(ValueError):
        reg.register_dataset("not-a-dict")


def test_register_defaults_license_unknown(iso):
    did = reg.register_dataset({"name": "fixture-a"})
    assert reg.get_dataset(did)["license_status"] == "UNKNOWN"
    assert not reg.is_legally_usable(did)


def test_register_rejects_bad_license_status(iso):
    with pytest.raises(ValueError):
        reg.register_dataset({"name": "x", "license_status": "BOGUS"})


def test_self_certification_downgraded(iso):
    """Passing LICENSE_VERIFIED at registration never sticks."""
    did = reg.register_dataset({"name": "x", "license_status": "LICENSE_VERIFIED"})
    assert reg.get_dataset(did)["license_status"] == "LICENSE_REQUIRES_REVIEW"
    assert not reg.is_legally_usable(did)


def test_update_license_verified_requires_notes(iso):
    did = reg.register_dataset({"name": "x"})
    with pytest.raises(ValueError):
        reg.update_license_status(did, "LICENSE_VERIFIED", notes="   ")
    with pytest.raises(ValueError):
        reg.update_license_status(did, "BOGUS", notes="n")
    with pytest.raises(KeyError):
        reg.update_license_status("RWD-2099-000001", "RESTRICTED", notes="n")


def test_is_legally_usable_only_when_verified(iso):
    did = reg.register_dataset({"name": "x"})
    for st in ("UNKNOWN", "LICENSE_REQUIRES_REVIEW", "RESTRICTED"):
        reg.update_license_status(did, st, notes="reviewed")
        assert not reg.is_legally_usable(did)
    reg.update_license_status(did, "LICENSE_VERIFIED",
                              notes="Checked CC-BY-4.0 deed 2026-10-06.")
    assert reg.is_legally_usable(did)


def test_list_datasets(iso):
    reg.register_dataset({"name": "a"})
    reg.register_dataset({"name": "b"})
    ds = reg.list_datasets()
    assert len(ds) == 2
    assert {d["name"] for d in ds} == {"a", "b"}


# ------------------------------------------------------------ ingestion ---

def _src_with_images(tmp_path, n=3):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(n):
        make_jpg(str(src / f"img{i}.jpg"), color=(40 * i + 40, 90, 140))
    return str(src)


def test_ingest_valid_images(iso, tmp_path):
    src = _src_with_images(tmp_path)
    res = ing.ingest_dataset(src, {"name": "fixture-ingest"}, class_label="authentic")
    assert res["counts"]["accepted"] == 3
    assert res["counts"]["scanned"] == 3
    assert os.path.isfile(res["manifest_path"])
    assert len(res["records"]) == 3
    for r in res["records"]:
        assert len(r["sha256"]) == 64
        assert len(r["perceptual_hash"]) == 16
        assert r["class"] == "authentic"


def test_ingest_rejects_non_images(iso, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_jpg(str(src / "ok.jpg"))
    (src / "notes.txt").write_text("hello, not an image")
    (src / "fake.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10 + b"NOTREALDATA" * 50)
    res = ing.ingest_dataset(src, {"name": "fixture-mixed"})
    assert res["counts"]["accepted"] == 1
    # fake.png has PNG magic but is corrupt -> rejected as corrupt or not-image
    assert res["counts"]["rejected_not_image"] + res["counts"]["rejected_corrupt"] >= 1


def test_ingest_rejects_corrupt_image(iso, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    # valid JPEG magic, truncated body -> undecodable
    (src / "bad.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 20)
    with pytest.raises(ValueError, match="zero valid images"):
        ing.ingest_dataset(src, {"name": "fixture-corrupt"})


def test_ingest_detects_exact_duplicates(iso, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_jpg(str(src / "a.jpg"))
    # byte-identical copy
    data = open(src / "a.jpg", "rb").read()
    open(src / "b.jpg", "wb").write(data)
    res = ing.ingest_dataset(src, {"name": "fixture-dup"})
    assert res["counts"]["accepted"] == 1
    assert any("exact duplicate" in i for i in res["issues"])


def test_ingest_enforces_size_limit(iso, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_jpg(str(src / "ok.jpg"))
    (src / "big.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
    with pytest.raises(ValueError):
        # max_bytes=10 forces even the small jpg over the limit
        ing.ingest_dataset(src, {"name": "fixture-big"}, max_bytes=10)


def test_validate_ingestion_clean(iso, tmp_path):
    src = _src_with_images(tmp_path)
    res = ing.ingest_dataset(src, {"name": "fixture-v"})
    manifest = json.load(open(res["manifest_path"]))
    assert ing.validate_ingestion(manifest) == []


def test_validate_ingestion_flags_problems(iso, tmp_path):
    src = _src_with_images(tmp_path)
    res = ing.ingest_dataset(src, {"name": "fixture-v2"})
    manifest = json.load(open(res["manifest_path"]))
    manifest["records"][0]["sha256"] = manifest["records"][1]["sha256"]
    manifest["records"][2].pop("class")
    issues = ing.validate_ingestion(manifest)
    assert any("exact duplicate" in i for i in issues)
    assert any("missing class" in i for i in issues)


def test_metadata_only_mode(iso, tmp_path):
    src = _src_with_images(tmp_path, n=2)
    res = ing.ingest_dataset(src, {"name": "fixture-meta"}, metadata_only=True)
    assert all(r["local_ref"].startswith("metadata-only") for r in res["records"])
    assert res["counts"]["accepted"] == 2


# -------------------------------------------------------------- quality ---

def test_audit_ready_when_all_pass(iso, tmp_path):
    src = _src_with_images(tmp_path, n=3)
    res = ing.ingest_dataset(src, {"name": "fixture-q"})
    reg.update_license_status(res["dataset_id"], "LICENSE_VERIFIED",
                              notes="Test fixture license reviewed.")
    audit = qua.audit_dataset(res["dataset_id"])
    assert audit["status"] == "READY", audit["reasons"]
    assert audit["reasons"] == []
    names = {c["name"] for c in audit["checks"]}
    assert {"manifest_schema", "exact_duplicates", "license_validation",
            "file_availability", "image_decoding", "label_validation",
            "class_balance"} <= names


def test_audit_blocked_license_unknown(iso, tmp_path):
    src = _src_with_images(tmp_path, n=2)
    res = ing.ingest_dataset(src, {"name": "fixture-q2"})
    audit = qua.audit_dataset(res["dataset_id"])
    assert audit["status"] == "BLOCKED"
    assert any("license" in r for r in audit["reasons"])


def test_audit_blocked_missing_manifest(iso):
    did = reg.register_dataset({"name": "no-manifest"})
    audit = qua.audit_dataset(did)
    assert audit["status"] == "BLOCKED"


# ---------------------------------------------------------------- cards ---

def test_card_generation(iso, tmp_path):
    src = _src_with_images(tmp_path, n=2)
    res = ing.ingest_dataset(src, {"name": "fixture-card"})
    card = cards.generate_card(res["dataset_id"])
    assert os.path.isfile(cards.card_path(res["dataset_id"]))
    for section in ("## License", "## Provenance", "## Limitations",
                    "## Intended use", "## Prohibited use", "## Quality audit summary"):
        assert section in card
    assert "UNKNOWN" in card  # license state honestly shown


# ------------------------------------------------------------------ API ---

@pytest.fixture()
def client(iso):
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


def test_api_register_and_registry(client):
    r = client.post("/api/datasets/register", json={"name": "api-fixture",
                                                    "license": "CC-BY-4.0"})
    assert r.status_code == 200
    did = r.json()["dataset_id"]
    assert did.startswith("RWD-")
    assert r.json()["license_status"] == "UNKNOWN"

    r = client.get("/api/datasets/registry")
    assert r.status_code == 200
    ids = [d["dataset_id"] for d in r.json()["datasets"]]
    assert did in ids

    r = client.get(f"/api/datasets/registry/{did}")
    assert r.status_code == 200
    body = r.json()
    assert body["record"]["dataset_id"] == did
    assert body["quality_audit"]["status"] == "BLOCKED"  # no manifest yet

    r = client.get(f"/api/datasets/registry/{did}/card")
    assert r.status_code == 200
    assert "Dataset Card" in r.text

    r = client.get("/api/datasets/registry/RWD-2099-999999")
    assert r.status_code == 404


def test_api_register_requires_name(client):
    r = client.post("/api/datasets/register", json={"source": "x"})
    assert r.status_code == 400


def test_api_registry_route_not_swallowed(client):
    """Literal /registry path must not match /datasets/{dataset_id}."""
    r = client.get("/api/datasets/registry")
    assert r.status_code == 200
    assert "datasets" in r.json()
    # existing V4 manifest route still works
    r = client.get("/api/datasets/synthetic")
    assert r.status_code == 200
