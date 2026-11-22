"""Focused tests for the dataset-ingestion dHash convention and its
grouping thresholds.

Fixtures are tiny synthetic images generated in tmp_path / in memory —
clearly labeled test fixtures, never presented as real data, and never
taken from the pilot or dataset. No network access.

The authoritative convention under test is
backend/datasets/ingest.py::dhash_bytes:
  grayscale -> resize (9, 8) with LANCZOS -> horizontal comparisons
  (px[x] > px[x+1]) -> 64-bit value rendered as 16 lowercase hex chars.
"""
import io
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import backend.datasets.ingest as ing
from datasets.leakage import hamming_distance, perceptual_duplicates


def _png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _h_gradient(width=64, height=64, reverse=False):
    """Left-to-right (or reversed) horizontal luminance gradient."""
    img = Image.new("L", (width, height))
    px = img.load()
    for x in range(width):
        v = 255 - int(255 * x / (width - 1)) if reverse else int(255 * x / (width - 1))
        for y in range(height):
            px[x, y] = v
    return img


def test_dhash_output_format_is_16_lowercase_hex():
    h = ing.dhash_bytes(_png_bytes(Image.new("RGB", (32, 32), (200, 30, 30))))
    assert isinstance(h, str)
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_dhash_is_repeatable_for_same_bytes():
    data = _png_bytes(_h_gradient())
    assert ing.dhash_bytes(data) == ing.dhash_bytes(data)


def test_dhash_is_grayscale_based_rgb_vs_l_agree():
    """Same luminance content in RGB and L modes must hash identically,
    because the convention converts to grayscale first."""
    rgb = _h_gradient().convert("RGB")
    lum = _h_gradient()
    assert ing.dhash_bytes(_png_bytes(rgb)) == ing.dhash_bytes(_png_bytes(lum))


def test_dhash_flat_image_is_all_zero_bits():
    h = ing.dhash_bytes(_png_bytes(Image.new("RGB", (32, 32), (200, 30, 30))))
    assert h == "0000000000000000"


def test_dhash_mirrored_gradient_is_bitwise_complement():
    """Mirroring flips every horizontal comparison, so the mirrored hash
    must be the bitwise complement: Hamming distance 64."""
    a = ing.dhash_bytes(_png_bytes(_h_gradient()))
    b = ing.dhash_bytes(_png_bytes(_h_gradient(reverse=True)))
    assert hamming_distance(a, b) == 64


def test_dhash_invalid_and_empty_bytes_return_none():
    assert ing.dhash_bytes(b"not an image") is None
    assert ing.dhash_bytes(b"") is None
    assert ing.dhash_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10) is None


def test_decode_ok_accepts_valid_and_rejects_corrupt():
    ok, _ = ing.decode_ok(_png_bytes(Image.new("RGB", (16, 16), (1, 2, 3))))
    assert ok is True
    ok, why = ing.decode_ok(b"not an image")
    assert ok is False
    assert "undecodable" in why


def test_hamming_distance_known_values():
    assert hamming_distance("0000000000000000", "0000000000000000") == 0
    assert hamming_distance("0000000000000000", "ffffffffffffffff") == 64
    # First nibble f (1111) vs 0 (0000): exactly 4 differing bits.
    assert hamming_distance("ff00ff00ff00ff00", "0f00ff00ff00ff00") == 4
    # Distance is symmetric.
    assert hamming_distance("ff00ff00ff00ff00", "0f00ff00ff00ff00") == hamming_distance(
        "0f00ff00ff00ff00", "ff00ff00ff00ff00")


def _rec(image_id, ph):
    return {"image_id": image_id, "perceptual_hash": ph, "sha256": "x" * 64}


def test_grouping_threshold_3_vs_review_flag_5():
    """Grouping (<=3) is stricter than the review flag (<=5).
    Each case uses an isolated pair so cross-pair distances cannot
    interfere: distance 3 groups; distance 4 does not group but is
    flagged; distance 6 is neither."""
    base = "0000000000000000"
    d3 = "0000000000000007"  # 3 bits set
    d4 = "000000000000000f"  # 4 bits set
    d6 = "000000000000003f"  # 6 bits set
    assert hamming_distance(base, d3) == 3
    assert hamming_distance(base, d4) == 4
    assert hamming_distance(base, d6) == 6
    assert [(p["image_a"], p["image_b"]) for p in
            perceptual_duplicates([_rec("a", base), _rec("b", d3)],
                                  threshold=3)["pairs"]] == [("a", "b")]
    assert perceptual_duplicates([_rec("a", base), _rec("c", d4)],
                                 threshold=3)["pairs"] == []
    assert [(p["image_a"], p["image_b"]) for p in
            perceptual_duplicates([_rec("a", base), _rec("c", d4)],
                                  threshold=5)["pairs"]] == [("a", "c")]
    assert perceptual_duplicates([_rec("a", base), _rec("d", d6)],
                                 threshold=5)["pairs"] == []


def test_perceptual_duplicates_skips_records_without_hash():
    recs = [_rec("a", "0000000000000000"), {"image_id": "b", "sha256": "y" * 64}]
    out = perceptual_duplicates(recs, threshold=5)
    assert out["pairs"] == []
    assert out["skipped_no_phash"] == 1
    assert out["compared"] == 1


def test_ingest_dataset_rejects_missing_dir_and_non_images(tmp_path):
    with pytest.raises(ValueError, match="source_dir not found"):
        ing.ingest_dataset(str(tmp_path / "nope"), {"name": "x"})
    src = tmp_path / "src"
    src.mkdir()
    (src / "note.txt").write_text("hello")
    with pytest.raises(ValueError, match="zero valid images"):
        ing.ingest_dataset(str(src), {"name": "x"})


def test_ingest_dataset_accepts_fixture_and_flags_corrupt(tmp_path, monkeypatch):
    import backend.datasets.registry as reg
    monkeypatch.setattr(reg, "_REGISTRY_PATH", str(tmp_path / "registry.json"))
    monkeypatch.setattr(ing, "_DATASETS_ROOT", str(tmp_path / "datasets"))
    src = tmp_path / "src"
    src.mkdir()
    _h_gradient().convert("RGB").save(src / "good.png")
    (src / "corrupt.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
    res = ing.ingest_dataset(str(src), {"name": "fixture"}, class_label="authentic")
    assert res["counts"]["accepted"] == 1
    assert res["counts"]["rejected_corrupt"] == 1
    assert len(res["records"][0]["perceptual_hash"]) == 16
