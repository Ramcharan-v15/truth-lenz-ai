"""Tests for the V5.1.8.51 operational review manifest.

The manifest is a worklist built from existing inventory fields only.
These tests verify its structure and its untouched initial state:
exactly 50 rows, unique IDs matching the V5.1.8.49 inventory, all
NOT_REVIEWED, all decision fields empty, and the original inventory
itself unmodified (still all NOT_REVIEWED with empty decisions).
No images are opened by these tests.
"""
import csv
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OPS = os.path.join(REPO, "V5.1.8.51_SDXL_REVIEW_OPERATIONS.csv")
INV = os.path.join(REPO, "V5.1.8.49_SDXL_PILOT_REVIEW_INVENTORY.csv")

DECISION_FIELDS = [
    "reviewer",
    "review_timestamp",
    "visual_quality",
    "category_correctness",
    "artifact_notes",
    "prompt_mismatch_notes",
    "final_candidate_decision",
]

COPIED_FIELDS = [
    "file_path",
    "category",
    "prompt_id",
    "seed",
    "model_revision",
    "sha256_verified",
    "provenance_fields_present",
    "scheduler",
]


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_ops_csv_has_exactly_50_rows_and_unique_ids():
    rows = _rows(OPS)
    assert len(rows) == 50
    ids = [r["image_id"] for r in rows]
    assert len(set(ids)) == 50


def test_ops_ids_match_inventory_ids_exactly():
    ops_ids = {r["image_id"] for r in _rows(OPS)}
    inv_ids = {r["image_id"] for r in _rows(INV)}
    assert ops_ids == inv_ids


def test_ops_copied_fields_match_inventory_row_for_row():
    inv = {r["image_id"]: r for r in _rows(INV)}
    for row in _rows(OPS):
        src = inv[row["image_id"]]
        for col in COPIED_FIELDS:
            assert row[col] == src[col], (row["image_id"], col)


def test_ops_all_not_reviewed_and_decisions_empty():
    for row in _rows(OPS):
        assert row["review_status"] == "NOT_REVIEWED"
        for col in DECISION_FIELDS:
            assert row[col] == "", (row["image_id"], col)


def test_original_inventory_unmodified():
    """The V5.1.8.49 inventory must still be 50 rows, all NOT_REVIEWED,
    with empty review decisions and scheduler NOT_RECORDED throughout."""
    rows = _rows(INV)
    assert len(rows) == 50
    assert len({r["image_id"] for r in rows}) == 50
    for row in rows:
        assert row["eligibility_status"] == "NOT_REVIEWED"
        assert row["review_decision"] == ""
        assert row["scheduler"] == "NOT_RECORDED"
        assert row["sha256_verified"] == "VERIFIED"
