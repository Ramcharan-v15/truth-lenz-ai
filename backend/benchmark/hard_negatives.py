"""Hard-negative registry (V5).

Hard negatives are authentic images whose processing naturally resembles
AI/manipulation artifacts. The purpose: reduce false positives without
weakening genuine detection. Every entry retains provenance.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

CATEGORIES = [
    "heavy_compression",
    "denoising",
    "sharpening",
    "hdr",
    "computational_photography",
    "panorama",
    "portrait_mode",
    "night_mode",
    "ai_assisted_enhancement",
    "aggressive_resize",
    "screenshot",
    "unusual_pipeline",
    "scanned",
    "old_photograph",
    "low_light_noise",
    "repeated_editing",
    "other",
]

DEFAULT_REGISTRY = os.path.join(os.path.dirname(__file__), "..", "..",
                                "datasets", "manifests", "hard_negatives",
                                "registry.jsonl")


def register_hard_negative(image_hash: str, category: str,
                           provenance: Dict,
                           registry_path: Optional[str] = None) -> Dict:
    """Append a hard-negative entry. Raises ValueError on bad category/hash."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown hard-negative category '{category}'; "
                         f"valid: {', '.join(CATEGORIES)}")
    if not image_hash or len(image_hash) < 8:
        raise ValueError("image_hash must be a non-trivial hash string")
    if not isinstance(provenance, dict) or not provenance.get("source"):
        raise ValueError("provenance must be a dict with at least 'source'")
    path = registry_path or DEFAULT_REGISTRY
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entry = {"image_hash": image_hash, "category": category,
             "provenance": provenance}
    with open(path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def load_registry(registry_path: Optional[str] = None) -> List[Dict]:
    path = registry_path or DEFAULT_REGISTRY
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out


def evaluate_hard_negatives(rows: List[Dict]) -> Dict:
    """FPR on hard negatives: how often are real-but-tricky images accused?

    rows: dicts with hard_negative (bool), y_true (0=authentic), y_pred.
    Only authentic hard negatives count (y_true==0).
    """
    hn = [r for r in rows if r.get("hard_negative") and r.get("y_true") == 0]
    n = len(hn)
    if n == 0:
        return {"status": "NOT_MEASURED", "n": 0,
                "note": "no hard-negative rows evaluated"}
    fp = sum(1 for r in hn if r.get("y_pred") == 1)
    by_category: Dict[str, Dict] = {}
    for r in hn:
        cat = r.get("hard_negative_category", "other")
        d = by_category.setdefault(cat, {"n": 0, "fp": 0, "case_ids": []})
        d["n"] += 1
        if r.get("y_pred") == 1:
            d["fp"] += 1
            if len(d["case_ids"]) < 5:
                d["case_ids"].append(r.get("case_id"))
    for d in by_category.values():
        d["false_positive_rate"] = round(d["fp"] / d["n"], 4) if d["n"] else None
    return {
        "status": "MEASURED",
        "n": n,
        "false_positives": fp,
        "false_positive_rate": round(fp / n, 4),
        "by_category": by_category,
        "note": ("Share of authentic hard-negative images incorrectly "
                 "classified as AI-generated/manipulated."),
    }
