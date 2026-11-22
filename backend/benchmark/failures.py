"""False-positive / false-negative / inconclusive analysis (V5).

Answers:
  FP:  "Which authentic images are incorrectly classified as AI/manipulated?"
  FN:  "Which real AI/manipulated images are incorrectly called authentic?"
  INC: "Where is the evidence insufficient or contradictory?"

Rows are dicts produced by the benchmark engine:
  case_id, y_true, y_pred, ai_prob, verdict, verdict_reason,
  class, generator_family, source, ood_status, hard_negative,
  attributes: {camera, compression, resolution, screenshot,
               post_processing, forensic_pattern, model_contribution}

Clustering is descriptive, never inferential: counts and example case IDs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List


def _cluster(rows: List[Dict], dims: List[str], max_examples: int = 5) -> Dict:
    """Group rows by each dimension; return counts + example case IDs."""
    out: Dict[str, List[Dict]] = {}
    for dim in dims:
        counts: Counter = Counter()
        examples: Dict[str, List[str]] = defaultdict(list)
        for r in rows:
            attrs = r.get("attributes", {}) or {}
            val = r.get(dim, attrs.get(dim, "unknown"))
            if val is None:
                val = "unknown"
            val = str(val)
            counts[val] += 1
            if len(examples[val]) < max_examples:
                examples[val].append(r.get("case_id", "?"))
        total = len(rows)
        out[dim] = [
            {"value": v,
             "count": c,
             "share": round(c / total, 4) if total else 0.0,
             "example_case_ids": examples[v]}
            for v, c in counts.most_common()
        ]
    return out


def _conf_bin(p) -> str:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "unknown"
    if p < 0.6:
        return "low (<0.6)"
    if p < 0.8:
        return "medium (0.6-0.8)"
    return "high (>=0.8)"


def analyze_false_positives(rows: List[Dict]) -> Dict:
    """Authentic (y_true==0) images predicted AI/manipulated (y_pred==1)."""
    fps = [r for r in rows if r.get("y_true") == 0 and r.get("y_pred") == 1]
    n_auth = sum(1 for r in rows if r.get("y_true") == 0)
    if not fps:
        return {"status": "MEASURED", "n_false_positives": 0,
                "n_authentic": n_auth,
                "note": "no false positives in this evaluation"}
    # attach derived confidence bins for clustering
    for r in fps:
        (r.setdefault("attributes", {}) or {}).setdefault(
            "confidence_bin", _conf_bin(r.get("ai_prob")))
    clusters = _cluster(fps, ["camera", "compression", "resolution",
                              "screenshot", "ood_status", "forensic_pattern",
                              "confidence_bin", "hard_negative_category"])
    return {
        "status": "MEASURED",
        "n_false_positives": len(fps),
        "n_authentic": n_auth,
        "false_positive_rate": round(len(fps) / n_auth, 4) if n_auth else None,
        "clusters": clusters,
        "question": "Which authentic images are incorrectly classified as "
                    "AI-generated or manipulated?",
    }


def analyze_false_negatives(rows: List[Dict]) -> Dict:
    """Real AI/manipulated (y_true==1) images predicted authentic (y_pred==0)."""
    fns = [r for r in rows if r.get("y_true") == 1 and r.get("y_pred") == 0]
    n_pos = sum(1 for r in rows if r.get("y_true") == 1)
    if not fns:
        return {"status": "MEASURED", "n_false_negatives": 0,
                "n_positive": n_pos,
                "note": "no false negatives in this evaluation"}
    for r in fns:
        (r.setdefault("attributes", {}) or {}).setdefault(
            "confidence_bin", _conf_bin(r.get("ai_prob")))
    clusters = _cluster(fns, ["generator_family", "post_processing",
                              "ood_status", "compression", "resolution",
                              "confidence_bin", "class"])
    return {
        "status": "MEASURED",
        "n_false_negatives": len(fns),
        "n_positive": n_pos,
        "false_negative_rate": round(len(fns) / n_pos, 4) if n_pos else None,
        "clusters": clusters,
        "question": "Which real AI-generated/manipulated images are "
                    "incorrectly classified as authentic?",
    }


def analyze_inconclusive(rows: List[Dict]) -> Dict:
    """INCONCLUSIVE verdicts: count, rate, and reason clusters."""
    inc = [r for r in rows if r.get("verdict") == "INCONCLUSIVE"]
    n = len(rows)
    reasons = Counter()
    for r in rows:
        if r.get("verdict") == "INCONCLUSIVE":
            reason = (r.get("verdict_reason") or "reason not recorded")[:120]
            reasons[reason] += 1
    ood_dist = Counter(str(r.get("ood_status", "unknown")) for r in inc)
    return {
        "status": "MEASURED",
        "n_inconclusive": len(inc),
        "n_total": n,
        "inconclusive_rate": round(len(inc) / n, 4) if n else None,
        "reason_clusters": [{"reason": k, "count": v} for k, v in
                            reasons.most_common(10)],
        "ood_distribution": dict(ood_dist),
        "note": ("INCONCLUSIVE is a first-class scientific outcome, not a "
                 "failure. Tracked as a legitimate metric."),
    }


def analyze_failures(rows: List[Dict]) -> Dict:
    """Combined failure analysis."""
    return {
        "false_positives": analyze_false_positives(rows),
        "false_negatives": analyze_false_negatives(rows),
        "inconclusive": analyze_inconclusive(rows),
        "n_rows": len(rows),
        "status": "MEASURED" if rows else "NOT_MEASURED",
    }
