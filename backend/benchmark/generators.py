"""Seen vs unseen generator evaluation (V5).

Entire generator families held out of development must be evaluated
separately. A generator family is only reported as evaluated when it has
measured samples — never "Generator X Accuracy: 94%" without measurement.

Rows: dicts with generator_family, y_true, y_prob (0..1), y_pred, verdict.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from . import metrics
from .statistics import bootstrap_ci


def _sub_report(rows: List[Dict], threshold: float = 0.5) -> Optional[Dict]:
    if not rows:
        return None
    yt = np.array([r["y_true"] for r in rows], dtype=int)
    yp = np.array([r["ai_prob"] for r in rows], dtype=float)
    base = metrics.full_evaluation(yt, yp, threshold)
    acc_fn = lambda a, b: metrics.binary_metrics(a, b, threshold)["accuracy"]
    inc = sum(1 for r in rows if r.get("verdict") == "INCONCLUSIVE")
    return {
        "status": "MEASURED",
        "n": len(rows),
        "accuracy": base["accuracy"],
        "f1": base["f1"],
        "roc_auc": base["roc_auc"],
        "brier_score": base["brier_score"],
        "ece": (base.get("calibration") or {}).get("ece"),
        "accuracy_ci": bootstrap_ci(yt, yp, acc_fn),
        "inconclusive_rate": round(inc / len(rows), 4),
    }


def evaluate_generators(rows: List[Dict],
                        seen_families: Sequence[str],
                        threshold: float = 0.5) -> Dict:
    """Compare seen-generator vs unseen-generator performance.

    seen_families: families used during development/tuning. Everything else
    counts as unseen — but only families with measured samples are reported.
    """
    seen_set = set(seen_families or [])
    seen_rows = [r for r in rows if r.get("generator_family") in seen_set]
    unseen_rows = [r for r in rows
                   if r.get("generator_family") not in seen_set
                   and r.get("generator_family") not in (None, "", "unknown")]

    seen = _sub_report(seen_rows, threshold)
    unseen = _sub_report(unseen_rows, threshold)

    # per-family breakdown: only families with measured samples
    per_family: Dict[str, Dict] = {}
    fams = sorted({r.get("generator_family") for r in rows
                   if r.get("generator_family") not in (None, "", "unknown")})
    for fam in fams:
        fr = [r for r in rows if r.get("generator_family") == fam]
        rep = _sub_report(fr, threshold)
        if rep:
            rep["seen_during_development"] = fam in seen_set
            per_family[fam] = rep

    degradation: Dict = {"status": "NOT_MEASURED"}
    if seen and unseen:
        def delta(k):
            s, u = seen.get(k), unseen.get(k)
            if s is None or u is None:
                return None
            return round(s - u, 4)
        degradation = {
            "status": "MEASURED",
            "accuracy_drop": delta("accuracy"),       # positive = worse on unseen
            "f1_drop": delta("f1"),
            "calibration_degradation": delta("ece"),  # positive = worse calibrated
            "brier_degradation": delta("brier_score"),
            "note": ("Positive drops mean worse performance on unseen "
                     "generators. Do not claim generalization unless measured."),
        }

    return {
        "status": "MEASURED" if (seen or unseen) else "NOT_MEASURED",
        "seen": seen or {"status": "NOT_MEASURED",
                         "note": "no measured samples from seen families"},
        "unseen": unseen or {"status": "NOT_MEASURED",
                             "note": "no measured samples from unseen families"},
        "degradation": degradation,
        "per_family": per_family,
        "families_evaluated": sorted(per_family),
        "families_seen": sorted(seen_set),
        "note": ("A family appears here only if it has measured samples. "
                 "Unmeasured families are NOT_MEASURED, never imputed."),
    }
