"""Generalization scorecard (V5).

Compares evaluation coverage across dataset types, generators, splits and
subsets. Every cell is generated from actual experiment artifacts — never
placeholder numbers. Statuses: MEASURED | NOT_MEASURED | INSUFFICIENT_SAMPLE.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .. import validation as _val

ROWS = [
    ("synthetic_heldout", "Synthetic held-out"),
    ("real_authentic", "Real authentic"),
    ("real_ai", "Real AI-generated"),
    ("real_manipulation", "Real manipulated"),
    ("seen_generators", "Seen generators"),
    ("unseen_generators", "Unseen generators"),
    ("source_heldout", "Source-held-out"),
    ("dataset_heldout", "Dataset-held-out"),
    ("hard_negatives", "Hard negatives"),
    ("screenshot_subset", "Screenshot subset"),
    ("compression_subset", "Compression subset"),
    ("ood_subset", "OOD subset"),
    ("blind_test", "Blind test"),
]

MIN_N = 30  # below this -> INSUFFICIENT_SAMPLE


def _not_measured(row_id: str, label: str, note: str) -> Dict:
    return {"id": row_id, "label": label, "status": "NOT_MEASURED",
            "value": None, "experiment_id": None, "n": 0, "note": note}


def _measured(row_id: str, label: str, rep: Dict, exp_id: str,
              metric: str = "accuracy") -> Dict:
    n = int(rep.get("n", 0) or 0)
    status = "MEASURED" if n >= MIN_N else "INSUFFICIENT_SAMPLE"
    val = (rep.get("metrics", {}) or {}).get(metric, {})
    acc = val.get("value") if isinstance(val, dict) else val
    ci = (rep.get("confidence_intervals", {}) or {}).get("accuracy_bootstrap", {})
    return {
        "id": row_id, "label": label, "status": status,
        "value": {"metric": metric, "accuracy": acc,
                  "accuracy_ci": [ci.get("lo"), ci.get("hi")]
                  if ci.get("lo") is not None else None},
        "experiment_id": exp_id, "n": n,
        "note": ("measured on this experiment's data"
                 if status == "MEASURED"
                 else f"INSUFFICIENT SAMPLE SIZE — n={n} < {MIN_N}"),
    }


def _latest(experiments: List[Dict]) -> Optional[Dict]:
    if not experiments:
        return None
    return max(experiments,
               key=lambda e: str((e.get("reproducibility") or {}).get("created_at", "")))


def build_scorecard(experiment_ids: Optional[List[str]] = None) -> Dict:
    """Build the scorecard from experiment artifacts."""
    ids = experiment_ids or [e["experiment_id"]
                             for e in _val.list_experiments()]
    exps = [r for r in (_val.load_experiment(i) for i in ids) if r]
    metrics_of = {e["experiment_id"]: (e.get("metrics") or {}) for e in exps}

    def find(pred):
        cands = [e for e in exps if pred(e, metrics_of[e["experiment_id"]])]
        return _latest(cands)

    cells: Dict[str, Dict] = {}

    # synthetic held-out
    e = find(lambda e, m: m.get("dataset_type") == "synthetic" and m.get("overall"))
    cells["synthetic_heldout"] = (
        _measured("synthetic_heldout", "Synthetic held-out",
                  metrics_of[e["experiment_id"]]["overall"], e["experiment_id"])
        if e
        else _not_measured("synthetic_heldout", "Synthetic held-out",
                           "no synthetic benchmark experiment recorded"))

    # real class groups
    for row_id, cls in (("real_authentic", "authentic"),
                        ("real_ai", "ai_generated"),
                        ("real_manipulation", "manipulated")):
        e = find(lambda e, m, c=cls: m.get("dataset_type") == "real"
                 and ((m.get("groups") or {}).get("class") or {}).get(c))
        if e:
            rep = metrics_of[e["experiment_id"]]["groups"]["class"][cls]
            cells[row_id] = _measured(row_id, dict(ROWS)[row_id], rep,
                                      e["experiment_id"])
        else:
            cells[row_id] = _not_measured(row_id, dict(ROWS)[row_id],
                                          "no real-world experiment with this "
                                          "class evaluated")

    # generators
    for row_id, key in (("seen_generators", "seen"),
                        ("unseen_generators", "unseen")):
        e = find(lambda e, m, k=key: ((m.get("generators") or {}).get(k) or {})
                 .get("status") == "MEASURED")
        if e:
            g = metrics_of[e["experiment_id"]]["generators"][key]
            cells[row_id] = {
                "id": row_id, "label": dict(ROWS)[row_id],
                "status": "MEASURED" if g.get("n", 0) >= MIN_N else "INSUFFICIENT_SAMPLE",
                "value": {"metric": "accuracy", "accuracy": g.get("accuracy"),
                          "accuracy_ci": [g.get("accuracy_ci", {}).get("lo"),
                                          g.get("accuracy_ci", {}).get("hi")]},
                "experiment_id": e["experiment_id"], "n": g.get("n", 0),
                "note": f"families: {', '.join(metrics_of[e['experiment_id']]['generators'].get('families_evaluated', [])) or 'none listed'}",
            }
        else:
            cells[row_id] = _not_measured(row_id, dict(ROWS)[row_id],
                                          "no generator-held-out evaluation recorded")

    # split types
    for row_id, stype in (("source_heldout", "source_heldout"),
                          ("dataset_heldout", "dataset_heldout")):
        e = find(lambda e, m, s=stype: (m.get("split_config") or {}).get("type") == s)
        if e and metrics_of[e["experiment_id"]].get("overall"):
            cells[row_id] = _measured(
                row_id, dict(ROWS)[row_id],
                metrics_of[e["experiment_id"]]["overall"], e["experiment_id"])
        else:
            cells[row_id] = _not_measured(row_id, dict(ROWS)[row_id],
                                          f"no {stype.replace('_', ' ')} experiment recorded")

    # hard negatives
    e = find(lambda e, m: (m.get("hard_negatives") or {}).get("status") == "MEASURED")
    if e:
        hn = metrics_of[e["experiment_id"]]["hard_negatives"]
        cells["hard_negatives"] = {
            "id": "hard_negatives", "label": "Hard negatives",
            "status": "MEASURED" if hn.get("n", 0) >= MIN_N else "INSUFFICIENT_SAMPLE",
            "value": {"metric": "false_positive_rate",
                      "false_positive_rate": hn.get("false_positive_rate")},
            "experiment_id": e["experiment_id"], "n": hn.get("n", 0),
            "note": "FPR on authentic hard negatives",
        }
    else:
        cells["hard_negatives"] = _not_measured("hard_negatives", "Hard negatives",
                                                "no hard-negative evaluation recorded")

    # attribute subsets
    def attr_subset(row_id, attr, val):
        e = find(lambda e, m, a=attr, v=val:
                 ((m.get("groups") or {}).get("attributes") or {})
                 .get(a, {}).get(v, {}).get("status") == "MEASURED")
        if e:
            rep = metrics_of[e["experiment_id"]]["groups"]["attributes"][attr][val]
            return _measured(row_id, dict(ROWS)[row_id], rep, e["experiment_id"])
        return _not_measured(row_id, dict(ROWS)[row_id],
                             f"no {val} {attr} subset evaluated")

    cells["screenshot_subset"] = attr_subset("screenshot_subset", "screenshot", "yes")
    cells["compression_subset"] = attr_subset("compression_subset", "compression", "heavy")

    # OOD subset
    e = find(lambda e, m: ((m.get("groups") or {}).get("ood_status") or {})
             .get("OUT_OF_DISTRIBUTION", {}).get("status") == "MEASURED")
    cells["ood_subset"] = (
        _measured("ood_subset", "OOD subset",
                  metrics_of[e["experiment_id"]]["groups"]["ood_status"]["OUT_OF_DISTRIBUTION"],
                  e["experiment_id"])
        if e else _not_measured("ood_subset", "OOD subset",
                                "no OUT_OF_DISTRIBUTION subset evaluated"))

    # blind test
    e = find(lambda e, m: m.get("blind") is True and m.get("overall"))
    cells["blind_test"] = (
        _measured("blind_test", "Blind test",
                  metrics_of[e["experiment_id"]]["overall"], e["experiment_id"])
        if e else _not_measured("blind_test", "Blind test",
                                "no sealed blind-test experiment recorded"))

    ordered = [cells[r] for r, _ in ROWS]
    measured = sum(1 for c in ordered if c["status"] == "MEASURED")
    return {
        "status": "MEASURED",
        "rows": ordered,
        "summary": {
            "n_measured": measured,
            "n_not_measured": sum(1 for c in ordered if c["status"] == "NOT_MEASURED"),
            "n_insufficient": sum(1 for c in ordered if c["status"] == "INSUFFICIENT_SAMPLE"),
        },
        "note": ("Each cell derives from a recorded experiment artifact. "
                 "NOT_MEASURED means no such evaluation exists — never imputed."),
    }
