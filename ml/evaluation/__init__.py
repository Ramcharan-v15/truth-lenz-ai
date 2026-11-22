"""ml.evaluation — held-out evaluation of the stacking model.

Evaluates ONLY on the test split (never train, never calibration).

Planned:
  - ml/evaluation/evaluate.py — accuracy, precision, recall, F1, specificity,
    FPR, FNR, ROC-AUC, PR-AUC, confusion matrix, calibration error (ECE),
    per-class and per-variant metrics. Reuses backend/benchmark/metrics.py.
  - ml/evaluation/robustness.py — transformation-stability on the test set,
    reusing backend/benchmark/robustness.py transforms.

No benchmark numbers exist until these scripts are run against a trained
model. Do not hardcode or invent metrics.
"""
