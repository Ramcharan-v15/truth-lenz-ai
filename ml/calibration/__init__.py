"""ml.calibration — probability calibration for the stacking model.

Fits the calibration layer (temperature scaling / Platt scaling / isotonic
regression — see backend/models/calibration.py) on the held-out CALIBRATION
split only. Never on train, never on test.

Planned: ml/calibration/calibrate.py
  - Loads trained weights + calibration split features.
  - Fits the calibrator; records method, parameters, calibration dataset
    version, and calibration metrics (ECE before/after).
  - Raw model scores are never presented as probabilities until this step
    has run and been documented.
"""
