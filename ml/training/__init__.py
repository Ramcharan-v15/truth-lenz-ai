"""ml.training — reproducible training of the forensic-feature stacking model.

Planned: ml/training/train.py
  - L2-regularized logistic regression via scipy (CPU, seconds, no GPU needed).
  - Reads ml/datasets/{train,calibration}; writes weights + metadata JSON.
  - Records: random seed, dataset version + manifest hash, architecture,
    hyperparameters, optimizer, training time, validation metrics.

This package currently holds only the hyperparameter config. No training has
been run; no weights exist. See ml/experiments/README.md for run logging.
"""
