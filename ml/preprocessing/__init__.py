"""ml.preprocessing — feature extraction for the stacking model.

Converts a Truth Lenz AI analysis report (analyzer outputs) into the fixed
feature vector consumed by the forensic-feature stacking model.

Planned: ml/preprocessing/features.py
  - report -> vector: per-analyzer ai_score/manip_score/confidence
    (10 analyzers x 3 = 30 dims) + key technical values
    (noise_std, spectral_slope, jpeg_quality_estimate, ela_mean,
    edge_density, ...) 
  - The exact feature list and dimension are recorded in the training config
    and model artifact at train time. Nothing here fabricates features.
"""
