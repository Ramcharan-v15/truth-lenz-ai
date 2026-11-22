# ml/models — model artifact index

Trained model artifacts live here, one JSON file per model version:

```
ml/models/
  tl-forensic-stacking-v1.json   # weights + metadata (planned)
```

Artifact schema (planned):

```json
{
  "name": "tl-forensic-stacking-v1",
  "version": "0.1.0",
  "architecture": "l2-logistic-regression",
  "feature_dim": 40,
  "features": ["metadata_ai_score", "..."],
  "weights": [...],
  "bias": 0.0,
  "preprocessing": {"normalization": "standard-scaler", "...": "..."},
  "calibration": {"method": "temperature_scaling", "params": {"temperature": 1.8}},
  "training": {"seed": 42, "dataset_manifest_sha256": "...", "l2_c": 1.0},
  "license": "generated-artifact — see MODEL_CARD.md",
  "limitations": "demonstration model trained on synthetic data — not validated on real-world imagery"
}
```

**No artifacts exist yet.** When training runs, the artifact is registered in
`models/registry.json` (project root `models/` dir) so the backend
`ModelRegistry` picks it up. Until then the system honestly reports
`MODEL_NOT_CONFIGURED`.
