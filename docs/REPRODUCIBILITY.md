# Reproducibility

Every experiment and analysis records what is needed to reproduce it.

## Analysis reproducibility

Each report's `reproducibility` block contains:

- case_id, analysis_id
- image_sha256, perceptual_hash (dHash)
- app_version, backend_version
- config_version, config_sha256
- detector name/version/status/calibration per model
- analyzed_at timestamp
- pipeline_ms
- report_hash (SHA-256 over canonical JSON, computed last)

Re-running the same image bytes through the same app/backend/config/model
versions reproduces the report (modulo timestamps and case IDs).

## Experiment reproducibility (spec §31)

Each experiment (`EXP-YYYY-NNNNNN`) stores:

- experiment_id, dataset_id, dataset_version, dataset_hash
- model_id, model_version, model_hash
- configuration_hash, random_seed, software_version
- timestamp, metrics, results_hash

Stored in `validation/results/{experiment_id}.json`.
API: `GET /api/experiments`, `GET /api/experiments/{id}`.

## Dataset reproducibility

- `datasets/manifests/*/manifest.json` with dataset_hash
- `hashes.jsonl` per image (synthetic)
- Deterministic generation: same seed → byte-identical images
- Blind sets: sealed manifest + hash; integrity verifiable

## What cannot be reproduced exactly

- Wall-clock timestamps and case/experiment IDs (sequential counters)
- Any run involving randomness without a recorded seed (none in the pipeline;
  all RNG is seeded)

## Verification

`GET /api/validation/{experiment_id}` returns the full bundle.
Recompute `results_hash` from metrics + reproducibility to verify integrity.
