# Model Card — tl-forensic-stacking v1.0.0 (V4)

## 1. Model overview

L2-regularized logistic regression on forensic analyzer outputs.
Demonstration/research-grade. Lifecycle status: **DEMONSTRATION**.

## 2. Intended use

Demonstration and research evaluation of the hybrid ensemble.
Shows the ML subsystem works end-to-end. **Not for high-stakes decisions.**

## 3. Architecture

L2 logistic regression (L-BFGS, C=1.0), 20 features after variance threshold
(1e-3), z-score standardization, temperature scaling (T=0.2184). Weights: JSON
(no pickle). Inference: dot product + sigmoid.

## 4. Training data

Synthetic procedural renders, `ml/datasets` (dataset seed 42).
780 images: train 460 / calibration 160 / test 160.
Classes: authentic, ai_generated, manipulated, hard_negatives, ai_edited.

## 5. Dataset size

780 images total (256×256). Small by ML standards — stated, not hidden.

## 6. Dataset provenance

Procedurally generated in this repository (`ml/datasets/generate.py`).
No external sources, no copyrighted material, no real photographs.

## 7. Training procedure

`ml/training/train.py --seed 20261006`. Deterministic. Hyperparameters in
`ml/training/config.json`. Training time recorded in model.json.

## 8. Preprocessing

40 forensic features extracted → 20 kept (variance threshold).
Z-score using train-split mean/std. No pixel preprocessing.

## 9. Calibration

Temperature scaling, T=0.2184, fitted on held-out calibration split (n=160).
ECE 0.2719 → 0.1171 (calibration), 0.1047 (test). Raw scores never shown as probabilities.

## 10. Synthetic benchmark

n=160 held-out. Accuracy 0.85, precision 0.9286, recall 0.65, F1 0.7647,
specificity 0.97, FPR 0.03, FNR 0.35, ROC-AUC 0.9822, PR-AUC 0.9658, ECE 0.1047,
Brier (see evaluation artifact).
Source: `ml/experiments/run1/evaluation_report.json`. **Synthetic only.**

## 11. Real-world benchmark

**MEASURED (2026-10-06/07).** See "Real-world metrics" section below for
EXP-2026-000273 (Columbia splicing, n=1831). Covers **splicing/manipulation
detection only**. Real-world AI-generation detection: NOT MEASURED.

## 12. Cross-generator results

**NOT MEASURED.** No multi-generator dataset available.

## 13. Hard-negative results

Synthetic hard negatives (n=30): accuracy 0.90. Real-world: NOT MEASURED.

## 14. Robustness results

Measured on synthetic images (2026-10-06): overall 82.9 (AI) / 94.6 (authentic).
Weakest: JPEG recompress, resize 50%, Gaussian noise. See ROBUSTNESS.md.

## 15. OOD behavior

4-level status (IN_DISTRIBUTION/MILD_SHIFT/OUT_OF_DISTRIBUTION/UNKNOWN).
OOD reduces self-reported confidence (70→55→40). OOD ≠ AI.

## 16. Known limitations

Synthetic-only training. Modest AI recall (0.65). Unvalidated on real generators.
Calibration fitted on synthetic data.

## 17. Failure cases

- Real Stable Diffusion outputs: model under-predicts (out of training distribution)
- Zero-variance features caused |z| explosion (found and fixed 2026-10-06)
- Heavy compression degrades forensic features the model depends on

## 18. Bias analysis

Training data lacks: real sensor noise, lens artifacts, EXIF diversity,
real generator fingerprints, demographic/scene diversity. Performance across
real-world attributes: NOT MEASURED. Where sample size insufficient:
INSUFFICIENT_SAMPLE.

## 19. Security considerations

JSON weights (no code execution). Model path confinement. SHA-256 recorded at load.
No personal data in training.

## 20. License

Project-internal demonstration model.

## 21. Reproducibility instructions

```bash
cd ~/workspace/truth-lenz-ai
.venv/bin/python -m ml.training.train --dataset ml/datasets --out ml/experiments/run1 --seed 20261006
.venv/bin/python -m ml.calibration.calibrate --run ml/experiments/run1
.venv/bin/python -m ml.evaluation.evaluate --run ml/experiments/run1
```

Dataset: `.venv/bin/python ml/datasets/generate.py --seed 42 --out ml/datasets`
(byte-identical reproduction).


---

# V5 Update (2026-10-06)

## V5 experiments
No real-world experiments have been executed. The validation infrastructure
(blind protocol, benchmark engine, scorecard, FP/FN analysis) is implemented
and tested, but no real-world dataset is available.

## Real-world metrics (MEASURED 2026-10-06/07)

**Dataset:** RWD-2026-000001 (Columbia Image Splicing Detection Evaluation
Dataset), n=1831 (921 authentic, 910 spliced). Blind evaluation
EXP-2026-000273. License: research-only, verified. Blind integrity audit:
PASS (10/10) — see V5.1_BLIND_INTEGRITY_AUDIT.md.

**Task:** Splicing/manipulation detection (positive = spliced). This is NOT
AI-generation detection.

| Metric | Value | 95% CI |
|---|---|---|
| Inconclusive rate | 0.887 (1624/1831) | [0.872, 0.901] |
| Decisive predictions | 207/1831 | — |
| Confusion (decisive) | TP=156, FP=51, TN=0, FN=0 | — |
| Accuracy (decisive) | 0.754 | [0.691, 0.807] |
| Balanced accuracy (decisive) | 0.500 | — |
| Precision (decisive) | 0.754 | [0.691, 0.807] |
| Recall/sensitivity (decisive) | 1.000 | [0.982, 1.000] |
| Specificity (decisive) | 0.000 | [0.000, 0.018] |
| F1 (decisive) | 0.860 | [0.806, 0.900] |
| Population FPR (51/921 authentic) | 0.055 | — |
| Population FNR (0/910 spliced) | 0.000 | [0.000, 0.024] |
| Brier score (decisive) | 0.203 | — |
| ECE (decisive, 10 bins) | 0.148 | — |
| OOD status | all 1831 MILD_SHIFT | — |

**Honest interpretation:** The system is conservative on real data (88.7%
inconclusive, correctly recognizing OOD inputs). When decisive, it is
**biased toward "manipulated"**: it never confidently predicts "authentic"
(specificity 0.0), and balanced accuracy on the decisive subset is 0.50
(chance level). The 75% accuracy / 76% precision figures reflect this bias,
not discriminative power. Population FPR of 5.5% is the practically relevant
false-alarm rate. This is the first real-world measurement; it covers
splicing manipulation only, not AI generation. Model remains DEMONSTRATION.

**Historical note:** Synthetic evaluation results (V3/V4/V5) are preserved
in their original sections above. They are not replaced by these real-world
results.

## Calibration (real-world)
NOT MEASURED.

## OOD (real-world)
NOT MEASURED.

## Robustness (real-world)
NOT MEASURED.

## False positives / false negatives (real-world)
NOT MEASURED.

## Unseen-generator results
NOT MEASURED.

## Sample sizes
Synthetic held-out: n=160. Real-world: n=0.

## Confidence intervals
Synthetic metrics predate CI reporting. Real-world: NOT MEASURED.

## Validation dates
No real-world validation performed.

## Model lifecycle status
**DEMONSTRATION** — promotion to RESEARCH/VALIDATED/PRODUCTION requires
measured real-world evidence per `backend/models/promotion_criteria.json`.
All promotion attempts without evidence are denied by the system.

## V5 separation
- Synthetic performance: §10 (measured on synthetic held-out)
- Real-world performance: NOT MEASURED
- OOD performance: NOT MEASURED
These are never combined into a single score.
