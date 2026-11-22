# Truth Lenz AI — Limitations

Consolidated statement of what this system cannot do. Read before relying on any verdict.

## Fundamental limits

1. **No AI image detector is perfectly reliable.** Adversarial generators evolve;
   today's signals may not detect tomorrow's models.
2. **Results are probabilistic estimates, not proof.** Every number is an estimate
   with a separate confidence value. An individual detector's raw output may
   reach 100% (for example, a demonstration model's output on
   out-of-distribution input); such a value is not, by itself, a calibrated
   probability or a validated real-world AI-detection result. The ensemble's
   displayed probabilities are bounded to the 1–99% range and must still be
   read as estimates, never as certainty.
3. **INCONCLUSIVE is a legitimate answer.** Conflicting or thin evidence produces
   INCONCLUSIVE rather than a forced classification.

## What degrades accuracy

- **Compression**: heavy JPEG recompression destroys the fine-grained signals
  (noise texture, high-frequency fingerprints) the detectors rely on.
- **Resizing / screenshots**: resampling and screen capture alter noise and
  frequency characteristics; screenshots of real photos can look synthetic.
- **Editing**: denoising, sharpening, HDR, and heavy color grading change the
  statistical profile of real photographs.
- **Metadata stripping**: messaging apps and social platforms strip EXIF/C2PA;
  absence of metadata is never evidence of AI generation (and is not treated as such).
- **Unseen generators**: the demonstration ML model was trained on synthetic
  procedural data; it has not been validated on real camera photos or commercial
  generators (Midjourney, DALL·E, Firefly, etc.).

## Model-specific limits (tl-forensic-stacking v1)

- Demonstration model trained on **synthetic data only**. Metrics on the held-out
  synthetic test set do not transfer to real-world imagery.
- The model reuses forensic analyzer outputs as features; it cannot see anything
  the analyzers did not measure.
- Calibration was fitted on synthetic data; real-world calibration is unknown.

## Provenance limits

- C2PA manifests are parsed heuristically; **cryptographic signatures are not
  validated** in this deployment. A manifest marked "unvalidated" is context, not proof.
- Metadata (EXIF, software tags) can be fabricated or stripped.

## Appropriate use

- Suitable for: education, research experimentation, journalistic triage,
  engineering demonstration.
- **Not suitable as the sole basis** for legal, academic-integrity, employment,
  financial, or other high-stakes decisions.
- When in doubt, seek corroborating evidence from independent sources.


---

# V5 Additions (2026-10-06)

- **Generator drift**: new generator versions may produce different artifacts;
  no drift monitoring data exists yet.
- **Unseen generators**: performance on unseen generators is NOT MEASURED.
- **Camera diversity**: one real dataset (Columbia, 2004-era cameras); modern
  smartphone/computational photography NOT MEASURED.
- **Dataset bias**: single real-world dataset (splicing only); no real sensor/
  lens/EXIF diversity beyond Columbia.
- **Compression**: heavy compression degrades forensic features (measured on synthetic).
- **Screenshots**: screenshot pipeline effects on real photos NOT MEASURED.
- **Computational photography**: HDR/night-mode/portrait effects NOT MEASURED.
- **False positives**: real-world population FPR measured 0.055 (51/921) on
  Columbia splicing ONLY; not generalizable.
- **False negatives**: real-world population FNR measured 0.000 (0/910) on
  Columbia splicing ONLY; decisive-subset specificity was 0.0 (system never
  confidently predicts "authentic").
- **OOD behavior**: all 1831 Columbia images flagged MILD_SHIFT; OOD×correctness
  correlation trivial (single OOD level observed).
- **Calibration drift**: real-world ECE 0.148 (decisive subset, n=207);
  calibration fitted on synthetic data.
- **Attribution limitations**: ATTRIBUTION_UNAVAILABLE — no attribution model exists.
- **Sample-size limitations**: real-world n=1831 (one dataset, one manipulation
  type). AI-generation detection on real data: n=0, NOT MEASURED.
  Cross-dataset validation: NOT MEASURED.
