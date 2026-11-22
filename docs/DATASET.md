# DATASET.md — Truth Lenz AI V3 Synthetic Dataset

## 1. Purpose

Training, calibrating, and evaluating the V3 forensic-feature stacking model
(`tl-forensic-stacking-v1`, planned). This dataset exists because no labeled,
licensed real-world dataset is available in this environment — and inventing
one, scraping copyrighted images, or silently downloading restricted datasets
would be worse than building an honest synthetic one.

> **SYNTHETIC DATA — READ THIS FIRST**
>
> Every image in this dataset is **procedurally generated** by
> `ml/datasets/generate.py` (numpy + PIL + scipy). There are no real camera
> photographs and no outputs of real AI generators here.
>
> Consequences:
> - A model trained on this data learns to separate *our synthetic "camera-like"
>   renders* from *our synthetic "AI-like" renders*. That is a legitimate
>   demonstration of the training/calibration/evaluation pipeline, and it lets
>   us validate the full V3 machinery end to end.
> - It does **NOT** validate real-world performance. Any metric computed on this
>   dataset is a **synthetic-held-out metric only** and must be reported with
>   that qualifier, every time, without exception.
> - The shipped model card will label the model "demonstration model trained on
>   synthetic data — not validated on real-world imagery."

## 2. Procedural sources

| Component | Method |
|---|---|
| Natural texture | Multi-octave value noise (bilinear-interpolated random grids, 4/8/16/32 cells) |
| Sensor noise | Gaussian with std ∝ √(brightness) — Poisson approximation |
| Scenes | 4 procedural scene types: sky-over-ground, diagonal wash, radial glow, horizontal bands; random per-channel tints |
| Vignette | Radial falloff, 10–20% at corners |
| Synthetic fingerprints | Over-smoothed concentric rings; 16×16 tiled micro-patterns; faint 8px sinusoidal grid (upsampling-like); near-zero noise floors |
| EXIF | Synthetic `Image.Exif` (rotating fake makes/models) on `authentic` only, for pipeline testing |

No external downloads. No copyrighted material. No photographs of people.

## 3. Class definitions

| Class | Label semantics | Generation |
|---|---|---|
| `authentic` | Camera-like | Procedural scenes + sensor noise + vignette; JPEG q75–95; synthetic EXIF |
| `ai_generated` | Synthetic | Over-smooth gradients / tiled patterns / periodic grid; JPEG q88–95 or PNG (alternating) |
| `manipulated` | Edited photo | Authentic render + 48×48 copy-move clone (≥64px displacement) **or** 64×64 splice from a different render; JPEG q90–95 |
| `hard_negatives` | Authentic + processing | Authentic render + one of: heavy JPEG q30, screenshot simulation (0.6× downscale + recompress), strong denoise (σ=2.5 blur), metadata strip. **These are NOT AI and must not score as AI.** |
| `ai_edited` | Synthetic + processing | AI render + one of: JPEG q60, 0.7× resize, 80% center crop. Evaluated separately. |

All images 256×256.

## 4. Split sizes

| Class | train | calibration | test | total |
|---|---|---|---|---|
| authentic | 120 | 40 | 40 | 200 |
| ai_generated | 120 | 40 | 40 | 200 |
| manipulated | 80 | 30 | 30 | 140 |
| hard_negatives | 80 | 30 | 30 | 140 |
| ai_edited | 60 | 20 | 20 | 100 |
| **Total** | **460** | **160** | **160** | **780** |

- **train**: fit model weights.
- **calibration**: fit temperature/Platt/isotonic scaling only — never used for weight fitting.
- **test**: held-out evaluation only — never touched during training or calibration.

## 5. Seed scheme & leakage prevention

Each image's RNG seed is derived arithmetically:

```
img_seed = seed + SPLIT_OFFSET[split] + CLASS_OFFSET[class] + index
SPLIT_OFFSET = {train: 0, calibration: 1_000_000, test: 2_000_000}
CLASS_OFFSET = {authentic: 0, ai_generated: 10_000, manipulated: 20_000,
                hard_negatives: 30_000, ai_edited: 40_000}
```

- Offsets are disjoint by construction: the same `(seed, split, class, index)`
  can never collide with another split. No image — and no near-duplicate render —
  can appear in two splits.
- Derived renders (splice donors) use sub-seeds drawn from the image's own RNG
  stream, so they cannot leak across splits either.
- Fixed iteration order (splits → classes → ascending index) makes the full
  pipeline deterministic: same `--seed` → byte-identical images.
- `manifest.json` records the seed, scheme, per-split file-list hashes;
  `hashes.jsonl` records per-image SHA-256. (`generated_at` is metadata only and
  excluded from the byte-identical guarantee.)

## 6. Preprocessing applied at generation

- All renders produced at exactly 256×256 float, clipped to [0,1], quantized to
  uint8 once at save.
- JPEG quality per class/variant as documented above; PNG for alternating
  `ai_generated` images.
- Resize operations (screenshot sim, ai_edited resize/crop) use PIL BILINEAR.
- No other preprocessing. In particular: no augmentation at generation time.

## 7. Augmentation policy

**None at generation.** Augmentation (if any) happens at *train time* inside
`ml/training/` and is recorded in the training config and experiment log —
never baked silently into the dataset.

## 8. Known biases (stated loudly)

1. **Synthetic ≠ real.** Procedural "camera noise" is a crude stand-in for real
   sensor characteristics (PRNU, demosaicing, lens effects). Real cameras,
   real generators, and real editing pipelines differ in ways this dataset
   cannot capture.
2. **Fingerprint caricature.** The AI class uses exaggerated, stereotyped
   fingerprints (visible grids, tiled patterns). Real generators are subtler;
   a model can ace this dataset and still fail on real diffusion outputs.
3. **Limited scene diversity.** Four procedural scene types vs. the real world.
4. **Single resolution.** All images 256×256; no resolution-robustness signal.
5. **JPEG-only pipeline.** No WebP/HEIC/PNG-camera-RAW variety (except alternating
   PNG in `ai_generated`).
6. **No people, text, or complex semantics.** Forensic signals here are purely
   low-level; semantic anomalies are out of scope.

## 9. Limitations

- Metrics on the `test` split measure **synthetic-held-out generalization only**.
- The dataset is a *pipeline validation artifact*, not a benchmark of
  real-world AI-detection capability.
- Do not publish test-split numbers without the synthetic-data qualifier.
- Regenerating with a different `--seed` produces a different dataset; the seed
  is part of the dataset version identity (record it alongside any trained model).

## 10. Reproduction

```bash
cd ~/workspace/truth-lenz-ai
.venv/bin/python ml/datasets/generate.py --seed 42 --out ml/datasets
```

Expected: 780 images, `manifest.json` + `hashes.jsonl` at the dataset root.
Verify determinism: generate twice into temp dirs and diff `hashes.jsonl`
image hashes (excluding `generated_at`).


---

# V4 Dataset Card (spec §34)

## Composition
- Synthetic: 780 images, 5 classes (authentic, ai_generated, manipulated,
  hard_negatives, ai_edited), 256×256, splits train 460 / cal 160 / test 160.
- Real-world: **none available** (NOT_YET_VALIDATED).

## Source
Procedural generation (`ml/datasets/generate.py`, numpy + PIL + scipy).
No external sources.

## Licensing
Project-internal synthetic data. No copyrighted material.

## Collection process
Deterministic generation, seed 42. Same seed → byte-identical images
(verified: two independent runs, zero hash differences).

## Labeling process
Labels assigned at generation time (ground truth by construction).

## Class definitions
See `ml/datasets/DATASET.md` §2. hard_negatives = authentic renders with
heavy processing (JPEG q30, screenshot sim, denoise, metadata strip).

## Generator families
Synthetic `ai_generated` class uses procedural fingerprints, not real
generators. Generator families: not applicable (see CROSS_GENERATOR.md).

## Transformations
Hard-negative transforms: JPEG q30, screenshot simulation, strong denoise,
metadata strip. Robustness transforms: see ROBUSTNESS.md.

## Train/test policy
Zero leakage: SHA-256 verified no duplicates across splits.
V4 adds: perceptual duplicate detection, near-duplicate blocking,
source-aware splitting, blind test workflow (`datasets/`).

## Leakage prevention
V3: SHA-256 exact duplicate detection. V4: + dHash perceptual detection,
split integrity checks, blocked assignments (`datasets/leakage.py`).

## Known biases
No real sensor noise, lens artifacts, EXIF diversity, real generator
fingerprints, or scene/demographic diversity. Synthetic camera renders are
cleaner than real photos.

## Limitations
Model trained on this data learns synthetic fingerprints, not real-world
forensics. All metrics are synthetic-held-out only.

## Version / hash
Dataset seed 42. Manifest: `ml/datasets/manifest.json`.
V4 manifests: `datasets/manifests/synthetic/manifest.json`.
