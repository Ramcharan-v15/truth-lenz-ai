# TRUTH LENZ AI — Technical Architecture

> "See beyond the pixels." · AI Image Authenticity & Digital Forensics Platform

## 1. What this is

A multi-signal forensic analysis platform that produces **evidence-based, probabilistic**
assessments of whether an image is AI-generated, an authentic camera capture, manipulated,
or inconclusive. It never claims 100% certainty.

**Core principle:** no single detector. Ten independent analyzers produce structured signals;
a weighted ensemble combines them; disagreement yields INCONCLUSIVE rather than a forced verdict.

## 2. What runs locally vs what needs external services

### Fully implemented locally (no external calls)
| Capability | Implementation |
|---|---|
| Metadata forensics | EXIF/PNG-text parsing, AI-generator signature DB, editing-software tags, C2PA byte scan |
| Image statistics | Channel stats, entropy, contrast, edge density, histograms |
| Compression analysis | JPEG quantization table matching, blockiness, recompression hints |
| Noise analysis | Flat-region Laplacian/MAD noise estimation, consistency mapping |
| Frequency analysis | 2D FFT, radial power spectrum, spectral slope, peak detection |
| Pixel forensics | Error Level Analysis (ELA), micro-texture autocorrelation |
| AI artifact heuristics | Symmetry, over-smoothness, pattern repetition (labelled heuristic) |
| Manipulation detection | DCT-hash copy-move with NCC verification + clustering, splicing noise map |
| Quality / screenshot | Blur, resolution tiers, UI-chrome heuristics, AI-size notes |
| Consistency | EXIF internal contradictions, dimension/timestamp checks |
| Ensemble scoring | Opinion-weighted evidence, disagreement detection, confidence model |
| Reports | PDF (reportlab), JSON, CSV |
| History | Opt-in SQLite (thumbnails only, never full images) |

### Designed as interfaces (transparent fallbacks, clearly labelled)
| Capability | Status |
|---|---|
| Trained AI-detection model | **V3**: `tl-forensic-stacking` v1.0.0 registered and active — L2 logistic regression on 20 forensic features, trained on 780 synthetic images (seed 42), temperature-calibrated (T=0.2184), held-out test accuracy 0.85 / ROC-AUC 0.9822 (synthetic only). Demonstration-grade: reduced ensemble weight (×0.5), no verdict veto, OOD guard. Full card in `MODEL_CARD.md`. |
| C2PA credential *validation* | **V2**: states `unvalidated`/`invalid`/`missing`/`unsupported`; heuristic manifest parsing (issuer, generator, actions, AI markers). Cryptographic signature validation remains out of scope — a found manifest is never reported as VERIFIED. |
| Generator-family attribution | `GeneratorClassifier` interface; `UnavailableGeneratorClassifier` fallback ("Generator attribution unavailable"). |
| LLM explanations | `backend/llm/interface.py`: `LLMExplainer` ABC + `DisabledLLMExplainer`. Hard rules: LLM never scores, never invents evidence; outputs validated. Deterministic 3-level explanations (basic/technical/forensic) ship instead. |
| Camera PRNU fingerprinting | Not implemented (requires reference images); documented as future work. |
| Video/audio deepfake detection | Architecture-ready (analyzer interface is media-agnostic in shape); not implemented. |

## 3. Repository layout

```
truth-lenz-ai/
├── backend/
│   ├── main.py                  # FastAPI app, static frontend serving, safe error handler
│   ├── detector_config.json     # weights, thresholds, penalties, quality gates, ensemble_v2
│   ├── api/routes.py            # /api/* endpoints, validation, rate limiting
│   ├── analyzers/
│   │   ├── base.py              # AnalyzerResult schema, BaseAnalyzer, AIDetectorProvider,
│   │   │                        #   AIDetectorResult contract, ModelStatus
│   │   ├── metadata.py          # EXIF / PNG text / AI signatures / C2PA states
│   │   ├── statistics.py        # descriptive stats + histogram payloads
│   │   ├── compression.py       # quantization tables, blockiness
│   │   ├── noise.py             # flat-region noise estimation
│   │   ├── noise_utils.py       # shared flat-region noise helper
│   │   ├── frequency.py         # FFT, spectral slope, peak detection
│   │   ├── pixel_forensics.py   # ELA, texture autocorrelation
│   │   ├── ai_artifacts.py      # heuristic synthetic indicators
│   │   ├── manipulation.py      # copy-move + splicing heatmap
│   │   ├── quality.py           # quality tiers, screenshot detection
│   │   ├── consistency.py       # metadata contradiction checks
│   │   └── ensemble.py          # hybrid ensemble: forensics + ML + provenance,
│   │                            #   disagreement handling, confidence
│   ├── models/                  # V2: ML detector subsystem
│   │   ├── registry.py          # ModelRegistry: register/load/unload/list/status
│   │   ├── adapters.py          # TorchDetectorProvider, ONNXDetectorProvider,
│   │   │                        #   ExternalAPIDetectorProvider (graceful degradation)
│   │   ├── preprocessing.py     # StandardPreprocessor + full PreprocessingRecord
│   │   ├── calibration.py       # temperature / Platt / isotonic scaling
│   │   ├── tiling.py            # tiled inference + spatial confidence map
│   │   └── generator.py         # GeneratorClassifier interface
│   ├── llm/interface.py         # V2: LLM explanation layer (disabled stub + guardrails)
│   ├── benchmark/               # V2: evaluation framework
│   │   ├── metrics.py           # accuracy/precision/recall/F1/ROC-AUC/PR-AUC/ECE
│   │   ├── runner.py            # dataset walker, per-category evaluation
│   │   └── robustness.py        # transformation-stability testing
│   ├── services/
│   │   ├── pipeline.py          # decode → context → analyzers → detectors → ensemble
│   │   │                        #   + reproducibility block + explanations + report hash
│   │   ├── storage.py           # opt-in SQLite history
│   │   ├── reporting.py         # PDF/JSON/CSV (V2: 10-section report)
│   │   └── explain.py           # V2: deterministic basic/technical/forensic explanations
│   └── utils/
│       ├── image_io.py          # safe decoding, numpy conversion, data URIs
│       └── security.py          # V2: safe model loading, SSRF guard, LLM validation
├── models/                      # V2: local model dir (registry.json + README; no models bundled)
├── benchmark/datasets/          # V2: evaluation datasets (sample + README)
├── frontend/                    # Vite + React + TypeScript SPA
│   └── src/
│       ├── pages/               # Home, Analyze, Compare, Batch, History, Research,
│       │                        #   Benchmark, HowItWorks, Learn, Privacy, Settings
│       └── components/          # Layout, Gauge, Histogram, EvidenceCard, ForensicsViewer,
│                                #   ModelStatus, EvidenceGraph, HeatmapLayer, ...
├── tests/                       # pytest suite (74 tests) + synthetic dataset generator
└── test_data/                   # generated synthetic images (no copyrighted material)
```

## 4. Analyzer contract

Every analyzer returns:

```json
{
  "name": "noise",
  "display_name": "Noise Analysis",
  "ai_score": 44.0,
  "manip_score": 50.0,
  "confidence": 75.0,
  "status": "ok",
  "evidence": [{"text": "...", "direction": "supports_authentic", "detail": "..."}],
  "warnings": ["..."],
  "technical": {"noise_std": 0.0222, "...": "..."},
  "duration_ms": 123.4,
  "visuals": {"noise_map": {...}}
}
```

- Scores are 0–100; **50 = neutral**. Higher `ai_score` leans AI-generated; higher
  `manip_score` leans manipulated. `confidence` is the signal's reliability *for this image*.
- `status`: `ok` | `warning` | `unable`. A failed analyzer returns `unable` with 0
  confidence and never kills the pipeline.
- Evidence `direction`: `supports_ai` | `supports_authentic` | `supports_manip` | `neutral` | `uncertain`.

## 5. Ensemble design (V2: hybrid)

1. **Opinion-weighted mean**: each signal's effective weight =
   `base_weight × (0.25 + 1.75 × opinion × confidence)`, where `opinion = |score−50|/50`.
   A signal with no opinion barely counts; a decisive, confident signal dominates.
2. **Hybrid combination**: `P(AI) = (fw·forensic + mw·ml + pw·provenance) / (fw+mw+pw)`
   with `ensemble_v2` weights (`forensics_weight`, `ml_weight`, `provenance_weight`).
   ML is confidence-weighted across all ready detectors.
3. **ML never auto-dominates**: unless `allow_ml_dominance` is explicitly true, the
   effective ML weight is capped at the forensics weight.
4. **Decisive-evidence rule**: a signal ≥75 with ≥60% confidence and *no strong
   contradiction* (no signal ≤35 @ ≥60% on that axis) decides the verdict when the
   ensemble probability also crosses its threshold — unless the ML detector strongly
   disagrees (see 6).
5. **Disagreement → INCONCLUSIVE**: confidence-weighted std-dev of AI scores above the
   configured threshold (18) with probabilities in the uncertain band forces INCONCLUSIVE.
6. **ML/forensics disagreement → uncertainty**: if a ready ML detector differs from the
   forensic P(AI) by more than `ml_disagreement_threshold` (25) at ≥60% confidence,
   a penalty is applied and the verdict can be forced to INCONCLUSIVE rather than
   letting either side silently win.
7. **Confidence ≠ probability**: confidence starts from mean analyzer confidence and is
   reduced by explicit penalties (poor quality, heavy compression, screenshot, missing
   metadata, signal disagreement, ML disagreement, downscaling). Every penalty is listed.
8. **Calibration**: detector raw scores are mapped through a documented calibrator
   (temperature/Platt/isotonic); uncalibrated detectors are labelled as such and their
   influence is capped. Raw scores are never presented as probabilities.
9. **Probabilities are clamped to [1, 99]** — the system can never output 100%.

All weights, thresholds, bands and penalties live in `backend/detector_config.json`
and are surfaced read-only at `GET /api/config` and on the How It Works page.

## 6. API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/analyze?save=true` | Full forensic analysis (multipart image) |
| POST | `/api/compare` | Two-image comparison + pixel difference + ML deltas |
| POST | `/api/batch` | Up to 8 images, summary table |
| GET | `/api/history` | Saved analyses + dashboard stats |
| GET | `/api/analysis/{id}` | Fetch one saved analysis |
| DELETE | `/api/history/{id}` | Delete one saved analysis |
| DELETE | `/api/history` | Wipe history |
| GET | `/api/report/{id}?format=pdf\|json\|csv` | Forensic report export (V2: 10-section PDF) |
| GET | `/api/config` | Public detector configuration |
| GET | `/api/health` | Service + analyzer inventory |
| GET | `/api/models` | ML registry status |
| POST | `/api/models/reload` | Reload `models/registry.json` |
| GET | `/api/research/{id}` | Raw research view |
| POST | `/api/explain?level=` | Deterministic explanation + LLM status |
| POST | `/api/benchmark/run?dataset=` | Run benchmark evaluation |
| GET | `/api/benchmark/results?dataset=` | Benchmark metrics |
| POST | `/api/robustness` | Transformation-stability test |

## 7. Security & privacy

- **Magic-byte validation** — file type is determined from content, never from the
  filename or claimed MIME type. Mismatches are rejected.
- **Size limits** (25 MB), **decompression-bomb protection**, truncated-file rejection.
- **Filename sanitization**; **in-memory processing** — uploads are never written to disk.
- **Nothing persisted** unless `save=true`; history stores thumbnails + JSON only.
- **Rate limiting** (30 analyze/min per IP); **no tracebacks** leak to clients —
  users get actionable messages, technical details stay server-side.
- No external network calls during analysis. No third-party AI API receives images.

## 8. Honesty rules (enforced in code and UI)

- Missing metadata is never AI evidence (`test_missing_metadata_not_ai_evidence`).
- ELA is worded as "potentially suspicious regions", never proof.
- Heuristic AI-artifact signals carry a visible "heuristic, not a trained model" warning.
- Demo mode results are bannered "DEMO RESULT — NOT REAL ANALYSIS" and can't be saved/exported.
- Every dashboard shows probability-vs-confidence separately and a limitations section.

## 9. Testing

`tests/test_analyzers.py` — 30 tests covering file validation, each analyzer,
ensemble behavior (incl. bounded probabilities and confidence penalties), and API
endpoints. Test images are generated synthetically by `tests/generate_test_data.py`.

## 10. Future extension points

- `AIDetectorProvider`: drop in a trained model (local ONNX/HF or remote API) as one
  ensemble signal — no verdict logic changes needed.
- New analyzers: subclass `BaseAnalyzer`, append to `ANALYZERS` in `services/pipeline.py`,
  add weights in `detector_config.json`. The frontend renders new signals generically.
- C2PA validation, PRNU, video/audio: new analyzer modules behind the same contract.
