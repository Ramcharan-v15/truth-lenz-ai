# TRUTH LENZ AI

**See beyond the pixels.** A web app that inspects an image and explains, signal by signal,
whether it shows signs of editing or AI generation.

> **Honest status — please read.** This is a **demonstration-grade research prototype**.
> It uses a heuristic forensic ensemble, **not a trained AI model**. AI-generation detection
> is **NOT validated** on real-world data, so treat every verdict as a hint, not proof.
> When signals disagree the app says **INCONCLUSIVE** instead of guessing.

## What it does
- Runs **10 independent forensic analyzers** on an uploaded image: metadata, compression (JPEG/ELA),
  noise residuals, frequency spectrum, pixel statistics, colour/lighting consistency,
  manipulation cues, AI-artifact cues, image quality and more.
- Combines them with an **opinion-weighted ensemble**. Probabilities are clamped to 1–99 % —
  the app never claims 100 % certainty.
- Shows the evidence: heatmaps, histograms, an evidence graph and a per-signal scorecard.
- Extra pages: compare two images, batch analysis with CSV export, history, PDF reports,
  a "learn" section on how image forensics works, privacy and settings, plus a demo mode.
- Includes dataset/validation tooling (hashing, duplicate detection, leakage checks,
  group-level splits) built to evaluate the system honestly in future.

## Tech stack
- **Backend:** Python, FastAPI, NumPy, SciPy, Pillow, ReportLab
- **Frontend:** React + TypeScript (Vite)
- **Tests:** pytest

## Run it locally
```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

cd frontend && npm install && npm run build         # served by the backend
```
Open http://127.0.0.1:8000. Run tests with `python -m pytest tests -q`.

Local use only: there is **no authentication**, so never expose the server on a network.

## Limitations
- No trained ML detector ships with this repo; the optional ML provider falls back to a transparent null detector.
- Detection accuracy has **not been measured** on real-world AI images — no accuracy numbers are claimed.
- Datasets, validation records and images are intentionally **not** included in this repository.

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) and [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Author
Vannala Ramcharan — B.Tech CSE, Vardhaman College of Engineering
