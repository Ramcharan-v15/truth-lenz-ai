"""Stage 1 browser regression tests (V5.1.8.67): H1, M6, M7, L11, L12, L13.

Runs the REAL frontend (vite dev on loopback, current source — not the stale
served bundle) in Chromium via Playwright, intercepting the API with
pipeline-generated reports. This is a browser test, not static inspection:
a passing run means the page actually rendered without JavaScript
exceptions. Skipped only when Playwright/Chromium is unavailable.

All probe images are PIL synthetics; no pilot images are opened. The vite
server binds loopback only and is terminated by the fixture. No repo files
are written by these tests.
"""

import io
import json
import math
import os
import random

import pytest
from PIL import Image

from backend.analyzers import ensemble as ens_mod
from backend.services import pipeline

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402
from portable_servers import probe_path, start_vite_server, stop_process

PORT = 5199
BASE = f"http://localhost:{PORT}"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- builders

def _noise_png(path, size=256):
    rng = random.Random(7)
    img = Image.new("RGB", (size, size))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                 for _ in range(size * size)])
    img.save(path, format="PNG")
    return path


def _fallback_report():
    orig = ens_mod.EnsembleEngine.combine

    def boom(self, ctx, results):
        raise RuntimeError("stage1-test forced combine failure")

    ens_mod.EnsembleEngine.combine = boom
    try:
        data = open(_noise_png(probe_path("stage1_probe.png")), "rb").read()
        return pipeline.run_analysis("noise.png", "image/png", data)
    finally:
        ens_mod.EnsembleEngine.combine = orig


def _numeric_report():
    data = open(_noise_png(probe_path("stage1_probe.png")), "rb").read()
    return pipeline.run_analysis("noise.png", "image/png", data)


def _null_technicals_report():
    """A numeric report whose frequency/statistics diagnostics are null —
    exactly the shape the backend's finite boundary produces for
    non-finite diagnostics (V5.1.8.62)."""
    rep = _numeric_report()
    for s in rep["signals"]:
        if s["name"] == "statistics":
            s["technical"]["edge_density"] = None
            s["technical"]["mean_rgb"] = None
            s["technical"]["luminance_entropy_bits"] = None
            s["technical"]["contrast_rms"] = None
        if s["name"] == "frequency":
            s["technical"]["high_freq_energy"] = None
            s["technical"]["spectral_slope"] = None
            s["technical"]["spectral_peak_count"] = None
    return rep


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def vite_server():
    proc = start_vite_server(REPO, PORT)
    try:
        yield BASE
    finally:
        stop_process(proc)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        yield b
        b.close()


def _analyze_page(browser, vite_server, report, tmp_path):
    """Drive /analyze with an intercepted report; return (body, errors)."""
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("**/api/analyze*", lambda r: r.fulfill(
        status=200, content_type="application/json", body=json.dumps(report)))
    page.goto(f"{vite_server}/analyze", wait_until="networkidle")
    page.locator("input[type=file]").first.set_input_files(
        _noise_png(str(tmp_path / "up.png")))
    page.get_by_role("button", name="Analyze Image").click()
    page.wait_for_selector("text=AI Generation", timeout=30000)
    page.wait_for_timeout(1500)
    body = page.locator("body").inner_text()
    page.close()
    return body, errors


# ---------------------------------------------------------------- H1

def test_fallback_report_renders_without_crash_or_misleading_label(browser, vite_server, tmp_path):
    rep = _fallback_report()
    assert rep["ensemble"]["ai_probability"] is None
    assert rep["ensemble"]["signal_disagreement"] is None
    body, errors = _analyze_page(browser, vite_server, rep, tmp_path)
    assert errors == [], f"page threw: {errors}"
    assert len(body) > 500, "page must not render blank on a fallback report"
    assert "unavailable" in body
    assert "Strongly likely authentic" not in body, \
        "a withheld probability must never receive an authentic-leaning band label"
    assert "undefined%" not in body
    assert "NaN%" not in body
    # L13: fallback detector dict carries detector_name, not model_name.
    assert "No trained model configured" in body


def test_numeric_report_still_renders_expected_values(browser, vite_server, tmp_path):
    rep = _numeric_report()
    ai = rep["ensemble"]["ai_probability"]
    assert ai is not None
    body, errors = _analyze_page(browser, vite_server, rep, tmp_path)
    assert errors == [], f"page threw: {errors}"
    assert f"{math.floor(ai + 0.5)}%" in body


# ---------------------------------------------------------------- M6

def test_null_technicals_render_unavailable_not_nan_or_zero(browser, vite_server, tmp_path):
    rep = _null_technicals_report()
    body, errors = _analyze_page(browser, vite_server, rep, tmp_path)
    assert errors == [], f"page threw: {errors}"
    assert "unavailable" in body
    assert "NaN%" not in body
    assert "0.00%" not in body, "a missing diagnostic must not be fabricated as 0.00%"


# ---------------------------------------------------------------- M7 / L11

def test_history_null_rows_and_empty_average(browser, vite_server):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    row = {"id": "TL-2026-000001", "filename": "noise.png",
           "created_at": "2026-10-24", "verdict": "INCONCLUSIVE",
           "ai_probability": None, "manipulation_probability": None,
           "authenticity_probability": None, "confidence": None,
           "image_quality": "Unknown", "width": 256, "height": 256,
           "thumbnail": None}
    page.route("**/api/history", lambda r: r.fulfill(
        status=200, content_type="application/json", body=json.dumps({
            "analyses": [row],
            "stats": {"total": 1, "by_verdict": {"INCONCLUSIVE": 1},
                      "avg_confidence": None}})))
    page.goto(f"{vite_server}/history", wait_until="networkidle")
    page.wait_for_selector("text=noise.png", timeout=30000)
    page.wait_for_timeout(500)
    body = page.locator("body").inner_text()
    assert errors == [], f"page threw: {errors}"
    assert "undefined%" not in body
    assert "unavailable" in body
    avg_card = page.locator("div.card", has_text="Average confidence")
    assert "unavailable" in avg_card.inner_text()
    page.close()

    # Empty history: no fabricated 0% average.
    page = browser.new_page()
    page.route("**/api/history", lambda r: r.fulfill(
        status=200, content_type="application/json", body=json.dumps({
            "analyses": [],
            "stats": {"total": 0, "by_verdict": {}, "avg_confidence": None}})))
    page.goto(f"{vite_server}/history", wait_until="networkidle")
    page.wait_for_selector("text=No saved analyses yet.", timeout=30000)
    avg_card = page.locator("div.card", has_text="Average confidence")
    assert "unavailable" in avg_card.inner_text()
    page.close()


# ---------------------------------------------------------------- L12

def test_batch_nulls_sort_last_in_both_directions(browser, vite_server, tmp_path):
    rows = [
        {"filename": "a.png", "verdict": "LIKELY_AI_GENERATED",
         "ai_probability": 80, "manipulation_probability": 10, "confidence": 70,
         "confidence_label": "HIGH", "image_quality": "high", "dimensions": "256x256"},
        {"filename": "b.png", "verdict": "INCONCLUSIVE",
         "ai_probability": None, "manipulation_probability": None,
         "confidence": None, "confidence_label": "LOW",
         "image_quality": "Unknown", "dimensions": "256x256"},
        {"filename": "c.png", "verdict": "LIKELY_AUTHENTIC",
         "ai_probability": 40, "manipulation_probability": 20, "confidence": 60,
         "confidence_label": "MEDIUM", "image_quality": "high", "dimensions": "256x256"},
    ]
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("**/api/batch", lambda r: r.fulfill(
        status=200, content_type="application/json",
        body=json.dumps({"results": rows, "count": 3})))
    page.goto(f"{vite_server}/batch", wait_until="networkidle")
    page.locator("input[type=file]").first.set_input_files([
        _noise_png(str(tmp_path / "b1.png")),
        _noise_png(str(tmp_path / "b2.png"))])
    page.get_by_role("button", name="Analyze 2 images").click()
    page.wait_for_selector("table.data tbody tr", timeout=30000)

    def ai_column():
        return page.locator("table.data tbody tr td:nth-child(3)").all_inner_texts()

    assert errors == [], f"page threw: {errors}"
    assert ai_column() == ["80", "40", "—"], ai_column()  # default: desc, null last
    page.get_by_text("AI %").first.click()
    page.wait_for_timeout(500)
    assert ai_column() == ["40", "80", "—"], ai_column()  # asc, null STILL last
    page.close()
