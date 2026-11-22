"""Stage 2 browser regression tests (S2-1..S2-4).

Runs the REAL frontend (vite dev on loopback, current source — not the
stale served bundle) in Chromium via Playwright, intercepting the API
with pipeline-generated reports. This is a browser test, not static
inspection: a passing run means the page actually rendered the withheld
ML state without JavaScript exceptions and without presenting a
suppressed probability as calibrated.

All probe images are PIL synthetics; no pilot images are opened. The
vite server binds loopback only and is terminated by the fixture. No
repo files are written by these tests. Pipeline runs go through the
test suite's conftest isolation.
"""

import io
import json
import os
import random

import pytest
from PIL import Image

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


def _inject_nan_scores(monkeypatch, names):
    for a in pipeline.ANALYZERS:
        if a.name not in names:
            continue
        orig = a.analyze

        def bad(ctx, orig=orig):
            r = orig(ctx)
            r.ai_score = float("nan")
            r.manip_score = float("nan")
            r.confidence = float("nan")
            return r

        monkeypatch.setattr(a, "analyze", bad)


def _unable_report(monkeypatch):
    _inject_nan_scores(monkeypatch, {a.name for a in pipeline.ANALYZERS})
    data = open(_noise_png(probe_path("stage2_probe.png")), "rb").read()
    return pipeline.run_analysis("noise.png", "image/png", data)


def _thin_report(monkeypatch):
    _inject_nan_scores(monkeypatch, {a.name for a in pipeline.ANALYZERS} - {"noise"})
    data = open(_noise_png(probe_path("stage2_probe.png")), "rb").read()
    return pipeline.run_analysis("noise.png", "image/png", data)


def _numeric_report():
    data = open(_noise_png(probe_path("stage2_probe.png")), "rb").read()
    return pipeline.run_analysis("noise.png", "image/png", data)


def _raw_ml_probability(report):
    dets = [d for d in report["ensemble"]["detectors"]
            if d.get("probability_ai") is not None]
    assert dets, "expected a detector entry with a raw probability"
    return dets[0]["probability_ai"]


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


# ---------------------------------------------------------------- S2-1

def test_suppressed_ml_dashboard_withholds_not_calibrated(browser, vite_server, tmp_path, monkeypatch):
    rep = _unable_report(monkeypatch)
    ens = rep["ensemble"]
    assert ens["ml_available"] is False
    raw = _raw_ml_probability(rep)
    body, errors = _analyze_page(browser, vite_server, rep, tmp_path)
    assert errors == [], f"page threw: {errors}"
    assert "ML probability withheld" in body
    assert "No usable forensic signals" in body
    # The raw detector output must never appear as a calibrated value.
    assert f"Calibrated P(AI) {raw:.1f}%" not in body
    # And the model must not be misdescribed as unconfigured.
    assert "MODEL_NOT_CONFIGURED" not in body


def test_thin_coverage_ml_dashboard_withholds(browser, vite_server, tmp_path, monkeypatch):
    rep = _thin_report(monkeypatch)
    ens = rep["ensemble"]
    assert ens["ml_available"] is False
    raw = _raw_ml_probability(rep)
    body, errors = _analyze_page(browser, vite_server, rep, tmp_path)
    assert errors == [], f"page threw: {errors}"
    assert "ML probability withheld" in body
    assert "Only 1 of 10" in body
    assert f"Calibrated P(AI) {raw:.1f}%" not in body
    assert "MODEL_NOT_CONFIGURED" not in body


# ---------------------------------------------------------------- S2-2

def test_all_unable_banner_reason_is_neutral(browser, vite_server, tmp_path, monkeypatch):
    rep = _unable_report(monkeypatch)
    assert rep["ensemble"]["verdict"] == "INCONCLUSIVE"
    body, errors = _analyze_page(browser, vite_server, rep, tmp_path)
    assert errors == [], f"page threw: {errors}"
    assert "Insufficient usable forensic evidence" in body
    assert "Resolution too low" not in body


# ---------------------------------------------------------------- S2-4 / valid ML

def test_valid_ml_dashboard_renders_calibrated(browser, vite_server, tmp_path):
    rep = _numeric_report()
    ens = rep["ensemble"]
    assert ens["ml_available"] is True
    assert ens["ml_ai_probability"] is not None
    det = next(d for d in ens["detectors"] if d.get("available"))
    body, errors = _analyze_page(browser, vite_server, rep, tmp_path)
    assert errors == [], f"page threw: {errors}"
    assert "Calibrated P(AI)" in body
    assert f"{det['probability_ai']:.1f}%" in body
