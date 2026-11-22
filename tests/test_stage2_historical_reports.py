"""Historical suppressed-ML rendering regression tests (audit finding 1).

A report created under the earlier (V5.1.8.64) suppression implementation
carries ml_suppressed_reason, but its detector entries still say
available=true and lack the Stage 2 suppressed flag. The Research
calibration table and EvidenceGraph must still treat such output as
withheld — never as a ready detector or a calibrated probability.

Runs the REAL frontend (vite dev on loopback, current source) in
Chromium via Playwright, with controlled API fixtures. All probe images
are PIL synthetics; no pilot images are opened. No stored reports are
rewritten, migrated, or deleted: the historical shape is built only as
an in-memory fixture. No repo files are written by these tests.
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
    data = open(_noise_png(probe_path("stage2_hist_probe.png")), "rb").read()
    return pipeline.run_analysis("noise.png", "image/png", data)


def _historical_report(monkeypatch):
    """Reshape a current suppressed report into the historical (64-era)
    shape: the reason stays, but detector entries revert to
    available=true with no suppressed flag, and the raw probability is
    the audit's Case A value of exactly 100.0."""
    rep = _unable_report(monkeypatch)
    ens = rep["ensemble"]
    assert ens["ml_available"] is False
    assert ens["ml_suppressed_reason"]
    for d in ens["detectors"]:
        if d.get("probability_ai") is not None:
            d["available"] = True
            d.pop("suppressed", None)
            d.pop("suppressed_reason", None)
            d["probability_ai"] = 100.0
    return rep


def _numeric_report():
    data = open(_noise_png(probe_path("stage2_hist_probe.png")), "rb").read()
    return pipeline.run_analysis("noise.png", "image/png", data)


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


def _research_page(browser, vite_server, report):
    """Drive /research/<id> with the full report as the research view."""
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("**/api/research/*", lambda r: r.fulfill(
        status=200, content_type="application/json", body=json.dumps(report)))
    page.goto(f"{vite_server}/research/test-id", wait_until="networkidle")
    page.wait_for_selector("text=Calibrated P(AI)", timeout=30000)
    page.wait_for_timeout(1000)
    body = page.locator("body").inner_text()
    # The calibration table is the table whose header names Calibrated P(AI).
    table = page.locator("table", has_text="Calibrated P(AI)").first
    row = table.locator("tr", has_text="tl-forensic-stacking").first
    row_text = row.inner_text()
    page.close()
    return body, row_text, errors


# ---------------------------------------------------------------- tests

def test_historical_suppressed_analyze_and_graph(browser, vite_server, tmp_path, monkeypatch):
    rep = _historical_report(monkeypatch)
    body, errors = _analyze_page(browser, vite_server, rep, tmp_path)
    assert errors == [], f"page threw: {errors}"
    # Dashboard withheld panel (reason-keyed) still works.
    assert "ML probability withheld" in body
    assert "No usable forensic signals" in body
    # The raw value must never appear as calibrated.
    assert "Calibrated P(AI) 100.0%" not in body
    assert "MODEL_NOT_CONFIGURED" not in body
    # EvidenceGraph: branch withheld, leaf withheld — never a ready
    # detector leaf exposing the raw value as calibrated.
    assert "output withheld" in body
    assert "P(AI) 100%" not in body


def test_historical_suppressed_research_table(browser, vite_server, monkeypatch):
    rep = _historical_report(monkeypatch)
    body, row_text, errors = _research_page(browser, vite_server, rep)
    assert errors == [], f"page threw: {errors}"
    assert "ML probability withheld" in body
    assert "withheld" in row_text
    assert "100.0%" not in row_text


def test_current_suppressed_research_table(browser, vite_server, monkeypatch):
    rep = _unable_report(monkeypatch)
    raw = next(d["probability_ai"] for d in rep["ensemble"]["detectors"]
               if d.get("probability_ai") is not None)
    body, row_text, errors = _research_page(browser, vite_server, rep)
    assert errors == [], f"page threw: {errors}"
    assert "withheld" in row_text
    assert f"{raw:.1f}%" not in row_text


def test_valid_ml_research_table(browser, vite_server):
    rep = _numeric_report()
    ens = rep["ensemble"]
    assert ens["ml_available"] is True
    det = next(d for d in ens["detectors"] if d.get("available"))
    body, row_text, errors = _research_page(browser, vite_server, rep)
    assert errors == [], f"page threw: {errors}"
    assert f"{det['probability_ai']:.1f}%" in row_text
    assert "withheld" not in row_text
