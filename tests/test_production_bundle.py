"""Production-bundle verification tests.

Serves the REAL tracked frontend/dist bundle through the REAL backend
(uvicorn on loopback 127.0.0.1 only, static serving path — no deployment
configuration changed) and drives it in Chromium via Playwright with
intercepted API responses. This proves the served bundle — not vite dev —
renders the Stage 1/2 behavior.

Safety: the backend only serves static files in these tests; every API
response is intercepted in the browser, so no real analysis runs, no
history write happens, and no protected counter, database, registry, or
manifest is touched. All probe images are PIL synthetics; no pilot
images are opened. The server binds loopback only and is terminated by
the fixture.
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
from portable_servers import probe_path, start_backend_server, stop_process

PORT = 8001
BASE = f"http://127.0.0.1:{PORT}"
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
    data = open(_noise_png(probe_path("stage2_prod_probe.png")), "rb").read()
    return pipeline.run_analysis("noise.png", "image/png", data)


def _numeric_report():
    data = open(_noise_png(probe_path("stage2_prod_probe.png")), "rb").read()
    return pipeline.run_analysis("noise.png", "image/png", data)


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def prod_server():
    proc = start_backend_server(REPO, PORT)
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


def _analyze_page(browser, prod_server, report, tmp_path):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("**/api/analyze*", lambda r: r.fulfill(
        status=200, content_type="application/json", body=json.dumps(report)))
    page.goto(f"{prod_server}/analyze", wait_until="networkidle")
    page.locator("input[type=file]").first.set_input_files(
        _noise_png(str(tmp_path / "up.png")))
    page.get_by_role("button", name="Analyze Image").click()
    page.wait_for_selector("text=AI Generation", timeout=30000)
    page.wait_for_timeout(1500)
    body = page.locator("body").inner_text()
    page.close()
    return body, errors


# ---------------------------------------------------------------- tests

def test_served_bundle_withholds_suppressed_ml(browser, prod_server, tmp_path, monkeypatch):
    """The served bundle must show the Stage 2 withheld panel — the old
    bundle showed the raw value as calibrated here instead."""
    rep = _unable_report(monkeypatch)
    ens = rep["ensemble"]
    assert ens["ml_available"] is False
    raw = next(d["probability_ai"] for d in ens["detectors"]
               if d.get("probability_ai") is not None)
    body, errors = _analyze_page(browser, prod_server, rep, tmp_path)
    assert errors == [], f"page threw: {errors}"
    assert "ML probability withheld" in body
    assert f"Calibrated P(AI) {raw:.1f}%" not in body
    assert "MODEL_NOT_CONFIGURED" not in body


def test_served_bundle_renders_valid_ml(browser, prod_server, tmp_path):
    rep = _numeric_report()
    ens = rep["ensemble"]
    assert ens["ml_available"] is True
    det = next(d for d in ens["detectors"] if d.get("available"))
    body, errors = _analyze_page(browser, prod_server, rep, tmp_path)
    assert errors == [], f"page threw: {errors}"
    assert "Calibrated P(AI)" in body
    assert f"{det['probability_ai']:.1f}%" in body
