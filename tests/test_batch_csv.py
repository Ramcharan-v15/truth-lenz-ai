"""Batch CSV export regression test — vite dev + Chromium, no inference.

The batch API response is intercepted in the browser, so no detector
runs and no image is sent to a backend. Probe images are PIL
synthetics in tmp_path. This test proves two confirmed defects are
fixed: (1) a filename containing a comma or quote must not break
CSV structure (cells are quoted, quotes doubled), and (2) missing
numeric values must render as 'unavailable', never blank, so a
blank cell cannot be mistaken for an empty value.
"""

import json
import os

import pytest
from PIL import Image

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402
from portable_servers import start_vite_server, stop_process

PORT = 5222
BASE = f"http://localhost:{PORT}"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BATCH_RESPONSE = {
    "results": [
        {
            "filename": "a,b.png", "verdict": "LIKELY_AUTHENTIC",
            "ai_probability": None, "manipulation_probability": 12.3,
            "confidence": None, "image_quality": "Good", "dimensions": "64x48",
        },
        {
            "filename": 'a"b.png', "verdict": "INCONCLUSIVE",
            "ai_probability": 45.0, "manipulation_probability": None,
            "confidence": 70.0, "image_quality": None, "dimensions": "64x48",
        },
    ],
    "count": 2,
}


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


def test_batch_csv_escapes_and_marks_missing(browser, vite_server, tmp_path):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("**/api/batch", lambda r: r.fulfill(
        status=200, content_type="application/json", body=json.dumps(BATCH_RESPONSE)))
    try:
        page.goto(f"{vite_server}/batch", wait_until="networkidle")
        png = tmp_path / "batch.png"
        Image.new("RGB", (64, 48), color=(20, 90, 160)).save(png, format="PNG")
        page.locator("input[type=file]").first.set_input_files(str(png))
        page.get_by_role("button", name="Analyze 1 image").click()
        page.wait_for_selector("text=Download CSV", timeout=15000)
        with page.expect_download() as dl_info:
            page.get_by_role("button", name="Download CSV").click()
        download = dl_info.value
        text = open(download.path(), encoding="utf-8").read()
        assert text.startswith("filename,verdict,ai_probability,manipulation_probability,confidence,image_quality,dimensions")
        # Comma filename must be quoted; nulls must be 'unavailable', never blank.
        assert '"a,b.png",LIKELY_AUTHENTIC,unavailable,12.3,unavailable,Good,64x48' in text
        # Quote filename must have doubled quotes.
        assert '"a""b.png",INCONCLUSIVE,45,unavailable,70,unavailable,64x48' in text
        assert errors == [], f"page threw: {errors}"
    finally:
        page.close()
