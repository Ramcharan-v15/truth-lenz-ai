"""Teacher Demo browser tests — vite dev + Chromium, no inference.

All probe images are PIL synthetics generated in tmp_path; no pilot
images are opened, no REAL test images are used, and no detector
inference runs. The /api/models response is intercepted in the
browser (a registry metadata response), so these tests do not depend
on a running backend and never call /api/analyze, /api/compare,
/api/batch, /api/explain, or /api/robustness. The vite server binds
loopback only and is terminated by the fixture.
"""

import os

import pytest
from PIL import Image

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402
from portable_servers import start_vite_server, stop_process

PORT = 5211
BASE = f"http://localhost:{PORT}"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEMO_REGISTRY = {
    "total": 1,
    "ready": 1,
    "models": [
        {
            "name": "tl-forensic-stacking",
            "version": "1.0.0",
            "model_type": "stacking",
            "status": "ready",
            "limitations": "DEMONSTRATION MODEL. Trained on SYNTHETIC procedural data only; not validated on real camera photos or commercial AI generators.",
            "supported_formats": ["image/jpeg", "image/png", "image/webp"],
            "preprocessing": {},
            "lifecycle_status": "DEMONSTRATION",
            "validation_status": "SYNTHETIC_ONLY",
        }
    ],
    "errors": {},
    "ml_available": True,
}


def _synthetic_png(path, size=(64, 48)):
    img = Image.new("RGB", size, color=(20, 90, 160))
    img.save(path, format="PNG")
    return path


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


def _new_page(browser, vite_server):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("**/api/models", lambda r: r.fulfill(
        status=200, content_type="application/json",
        body=__import__("json").dumps(DEMO_REGISTRY)))
    page.goto(f"{vite_server}/teacher-demo", wait_until="networkidle")
    return page, errors


def test_teacher_demo_page_loads_with_status_and_interpretation(browser, vite_server):
    page, errors = _new_page(browser, vite_server)
    try:
        assert errors == [], f"page threw: {errors}"
        body = page.locator("body").inner_text()
        assert "TRUTH LENZ AI" in body
        assert "VALIDATION INCOMPLETE" in body
        assert "How to Interpret Results" in body
        assert "False positive" in body
        assert "False negative" in body
        assert "Inconclusive" in body
        # No Analyze button until a valid image is selected.
        assert page.get_by_test_id("teacher-analyze-button").count() == 0
    finally:
        page.close()


def test_valid_upload_preview_and_model_not_configured(browser, vite_server, tmp_path):
    page, errors = _new_page(browser, vite_server)
    requested = []
    page.on("request", lambda req: requested.append(req.url))
    try:
        png = _synthetic_png(str(tmp_path / "demo.png"))
        page.locator("input[type=file]").first.set_input_files(png)
        page.wait_for_selector('[data-testid="teacher-preview"]', timeout=15000)
        preview = page.get_by_test_id("teacher-preview").inner_text()
        assert "demo.png" in preview
        assert "64×48" in preview
        page.get_by_test_id("teacher-analyze-button").click()
        page.wait_for_selector('[data-testid="teacher-result-status"]', timeout=15000)
        assert page.get_by_test_id("teacher-result-status").inner_text() == "MODEL_NOT_CONFIGURED"
        result = page.get_by_test_id("teacher-result").inner_text()
        assert "Classification is not currently available" in result
        assert "No probability" in result
        assert "%" not in result, "teacher demo must not show a fabricated probability"
        assert errors == [], f"page threw: {errors}"
        # The demo must never call an inference endpoint.
        for forbidden in ("/api/analyze", "/api/compare", "/api/batch", "/api/explain", "/api/robustness"):
            assert not any(forbidden in u for u in requested), f"teacher demo called {forbidden}"
    finally:
        page.close()


def test_unsupported_file_shows_error(browser, vite_server, tmp_path):
    page, errors = _new_page(browser, vite_server)
    try:
        txt = tmp_path / "note.txt"
        txt.write_text("this is not an image", encoding="utf-8")
        page.locator("input[type=file]").first.set_input_files(str(txt))
        page.wait_for_selector('[data-testid="teacher-error"]', timeout=15000)
        assert "Unsupported format" in page.get_by_test_id("teacher-error").inner_text()
        assert page.get_by_test_id("teacher-preview").count() == 0
    finally:
        page.close()


def test_malformed_png_shows_error(browser, vite_server, tmp_path):
    page, errors = _new_page(browser, vite_server)
    try:
        fake = tmp_path / "fake.png"
        fake.write_bytes(b"this is text, not PNG data")
        page.locator("input[type=file]").first.set_input_files(str(fake))
        page.wait_for_selector('[data-testid="teacher-error"]', timeout=15000)
        assert "Unsupported or unrecognized" in page.get_by_test_id("teacher-error").inner_text()
        assert page.get_by_test_id("teacher-preview").count() == 0
    finally:
        page.close()


def test_api_abort_shows_analysis_unavailable(browser, vite_server, tmp_path):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    # Abort model-status requests entirely: the demo must fail safely.
    page.route("**/api/models", lambda r: r.abort())
    page.goto(f"{vite_server}/teacher-demo", wait_until="networkidle")
    try:
        png = _synthetic_png(str(tmp_path / "demo2.png"))
        page.locator("input[type=file]").first.set_input_files(png)
        page.wait_for_selector('[data-testid="teacher-preview"]', timeout=15000)
        page.get_by_test_id("teacher-analyze-button").click()
        page.wait_for_selector('[data-testid="teacher-result-status"]', timeout=15000)
        assert page.get_by_test_id("teacher-result-status").inner_text() == "ANALYSIS_UNAVAILABLE"
        assert "No classification was made" in page.get_by_test_id("teacher-result").inner_text()
    finally:
        page.close()


def test_select_another_resets_demo(browser, vite_server, tmp_path):
    page, errors = _new_page(browser, vite_server)
    try:
        png = _synthetic_png(str(tmp_path / "demo3.png"))
        page.locator("input[type=file]").first.set_input_files(png)
        page.wait_for_selector('[data-testid="teacher-preview"]', timeout=15000)
        page.get_by_test_id("teacher-analyze-button").click()
        page.wait_for_selector('[data-testid="teacher-result-status"]', timeout=15000)
        page.get_by_test_id("teacher-select-another-result").click()
        assert page.get_by_test_id("teacher-result").count() == 0
        assert page.get_by_test_id("teacher-preview").count() == 0
    finally:
        page.close()


def test_image_info_panel_and_ai_vs_manipulation(browser, vite_server, tmp_path):
    page, errors = _new_page(browser, vite_server)
    try:
        png = _synthetic_png(str(tmp_path / "info.png"), size=(64, 48))
        page.locator("input[type=file]").first.set_input_files(png)
        page.wait_for_selector('[data-testid="teacher-image-info"]', timeout=15000)
        info = page.get_by_test_id("teacher-image-info").inner_text()
        assert "64×48" in info
        assert "4:3" in info
        assert "image/png" in info
        assert "not an ai verdict" in info.lower()
        assert "EXIF" in info
        # Larger preview: alt names the file, image is visible.
        preview_img = page.locator('[data-testid="teacher-preview"] img')
        assert preview_img.get_attribute("alt") == "Preview of info.png"
        assert preview_img.is_visible()
        # Dedicated AI vs manipulation explanation, present without upload too.
        ai_vs = page.get_by_test_id("teacher-ai-vs-manipulation").inner_text()
        assert "AI-generation detection" in ai_vs
        assert "Manipulation detection" in ai_vs
        assert "not, by itself, evidence of AI generation" in ai_vs
        assert errors == [], f"page threw: {errors}"
    finally:
        page.close()
