"""Teacher Demo production-bundle smoke test — served dist, no inference.

Starts the REAL backend on loopback (port 8002) serving the tracked
frontend/dist bundle, drives Chromium to /teacher-demo, intercepts
GET /api/models with a demonstration registry response, and checks
that the served page actually renders the teacher demo, previews a
synthetic PIL image, and reports MODEL_NOT_CONFIGURED without calling
any inference endpoint. No pilot images, no REAL test images, no
detector inference, no training, no evaluation. The backend is
terminated by the fixture. This test only passes after a rebuild:
it proves the served bundle — not just vite dev — contains the
teacher demo.
"""

import json
import os

import pytest
from PIL import Image

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402
from portable_servers import start_backend_server, stop_process

PORT = 8002
BASE = f"http://127.0.0.1:{PORT}"
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
            "limitations": "DEMONSTRATION MODEL. Trained on SYNTHETIC procedural data only.",
            "supported_formats": ["image/jpeg", "image/png", "image/webp"],
            "preprocessing": {},
            "lifecycle_status": "DEMONSTRATION",
            "validation_status": "SYNTHETIC_ONLY",
        }
    ],
    "errors": {},
    "ml_available": True,
}


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


def test_served_bundle_teacher_demo(browser, prod_server, tmp_path):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    requested = []
    page.on("request", lambda req: requested.append(req.url))
    page.route("**/api/models", lambda r: r.fulfill(
        status=200, content_type="application/json", body=json.dumps(DEMO_REGISTRY)))
    try:
        page.goto(f"{prod_server}/teacher-demo", wait_until="networkidle")
        body = page.locator("body").inner_text()
        assert "TRUTH LENZ AI" in body
        assert "VALIDATION INCOMPLETE" in body
        png = tmp_path / "prod.png"
        Image.new("RGB", (32, 32), color=(40, 120, 200)).save(png, format="PNG")
        page.locator("input[type=file]").first.set_input_files(str(png))
        page.wait_for_selector('[data-testid="teacher-preview"]', timeout=15000)
        page.get_by_test_id("teacher-analyze-button").click()
        page.wait_for_selector('[data-testid="teacher-result-status"]', timeout=15000)
        assert page.get_by_test_id("teacher-result-status").inner_text() == "MODEL_NOT_CONFIGURED"
        assert errors == [], f"page threw: {errors}"
        for forbidden in ("/api/analyze", "/api/compare", "/api/batch", "/api/explain", "/api/robustness"):
            assert not any(forbidden in u for u in requested)
    finally:
        page.close()
