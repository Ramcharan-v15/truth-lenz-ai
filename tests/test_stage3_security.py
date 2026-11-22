"""Stage 3 security regression tests (S3-1 through S3-8 scope).

Covers, in sections: S3-1 API 404 contract, S3-2 path disclosure and
dataset containment, S3-3 upload/JSON body limits, S3-4 privileged-route
rate limits, S3-5 CORS header allowlist, S3-6 Sec-Fetch-Site defense in
depth, S3-7 production bundle freshness.

Safety: all tests run against the app in-process (TestClient, loopback
"testclient" host) or against tmp directories. No live external calls,
no pilot images, no writes to protected data (history DB, counters,
registries, frozen manifests/results). Upload tests use synthetic
images generated in tmp; oversized bodies are generated as bytes, never
as real image files.
"""

import io
import os

import pytest
from fastapi.testclient import TestClient

from backend.main import app

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(REPO, "frontend", "dist")


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _current_js_asset():
    assets = os.path.join(DIST, "assets")
    js = sorted(f for f in os.listdir(assets) if f.endswith(".js"))
    assert len(js) == 1, f"expected exactly one JS bundle, found {js}"
    return js[0]


# ------------------------------------------------------------------ S3-1


class TestApi404Contract:
    def test_unknown_api_path_returns_json_404(self, client):
        r = client.get("/api/does-not-exist")
        assert r.status_code == 404
        assert r.headers["content-type"].startswith("application/json")
        assert r.json()["detail"] == "API route not found."

    def test_unknown_nested_api_path_returns_json_404(self, client):
        r = client.get("/api/v2/nested/path")
        assert r.status_code == 404
        assert r.json()["detail"] == "API route not found."

    def test_bare_api_path_returns_json_404(self, client):
        r = client.get("/api")
        assert r.status_code == 404
        assert r.json()["detail"] == "API route not found."

    def test_spa_paths_still_serve_html(self, client):
        for path in ("/", "/analyze", "/research", "/some-frontend-route"):
            r = client.get(path)
            assert r.status_code == 200, path
            assert "text/html" in r.headers["content-type"], path

    def test_asset_route_still_serves(self, client):
        r = client.get(f"/assets/{_current_js_asset()}")
        assert r.status_code == 200
        assert len(r.content) > 0

    def test_legitimate_api_routes_unaffected(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        r = client.get("/api/config")
        assert r.status_code == 200


# ------------------------------------------------------------------ S3-2


class TestPathDisclosureAndContainment:
    def test_models_status_omits_models_dir(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        body = r.json()
        assert "models_dir" not in body
        assert "total" in body and "models" in body and "ml_available" in body

    def test_models_status_contains_no_absolute_path(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200

        def walk(v):
            if isinstance(v, str):
                assert not v.startswith("/"), f"absolute path leaked: {v}"
            elif isinstance(v, dict):
                for x in v.values():
                    walk(x)
            elif isinstance(v, list):
                for x in v:
                    walk(x)

        walk(r.json())

    def test_models_reload_omits_models_dir(self, client):
        r = client.post("/api/models/reload")
        assert r.status_code == 200
        assert "models_dir" not in r.json()

    def test_within_directory_valid(self, tmp_path):
        from backend.api.routes import _is_within_directory

        base = tmp_path / "datasets"
        sub = base / "sample"
        sub.mkdir(parents=True)
        assert _is_within_directory(str(base), str(sub)) is True

    def test_within_directory_traversal(self, tmp_path):
        from backend.api.routes import _is_within_directory

        base = tmp_path / "datasets"
        base.mkdir()
        assert _is_within_directory(str(base), str(tmp_path / "other")) is False
        assert _is_within_directory(str(base), str(base / ".." / "other")) is False

    def test_within_directory_prefix_collision(self, tmp_path):
        from backend.api.routes import _is_within_directory

        base = tmp_path / "datasets"
        evil = tmp_path / "datasets-evil"
        base.mkdir()
        evil.mkdir()
        assert _is_within_directory(str(base), str(evil)) is False

    def test_within_directory_symlink_escape(self, tmp_path):
        from backend.api.routes import _is_within_directory

        base = tmp_path / "datasets"
        outside = tmp_path / "outside"
        base.mkdir()
        outside.mkdir()
        link = base / "escape"
        link.symlink_to(outside, target_is_directory=True)
        assert _is_within_directory(str(base), str(link)) is False
        # A symlink pointing inside stays inside.
        inner = base / "inner"
        inner.mkdir()
        good = base / "good"
        good.symlink_to(inner, target_is_directory=True)
        assert _is_within_directory(str(base), str(good)) is True

    def test_safe_dataset_dir_valid_and_rejected(self):
        from fastapi import HTTPException

        from backend.api.routes import _safe_dataset_dir

        # benchmark/datasets/sample ships with the repo.
        path = _safe_dataset_dir("sample")
        assert path.endswith(os.path.join("benchmark", "datasets", "sample"))
        for bad in ("..", "../..", "does-not-exist", "sample/../../.."):
            with pytest.raises(HTTPException):
                _safe_dataset_dir(bad)


# ------------------------------------------------------------------ S3-3


class _StubUpload:
    """Minimal UploadFile stand-in for _read_upload unit tests."""

    def __init__(self, data, filename="x.png", content_type="image/png"):
        self.filename = filename
        self.content_type = content_type
        self._data = data
        self.closed = False

    async def read(self):
        return self._data

    async def close(self):
        self.closed = True


class _StubJsonRequest:
    """Minimal request stand-in for _read_json_limited unit tests."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def stream(self):
        for c in self._chunks:
            yield c


class TestUploadLimits:
    def test_read_upload_exact_limit_accepted(self):
        import asyncio

        from backend.api.routes import MAX_BYTES, _read_upload

        data = b"\x89PNG\r\n\x1a\n" + b"\0" * (MAX_BYTES - 8)
        f = _StubUpload(data)
        name, mime, out = asyncio.run(_read_upload(f))
        assert mime == "image/png"
        assert len(out) == MAX_BYTES
        assert f.closed is True

    def test_read_upload_over_limit_rejected_and_closed(self):
        import asyncio

        from fastapi import HTTPException

        from backend.api.routes import MAX_BYTES, _read_upload

        f = _StubUpload(b"\x89PNG\r\n\x1a\n" + b"\0" * MAX_BYTES)
        with pytest.raises(HTTPException) as ei:
            asyncio.run(_read_upload(f))
        assert ei.value.status_code == 413
        assert f.closed is True

    def test_read_upload_empty_rejected_and_closed(self):
        import asyncio

        from fastapi import HTTPException

        from backend.api.routes import _read_upload

        f = _StubUpload(b"")
        with pytest.raises(HTTPException) as ei:
            asyncio.run(_read_upload(f))
        assert ei.value.status_code == 400
        assert f.closed is True


class TestJsonBodyLimit:
    def test_small_valid_json_parsed(self):
        import asyncio
        import json

        from backend.api.routes import _read_json_limited

        body = json.dumps({"a": 1}).encode()
        out = asyncio.run(_read_json_limited(_StubJsonRequest([body])))
        assert out == {"a": 1}

    def test_oversize_json_rejected(self):
        import asyncio

        from fastapi import HTTPException

        from backend.api.routes import JSON_MAX_BYTES, _read_json_limited

        big = b"x" * (JSON_MAX_BYTES + 1)
        chunks = [big[i:i + 65536] for i in range(0, len(big), 65536)]
        with pytest.raises(HTTPException) as ei:
            asyncio.run(_read_json_limited(_StubJsonRequest(chunks)))
        assert ei.value.status_code == 413

    def test_malformed_and_empty_json_rejected(self):
        import asyncio

        from fastapi import HTTPException

        from backend.api.routes import _read_json_limited

        for chunks in ([b"{not json"], []):
            with pytest.raises(HTTPException) as ei:
                asyncio.run(_read_json_limited(_StubJsonRequest(chunks)))
            assert ei.value.status_code == 400


# --- raw ASGI harness for transport-cap tests (counts consumed bytes) ---


async def _call_app_raw(app, method, path, headers, body_chunks,
                      client=("127.0.0.1", 12345)):
    consumed = 0
    chunks = list(body_chunks)
    idx = 0

    async def receive():
        nonlocal idx, consumed
        if idx < len(chunks):
            chunk = chunks[idx]
            idx += 1
            consumed += len(chunk)
            return {"type": "http.request", "body": chunk,
                    "more_body": idx < len(chunks)}
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = []

    async def send(msg):
        sent.append(msg)

    scope = {
        "type": "http", "http_version": "1.1", "method": method,
        "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
        "client": client, "server": ("127.0.0.1", 8000),
    }
    await app(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent
                    if m["type"] == "http.response.body")
    return status, body, consumed


def _multipart_body(file_bytes, filename="x.png", boundary="----testboundary"):
    head = (f"--{boundary}\r\nContent-Disposition: form-data; "
            f"name=\"file\"; filename=\"{filename}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + file_bytes + tail


def _tiny_png_bytes():
    import io

    from PIL import Image

    img = Image.new("RGB", (8, 8), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestTransportCaps:
    def test_declared_oversize_rejected_without_consuming_body(self):
        import asyncio
        import json

        status, body, consumed = asyncio.run(_call_app_raw(
            app, "POST", "/api/analyze",
            [("content-type", "multipart/form-data; boundary=----testboundary"),
             ("content-length", str(30 * 1024 * 1024))],
            [b"--fake--"],
        ))
        assert status == 413
        assert json.loads(body)["detail"] == "Request body too large."
        assert consumed == 0

    def test_chunked_over_cap_aborted_mid_stream(self):
        import asyncio
        import json

        from backend.api.body_limits import BODY_LIMITS

        cap = BODY_LIMITS["/api/analyze"]
        file_bytes = b"\x89PNG\r\n\x1a\n" + b"\0" * (30 * 1024 * 1024)
        body = _multipart_body(file_bytes)
        chunks = [body[i:i + 1024 * 1024] for i in range(0, len(body), 1024 * 1024)]
        assert len(body) > cap
        status, resp, consumed = asyncio.run(_call_app_raw(
            app, "POST", "/api/analyze",
            [("content-type", "multipart/form-data; boundary=----testboundary")],
            chunks,
        ))
        assert status == 413
        assert json.loads(resp)["detail"] == "Request body too large."
        # Aborted mid-stream: strictly less than the full body, and no
        # more than one chunk past the cap.
        assert consumed < len(body)
        assert consumed <= cap + 1024 * 1024

    def test_small_upload_passes_transport(self):
        import asyncio

        status, _, consumed = asyncio.run(_call_app_raw(
            app, "POST", "/api/analyze",
            [("content-type", "multipart/form-data; boundary=----testboundary")],
            [_multipart_body(_tiny_png_bytes())],
        ))
        assert status == 200
        assert consumed > 0


class TestBatchAggregate:
    def _png_blob(self, size):
        return b"\x89PNG\r\n\x1a\n" + b"\0" * (size - 8)

    def test_batch_over_aggregate_rejected_before_analysis(self, client, monkeypatch):
        # 5 x 20.1 MB decoded = 100.5 MB > 100 MB aggregate, each file
        # under the 25 MB per-file cap, total body under the 101 MB
        # transport cap: the aggregate gate must fire, not the transport
        # gate and not the per-file gate.
        import backend.api.routes as routes

        called = []
        monkeypatch.setattr(routes.pipeline, "run_analysis",
                            lambda *a, **k: called.append(True))
        files = [("files", (f"f{i}.png", self._png_blob(int(20.1 * 1024 * 1024)),
                            "image/png")) for i in range(5)]
        r = client.post("/api/batch", files=files)
        assert r.status_code == 413
        assert "Batch total exceeds 100 MB" in r.json()["detail"]
        assert called == []

    def test_batch_exactly_at_aggregate_passes(self, client, monkeypatch):
        import backend.api.routes as routes

        def stub(name, mime, data, include_visuals=False):
            return {
                "analysis_id": "test", "width": 8, "height": 8,
                "ensemble": {"verdict": "INCONCLUSIVE", "ai_probability": None,
                             "manipulation_probability": None, "confidence": None,
                             "confidence_label": "Unknown"},
                "signals": [{"name": "quality",
                             "technical": {"quality_tier": "Unknown"}}],
            }

        monkeypatch.setattr(routes.pipeline, "run_analysis", stub)
        # 5 x 20 MB decoded = exactly 100 MB: breach is `total > cap`,
        # so exactly-at-cap passes.
        files = [("files", (f"f{i}.png", self._png_blob(20 * 1024 * 1024),
                            "image/png")) for i in range(5)]
        r = client.post("/api/batch", files=files)
        assert r.status_code == 200
        assert r.json()["count"] == 5

    def test_batch_small_files_still_work(self, client):
        files = [("files", (f"f{i}.png", _tiny_png_bytes(), "image/png"))
                 for i in range(2)]
        r = client.post("/api/batch", files=files)
        assert r.status_code == 200
        assert r.json()["count"] == 2


class TestJsonRouteWiring:
    def test_register_oversize_json_rejected_before_handler(self, client, monkeypatch):
        import backend.datasets.registry as registry_mod

        called = []
        monkeypatch.setattr(
            registry_mod, "register_dataset",
            lambda meta: called.append(meta) or "RWD-2099-999999")
        big = b'{"name": "' + b"x" * (1024 * 1024) + b'"}'
        r = client.post("/api/datasets/register", content=big,
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 413
        assert called == []

    def test_register_small_json_reaches_handler(self, client, monkeypatch):
        import backend.datasets.registry as registry_mod

        monkeypatch.setattr(
            registry_mod, "register_dataset",
            lambda meta: "RWD-2099-999999")
        r = client.post("/api/datasets/register",
                        json={"name": "stage3-test-dataset"})
        assert r.status_code == 200
        assert r.json()["dataset_id"] == "RWD-2099-999999"


# ------------------------------------------------------------------ S3-4


class TestPrivilegedRateLimits:
    @pytest.fixture(autouse=True)
    def _clean_buckets(self):
        import backend.api.routes as routes

        routes._MUTATION_RATE.clear()
        routes._RATE.clear()
        yield
        routes._MUTATION_RATE.clear()
        routes._RATE.clear()

    def test_destructive_delete_limit_and_429(self, client):
        # Deleting a nonexistent id touches nothing (404) but still
        # consumes the destructive budget.
        for _ in range(10):
            r = client.delete("/api/history/does-not-exist")
            assert r.status_code == 404
        r = client.delete("/api/history/does-not-exist")
        assert r.status_code == 429
        assert "Rate limit exceeded" in r.json()["detail"]

    def test_destructive_window_recovery(self, client):
        import time

        import backend.api.routes as routes

        old = time.time() - 61
        routes._MUTATION_RATE[("testclient", "destructive")] = [old] * 10
        r = client.delete("/api/history/does-not-exist")
        assert r.status_code == 404  # aged entries expired, not 429

    def test_mutation_reload_limit_and_429(self, client):
        for _ in range(30):
            r = client.post("/api/models/reload")
            assert r.status_code == 200
        r = client.post("/api/models/reload")
        assert r.status_code == 429

    def test_expensive_group_limit_direct(self):
        import time
        from types import SimpleNamespace

        import backend.api.routes as routes
        from fastapi import HTTPException

        req = SimpleNamespace(client=SimpleNamespace(host="203.0.113.7"))
        for _ in range(5):
            routes._check_mutation_rate(req, "expensive")
        with pytest.raises(HTTPException) as ei:
            routes._check_mutation_rate(req, "expensive")
        assert ei.value.status_code == 429
        # Aged window recovers without waiting.
        routes._MUTATION_RATE[("203.0.113.7", "expensive")] = [time.time() - 61] * 5
        routes._check_mutation_rate(req, "expensive")

    def test_analysis_limit_preserved(self):
        from types import SimpleNamespace

        import backend.api.routes as routes
        from fastapi import HTTPException

        req = SimpleNamespace(client=SimpleNamespace(host="203.0.113.8"))
        for _ in range(30):
            routes._check_rate(req)
        with pytest.raises(HTTPException) as ei:
            routes._check_rate(req)
        assert ei.value.status_code == 429


# ------------------------------------------------------------------ S3-5


class TestCorsHeaderAllowlist:
    def _preflight(self, client, origin, headers):
        return client.options(
            "/api/analyze",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": headers,
            },
        )

    def test_content_type_preflight_granted(self, client):
        r = self._preflight(client, "http://localhost:8000", "Content-Type")
        assert r.headers.get("access-control-allow-origin") == "http://localhost:8000"
        assert "content-type" in r.headers.get("access-control-allow-headers", "").lower()

    def test_custom_header_preflight_not_granted(self, client):
        # Starlette answers a disallowed-header preflight with 400
        # "Disallowed CORS headers", which browsers treat as a failed
        # preflight (the actual request is never sent) — even though
        # the failure response still echoes an allow-origin header.
        # The grant decision is the status + body, not that header.
        r = self._preflight(client, "http://localhost:8000", "X-Custom-Thing")
        assert r.status_code == 400
        assert r.text == "Disallowed CORS headers"

    def test_evil_origin_preflight_not_granted(self, client):
        r = self._preflight(client, "http://evil.example", "Content-Type")
        assert "access-control-allow-origin" not in r.headers

    def test_no_credentials_allowed(self, client):
        r = self._preflight(client, "http://localhost:8000", "Content-Type")
        assert r.headers.get("access-control-allow-credentials") != "true"


# ------------------------------------------------------------------ S3-6


class TestFetchMetadataDefense:
    @pytest.fixture(autouse=True)
    def _clean_mutation_buckets(self):
        import backend.api.routes as routes

        routes._MUTATION_RATE.clear()
        yield
        routes._MUTATION_RATE.clear()

    def test_missing_everything_still_passes(self, client):
        r = client.post("/api/models/reload")
        assert r.status_code == 200

    def test_cross_site_without_origin_rejected(self, client):
        r = client.post("/api/models/reload",
                        headers={"Sec-Fetch-Site": "cross-site"})
        assert r.status_code == 403
        assert "Cross-site" in r.json()["detail"]

    def test_cross_site_value_case_insensitive(self, client):
        r = client.post("/api/models/reload",
                        headers={"Sec-Fetch-Site": "Cross-Site"})
        assert r.status_code == 403

    def test_same_origin_and_same_site_pass(self, client):
        for value in ("same-origin", "same-site", "none"):
            r = client.post("/api/models/reload",
                            headers={"Sec-Fetch-Site": value})
            assert r.status_code == 200, value

    def test_trusted_origin_not_second_guessed(self, client):
        # Origin present: the Origin decision stands on its own.
        r = client.post("/api/models/reload",
                        headers={"Origin": "http://localhost:8000",
                                 "Sec-Fetch-Site": "cross-site"})
        assert r.status_code == 200

    def test_cross_site_multipart_analyze_rejected(self, client):
        r = client.post("/api/analyze", files={"file": ("x.png", _tiny_png_bytes(), "image/png")},
                        headers={"Sec-Fetch-Site": "cross-site"})
        assert r.status_code == 403


# ------------------------------------------------------------------ S3-7


class TestBundleFreshness:
    """The tracked bundle must contain the Stage 1/2 source fixes.

    If frontend source changes without a rebuild, this test fails
    closed: it reads the tracked frontend/dist bundle (the one the
    backend actually serves) and requires the markers that only the
    fixed source produces. The old (pre-Stage-1/2) bundle contained
    none of these markers. This is a process gate, not a substitute
    for rebuilding: when it fails, run the documented build in
    docs/DEPLOYMENT.md and re-run.
    """

    MARKERS = (
        # Stage 2 withheld-ML panels (dashboard + Research).
        "ML probability withheld",
        # Stage 2 EvidenceGraph withheld branch/leaf.
        "output withheld",
        # V5.1.8.64 Compare null-delta handling.
        "comparison_note",
        # Stage 2 + historical-fix suppression-reason rendering.
        "ml_suppressed_reason",
        # Stage 2 suppression-flag rendering. Replaces the former
        # "unavailable" marker (Stage 3 remediation, F4): "unavailable"
        # occurs 5 times in the historical bundle, so it cannot
        # distinguish stale from fresh. "suppressed" occurs 0 times
        # in the historical bundle and 12 times in the current one,
        # so it is a genuine freshness signal for the Stage 2
        # suppression work. Each marker above was verified
        # individually: 0 occurrences in the historical bundle
        # (HEAD frontend/dist JS) and >0 in the current bundle.
    )

    def _bundle_js(self):
        assets = os.path.join(DIST, "assets")
        js_files = sorted(f for f in os.listdir(assets) if f.endswith(".js"))
        assert len(js_files) == 1, f"expected exactly one JS bundle, found {js_files}"
        with open(os.path.join(assets, js_files[0]), encoding="utf-8", errors="replace") as fh:
            return fh.read()

    def test_bundle_contains_stage1_stage2_markers(self):
        js = self._bundle_js()
        for marker in self.MARKERS:
            assert marker in js, f"stale bundle: marker missing: {marker}"

    def test_index_html_references_current_bundle(self):
        with open(os.path.join(DIST, "index.html"), encoding="utf-8") as fh:
            html = fh.read()
        assets = os.path.join(DIST, "assets")
        js_files = sorted(f for f in os.listdir(assets) if f.endswith(".js"))
        assert len(js_files) == 1
        assert js_files[0] in html


# ------------------------------------------------------------------ F5 (remediation)


class TestLoopbackPrecedenceOverBodyLimits:
    def test_non_loopback_oversized_gets_403_not_413(self):
        # Discriminating case for middleware order: a non-loopback
        # peer sending an oversized declared body must be rejected by
        # the loopback gate (403) before body-limit processing (413),
        # and the body must not be consumed.
        import asyncio
        import json

        status, body, consumed = asyncio.run(_call_app_raw(
            app, "POST", "/api/analyze",
            [("content-type", "multipart/form-data; boundary=----testboundary"),
             ("content-length", str(30 * 1024 * 1024))],
            [b"--fake--"],
            client=("203.0.113.7", 1),
        ))
        assert status == 403
        assert "local machine" in json.loads(body)["detail"]
        assert consumed == 0
