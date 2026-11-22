"""Stage 3 audit remediation tests (F1, F2, F3, F6).

Covers the fixes, not the original Stage 3 scope (see
``test_stage3_security.py`` for S3-1–S3-8):

- F1: blind-experiment run upload caps (transport + handler).
- F2: mutation-limiter thread safety.
- F3: compare validates both uploads before any analysis.
- F6: mutation-limiter wiring, JSON-size-limit wiring, exact-1 MB
  boundary, and live expensive-group coverage.

Safety: all tests run in-process (TestClient, loopback "testclient"
host) or against stub doubles. No live external calls, no pilot
images, no writes to protected data: the suite-wide isolation
fixture redirects history, counters, registries, and experiment
records to tmp. Oversized bodies are synthetic bytes, never real
image files. The expensive-group live test stubs the benchmark
runner and redirects the results write to tmp, so no benchmark
runs and no repository file is written.
"""

import asyncio
import io
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.main import app
import backend.api.routes as routes
from backend.services import pipeline as pipeline_mod


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean_buckets():
    routes._MUTATION_RATE.clear()
    routes._RATE.clear()
    yield
    routes._MUTATION_RATE.clear()
    routes._RATE.clear()


def _tiny_png_bytes():
    from PIL import Image

    img = Image.new("RGB", (8, 8), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _png_blob(size):
    return b"\x89PNG\r\n\x1a\n" + b"\0" * (size - 8)


class _StubUpload:
    """Minimal UploadFile stand-in for direct handler calls."""

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


def _fake_request(host="test-remediation"):
    return SimpleNamespace(client=SimpleNamespace(host=host))


# --- raw ASGI harness (same shape as test_stage3_security's) ---


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


# ------------------------------------------------------------------ F1


class TestExperimentRunLimits:
    def _stub_inference(self, called):
        def stub(experiment_id, images):
            called.append(images)
            return {"experiment_id": experiment_id, "status": "COMPLETED",
                    "n_predictions": len(images),
                    "predictions": [
                        {"image_id": im["image_id"], "verdict": "INCONCLUSIVE",
                         "ai_probability": None, "confidence": None,
                         "ood_status": None, "case_id": None}
                        for im in images]}
        return stub

    def test_valid_run_within_limits(self, client, monkeypatch):
        import backend.validation.blind_protocol as blind_mod

        called = []
        monkeypatch.setattr(blind_mod, "run_blind_inference",
                            self._stub_inference(called))
        files = [("files", (f"f{i}.png", _tiny_png_bytes(), "image/png"))
                 for i in range(2)]
        r = client.post("/api/experiments/EXP-TEST/run", files=files)
        assert r.status_code == 200
        assert r.json()["experiment_id"] == "EXP-TEST"
        assert len(called) == 1 and len(called[0]) == 2

    def test_too_many_files_rejected(self, client, monkeypatch):
        import backend.validation.blind_protocol as blind_mod

        called = []
        monkeypatch.setattr(blind_mod, "run_blind_inference",
                            self._stub_inference(called))
        files = [("files", (f"f{i}.png", _tiny_png_bytes(), "image/png"))
                 for i in range(21)]
        r = client.post("/api/experiments/EXP-TEST/run", files=files)
        assert r.status_code == 400
        assert "limited to 20" in r.json()["detail"]
        assert called == []

    def test_per_file_over_limit_rejected(self, client, monkeypatch):
        import backend.validation.blind_protocol as blind_mod

        called = []
        monkeypatch.setattr(blind_mod, "run_blind_inference",
                            self._stub_inference(called))
        files = [("files", ("big.png", _png_blob(routes.MAX_BYTES + 1), "image/png"))]
        r = client.post("/api/experiments/EXP-TEST/run", files=files)
        assert r.status_code == 413
        assert called == []

    def test_aggregate_over_limit_no_inference(self, client, monkeypatch):
        import backend.validation.blind_protocol as blind_mod

        called = []
        monkeypatch.setattr(blind_mod, "run_blind_inference",
                            self._stub_inference(called))
        # 5 x 20.1 MB decoded = 100.5 MB > 100 MB aggregate, each file
        # under the 25 MB per-file cap, total body under the 101 MB
        # transport cap: the aggregate gate must fire.
        files = [("files", (f"f{i}.png", _png_blob(int(20.1 * 1024 * 1024)),
                            "image/png")) for i in range(5)]
        r = client.post("/api/experiments/EXP-TEST/run", files=files)
        assert r.status_code == 413
        assert "Blind run total exceeds 100 MB" in r.json()["detail"]
        assert called == []

    def test_aggregate_exactly_at_limit_passes(self, client, monkeypatch):
        import backend.validation.blind_protocol as blind_mod

        called = []
        monkeypatch.setattr(blind_mod, "run_blind_inference",
                            self._stub_inference(called))
        # 5 x 20 MB decoded = exactly 100 MB: breach is `total > cap`.
        files = [("files", (f"f{i}.png", _png_blob(20 * 1024 * 1024),
                            "image/png")) for i in range(5)]
        r = client.post("/api/experiments/EXP-TEST/run", files=files)
        assert r.status_code == 200
        assert len(called) == 1

    def test_declared_oversize_rejected_without_consuming_body(self):
        import json

        status, body, consumed = asyncio.run(_call_app_raw(
            app, "POST", "/api/experiments/EXP-TEST/run",
            [("content-type", "multipart/form-data; boundary=----testboundary"),
             ("content-length", str(200 * 1024 * 1024))],
            [b"--fake--"],
        ))
        assert status == 413
        assert json.loads(body)["detail"] == "Request body too large."
        assert consumed == 0

    def test_chunked_over_transport_cap_aborted(self):
        import json

        from backend.api.body_limits import EXPERIMENT_RUN_TRANSPORT_CAP

        cap = EXPERIMENT_RUN_TRANSPORT_CAP
        # Valid multipart framing, memory-efficient: one small part
        # header, then repeats of a single shared 1 MB zero chunk
        # (the parser sees continuous file content; zeros never
        # contain the boundary), then the closing boundary. The
        # transport gate fires mid-content, before parsing completes.
        # Arbitrary garbage bytes would NOT test this: Starlette's
        # multipart parser rejects non-boundary input immediately
        # with 400, before the cap is ever reached.
        head = (f"------testboundary\r\nContent-Disposition: form-data; "
                f"name=\"files\"; filename=\"x.png\"\r\n"
                f"Content-Type: image/png\r\n\r\n").encode()
        tail = b"\r\n------testboundary--\r\n"
        chunk = b"\0" * (1024 * 1024)
        chunks = [head] + [chunk] * 102 + [tail]
        total = sum(len(c) for c in chunks)
        assert total > cap
        status, resp, consumed = asyncio.run(_call_app_raw(
            app, "POST", "/api/experiments/EXP-TEST/run",
            [("content-type", "multipart/form-data; boundary=----testboundary")],
            chunks,
        ))
        assert status == 413
        assert json.loads(resp)["detail"] == "Request body too large."
        assert consumed <= cap + 1024 * 1024
        assert consumed < total

    def test_cleanup_on_success_and_failure(self, monkeypatch):
        import backend.validation.blind_protocol as blind_mod
        from fastapi import HTTPException

        called = []
        monkeypatch.setattr(blind_mod, "run_blind_inference",
                            self._stub_inference(called))
        # Success: both stubs closed after the handler returns.
        stubs = [_StubUpload(_tiny_png_bytes(), f"f{i}.png") for i in range(2)]
        resp = asyncio.run(routes.run_blind_experiment_api(
            "EXP-TEST", _fake_request("test-cleanup-success"), stubs))
        assert resp.status_code == 200
        assert all(s.closed for s in stubs)
        assert len(called) == 1
        # Failure: aggregate breach on the first file (patched cap);
        # stubs never read by the loop are still closed by finally,
        # and inference never runs.
        called.clear()
        monkeypatch.setattr(routes, "BATCH_MAX_BYTES", 64)
        stubs = [_StubUpload(_tiny_png_bytes(), f"f{i}.png") for i in range(3)]
        with pytest.raises(HTTPException) as ei:
            asyncio.run(routes.run_blind_experiment_api(
                "EXP-TEST", _fake_request("test-cleanup-failure"), stubs))
        assert ei.value.status_code == 413
        assert all(s.closed for s in stubs)
        assert called == []


# ------------------------------------------------------------------ F3


class TestCompareReadBeforeAnalysis:
    def test_oversized_second_file_no_analysis_no_counter_change(self, client, monkeypatch):
        called = []
        monkeypatch.setattr(routes.pipeline, "run_analysis",
                            lambda *a, **k: called.append(True))
        counter_path = pipeline_mod._CASE_COUNTER_PATH
        before = (open(counter_path, "rb").read()
                  if os.path.exists(counter_path) else None)
        files = {"file_a": ("a.png", _tiny_png_bytes(), "image/png"),
                 "file_b": ("b.png", _png_blob(routes.MAX_BYTES + 1), "image/png")}
        r = client.post("/api/compare", files=files)
        assert r.status_code == 413
        assert called == []
        after = (open(counter_path, "rb").read()
                 if os.path.exists(counter_path) else None)
        assert before == after

    def test_both_uploads_closed_on_rejection(self):
        from fastapi import HTTPException

        called = []
        orig = routes.pipeline.run_analysis
        routes.pipeline.run_analysis = lambda *a, **k: called.append(True)
        try:
            stubs = [_StubUpload(_tiny_png_bytes(), "a.png"),
                     _StubUpload(_png_blob(routes.MAX_BYTES + 1), "b.png")]
            with pytest.raises(HTTPException) as ei:
                asyncio.run(routes.compare(_fake_request("test-compare-close"),
                                           stubs[0], stubs[1]))
            assert ei.value.status_code == 413
            assert called == []
            assert all(s.closed for s in stubs)
        finally:
            routes.pipeline.run_analysis = orig

    def test_valid_compare_still_works(self, client, monkeypatch):
        def stub(name, mime, data, include_visuals=False):
            return {
                "ensemble": {"verdict": "INCONCLUSIVE", "ai_probability": None,
                             "manipulation_probability": None,
                             "ml_ai_probability": None},
                "signals": [{"name": "metadata", "display_name": "Metadata",
                             "ai_score": 0, "manip_score": 0,
                             "technical": {}}],
            }

        monkeypatch.setattr(routes.pipeline, "run_analysis", stub)
        files = {"file_a": ("a.png", _tiny_png_bytes(), "image/png"),
                 "file_b": ("b.png", _tiny_png_bytes(), "image/png")}
        r = client.post("/api/compare", files=files)
        assert r.status_code == 200
        assert "comparison" in r.json()


# ------------------------------------------------------------------ F2


class TestMutationLimiterThreadSafety:
    def test_concurrent_requests_cannot_exceed_bucket(self):
        import threading

        from fastapi import HTTPException

        host = "203.0.113.99"
        routes._MUTATION_RATE.clear()
        n = 50
        barrier = threading.Barrier(n)
        results = []

        def attempt():
            req = SimpleNamespace(client=SimpleNamespace(host=host))
            barrier.wait(timeout=10)
            try:
                routes._check_mutation_rate(req, "destructive")
                results.append(True)
            except HTTPException as exc:
                assert exc.status_code == 429
                results.append(False)

        threads = [threading.Thread(target=attempt) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(results) == n
        # Exactly the bucket limit succeeds; the race that let a
        # synchronized burst overshoot is closed by the lock.
        assert sum(results) == 10
        routes._MUTATION_RATE.clear()

    def test_sequential_limit_and_window_recovery(self):
        import time

        from fastapi import HTTPException

        host = "203.0.113.98"
        routes._MUTATION_RATE.clear()
        req = SimpleNamespace(client=SimpleNamespace(host=host))
        for _ in range(10):
            routes._check_mutation_rate(req, "destructive")
        with pytest.raises(HTTPException) as ei:
            routes._check_mutation_rate(req, "destructive")
        assert ei.value.status_code == 429
        routes._MUTATION_RATE[(host, "destructive")] = [time.time() - 61] * 10
        routes._check_mutation_rate(req, "destructive")
        routes._MUTATION_RATE.clear()


# ------------------------------------------------------------------ F6


EXPECTED_MUTATION_ROUTES = {
    ("DELETE", "/api/history/{analysis_id}"): "destructive",
    ("DELETE", "/api/history"): "destructive",
    ("POST", "/api/models/reload"): "mutation",
    ("POST", "/api/datasets/register"): "mutation",
    ("POST", "/api/benchmark/run"): "expensive",
    ("POST", "/api/experiments/blind"): "mutation",
    ("POST", "/api/experiments/{experiment_id}/reveal"): "mutation",
    ("POST", "/api/experiments/{experiment_id}/invalidate"): "mutation",
    ("POST", "/api/models/{model_id}/promote"): "mutation",
}


class TestMutationLimiterWiring:
    def _all_api_routes(self):
        # Starlette groups included-router routes behind an
        # _IncludedRouter entry; the real APIRoute objects live on
        # its original_router. Flatten recursively so the structural
        # checks see every route regardless of grouping.
        out = []

        def walk(routes):
            for r in routes:
                inner = getattr(r, "original_router", None)
                if inner is not None:
                    walk(getattr(inner, "routes", []))
                else:
                    out.append(r)

        walk(app.routes)
        return out

    def test_every_mutation_route_calls_its_limiter(self):
        import inspect

        by_key = {}
        for route in self._all_api_routes():
            for method in (getattr(route, "methods", None) or set()):
                by_key[(method, getattr(route, "path", ""))] = route
        for (method, path), group in EXPECTED_MUTATION_ROUTES.items():
            route = by_key.get((method, path))
            assert route is not None, f"route missing: {method} {path}"
            source = inspect.getsource(route.endpoint)
            assert f'_check_mutation_rate(request, "{group}")' in source, \
                f"limiter wiring missing: {method} {path} (expected group {group})"

    def test_blind_run_uses_analysis_limiter_by_design(self):
        # Deliberate exception, pinned so a future edit cannot
        # silently move it: experiments/run is rate-limited with the
        # analysis bucket (30/min, same numeric limit as "mutation"),
        # not the mutation buckets.
        import inspect

        route = next(r for r in self._all_api_routes()
                     if getattr(r, "path", "") == "/api/experiments/{experiment_id}/run")
        source = inspect.getsource(route.endpoint)
        assert "_check_rate(request)" in source
        assert "_check_mutation_rate" not in source


class TestExpensiveGroupLive:
    def test_benchmark_run_expensive_limit_live(self, client, monkeypatch, tmp_path):
        # Live wiring proof without launching a benchmark: the
        # benchmark runner is stubbed and the results write is
        # redirected to tmp, so no benchmark executes and no
        # repository file is written. The limiter check runs first
        # in the handler, so the 6th call must be 429 before any
        # stubbed work happens.
        import backend.benchmark.runner as runner_mod

        monkeypatch.setattr(runner_mod, "run_benchmark", lambda *a, **k: {})
        monkeypatch.setattr(routes, "_safe_dataset_dir", lambda name: str(tmp_path))
        for _ in range(5):
            r = client.post("/api/benchmark/run?dataset=sample")
            assert r.status_code == 200
        r = client.post("/api/benchmark/run?dataset=sample")
        assert r.status_code == 429
        assert "Rate limit exceeded" in r.json()["detail"]


class TestJsonWiringLive:
    def _oversize_body(self):
        return b'{"name": "' + b"x" * (1024 * 1024) + b'"}'

    def test_register_json_limit_live(self, client, monkeypatch):
        import backend.datasets.registry as registry_mod

        called = []
        monkeypatch.setattr(registry_mod, "register_dataset",
                            lambda meta: called.append(meta) or "RWD-2099-999999")
        r = client.post("/api/datasets/register", content=self._oversize_body(),
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 413
        assert called == []
        r = client.post("/api/datasets/register", json={"name": "stage3-remediation"})
        assert r.status_code == 200
        assert called == [{"name": "stage3-remediation"}]

    def test_create_blind_json_limit_live(self, client, monkeypatch):
        import backend.validation.blind_protocol as blind_mod

        called = []
        rec = {"experiment_id": "EXP-1", "status": "CREATED",
               "reproducibility": {"dataset_hash": "h", "model_version": "v",
                                   "model_hash": "h", "configuration_hash": "h"}}

        def stub(*a, **k):
            called.append(True)
            return rec

        monkeypatch.setattr(blind_mod, "create_blind_experiment", stub)
        r = client.post("/api/experiments/blind", content=self._oversize_body(),
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 413
        assert called == []
        r = client.post("/api/experiments/blind",
                        json={"dataset_id": "d", "model_id": "m"})
        assert r.status_code == 200
        assert called == [True]

    def test_reveal_json_limit_live(self, client, monkeypatch):
        import backend.validation.blind_protocol as blind_mod

        called = []
        rec = {"experiment_id": "EXP-1", "status": "REVEALED", "revealed": True,
               "metrics": {}, "leakage_report": None, "results_hash": "h"}

        def stub(*a, **k):
            called.append(True)
            return rec

        monkeypatch.setattr(blind_mod, "reveal_and_score", stub)
        r = client.post("/api/experiments/EXP-1/reveal", content=self._oversize_body(),
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 413
        assert called == []
        r = client.post("/api/experiments/EXP-1/reveal",
                        json={"ground_truth": {"a.png": "REAL"}})
        assert r.status_code == 200
        assert called == [True]

    def test_invalidate_json_limit_live(self, client, monkeypatch):
        import backend.validation.blind_protocol as blind_mod

        called = []
        rec = {"experiment_id": "EXP-1", "status": "INVALIDATED"}

        def stub(*a, **k):
            called.append(True)
            return rec

        monkeypatch.setattr(blind_mod, "invalidate_experiment", stub)
        r = client.post("/api/experiments/EXP-1/invalidate", content=self._oversize_body(),
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 413
        assert called == []
        r = client.post("/api/experiments/EXP-1/invalidate",
                        json={"reason": "remediation test"})
        assert r.status_code == 200
        assert called == [True]

    def test_promote_json_limit_live(self, client, monkeypatch):
        import backend.models.promotion as promotion_mod

        called = []

        def stub(*a, **k):
            called.append(True)
            return {"model_id": "m", "status": "ok"}

        monkeypatch.setattr(promotion_mod, "promote", stub)
        r = client.post("/api/models/m/promote", content=self._oversize_body(),
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 413
        assert called == []
        r = client.post("/api/models/m/promote",
                        json={"target_status": "RESEARCH", "evidence": {}})
        assert r.status_code == 200
        assert called == [True]


class TestJsonHelperBoundaries:
    def test_exactly_1mb_valid_json_parses(self):
        import json

        from backend.api.routes import JSON_MAX_BYTES, _read_json_limited

        # Exactly JSON_MAX_BYTES bytes of valid JSON: 7-byte prefix
        # + filler + 2-byte suffix.
        filler = b"x" * (JSON_MAX_BYTES - 9)
        body = b'{"a": "' + filler + b'"}'
        assert len(body) == JSON_MAX_BYTES
        chunks = [body[i:i + 65536] for i in range(0, len(body), 65536)]
        out = asyncio.run(_read_json_limited(_StubJsonRequest(chunks)))
        assert out == {"a": filler.decode()}

    def test_over_1mb_rejected_before_parsing(self):
        from fastapi import HTTPException

        from backend.api.routes import JSON_MAX_BYTES, _read_json_limited

        # Valid JSON over the cap: the size gate must fire, not the parser.
        filler = b"x" * (JSON_MAX_BYTES - 9 + 1)
        body = b'{"a": "' + filler + b'"}'
        assert len(body) == JSON_MAX_BYTES + 1
        chunks = [body[i:i + 65536] for i in range(0, len(body), 65536)]
        with pytest.raises(HTTPException) as ei:
            asyncio.run(_read_json_limited(_StubJsonRequest(chunks)))
        assert ei.value.status_code == 413
