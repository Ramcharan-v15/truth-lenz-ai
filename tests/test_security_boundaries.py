"""Security-boundary regression tests (V5.1.8.59).

Covers: F1 FastAPI SPA traversal, F2 serve.py static traversal,
F3 loopback guard on privileged mutation routes, F4 CORS origins,
F7 startup bind check.

Safety rules for this file (do not weaken them):
- No test writes to the real datasets registry, history DB, frozen
  artifacts, or experiment records. Non-loopback calls are rejected
  by the guard before any handler runs, so they are safe for every
  route. The only loopback live call is DELETE of a nonexistent
  history id, which returns 404 without writing.
- Bodies are compared by length only where file content is involved;
  no file content is asserted or printed.
- The ASGI harness below builds raw scopes so plain ``..`` segments
  reach the handler exactly as a server would pass them (HTTP clients
  may normalize them away and hide the bug).
"""

import asyncio
import importlib.util
import os
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(REPO, "frontend", "dist")
INDEX_LEN = os.path.getsize(os.path.join(DIST, "index.html"))
README_LEN = os.path.getsize(os.path.join(REPO, "README.md"))

# ---------------------------------------------------------------- ASGI harness


def call_app(app, path, method="GET", client=("127.0.0.1", 12345), body=b"", query=b"", headers=None):
    """Call the ASGI app with an explicit client; return (status, length).

    ``headers`` (list of raw (name, value) byte pairs) overrides the
    default content-type handling when given, so Origin/Referer and
    multipart bodies can be set exactly.
    """
    status_holder: dict = {}
    length = 0
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query,
        "headers": headers if headers is not None else ([(b"content-type", b"application/json")] if body else []),
        "client": client,
        "server": ("127.0.0.1", 8000),
    }
    sent_body = False

    async def receive():
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal length
        if message["type"] == "http.response.start":
            status_holder["status"] = message["status"]
        elif message["type"] == "http.response.body":
            length += len(message.get("body", b""))

    asyncio.run(app(scope, receive, send))
    return status_holder.get("status"), length


@pytest.fixture(scope="module")
def app():
    from backend.main import app as _app

    return _app


# ---------------------------------------------------------------- F1: SPA fallback


class TestSpaContainment:
    @pytest.mark.parametrize(
        "path",
        [
            "/../../README.md",
            "/../../../README.md",
            "/../../../../etc/passwd",
            "/..%2f..%2fREADME.md",
        ],
    )
    def test_traversal_never_serves_outside_file(self, app, path):
        status, length = call_app(app, path)
        assert status == 200  # SPA fallback, not an error page
        assert length == INDEX_LEN
        assert length != README_LEN

    def test_assets_mount_traversal_never_serves_outside_file(self, app):
        # /assets/* is served by StaticFiles (its own containment), not
        # the SPA fallback, so the expectation is only: never README.
        status, length = call_app(app, "/assets/../../README.md")
        assert length != README_LEN

    def test_legitimate_asset_and_spa_routes_still_work(self, app):
        # The JS bundle name carries a content hash that changes on
        # every rebuild; resolve the current asset from dist instead of
        # pinning a stale hash, so this test keeps passing after
        # authorized rebuilds. Assertion strength is unchanged: a real
        # asset must still serve 200 with a non-empty body.
        assets_dir = os.path.join(DIST, "assets")
        js_assets = sorted(f for f in os.listdir(assets_dir) if f.endswith(".js"))
        assert len(js_assets) == 1, f"expected exactly one JS bundle, found {js_assets}"
        status, length = call_app(app, f"/assets/{js_assets[0]}")
        assert status == 200
        assert length > 0
        status, length = call_app(app, "/index.html")
        assert status == 200 and length == INDEX_LEN
        status, length = call_app(app, "/analyze")
        assert status == 200 and length == INDEX_LEN


# ---------------------------------------------------------------- F2: serve.py


def _load_serve():
    spec = importlib.util.spec_from_file_location("serve_under_test", os.path.join(REPO, "serve.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestServeResolveStatic:
    def test_legitimate_file_resolves(self, tmp_path):
        serve = _load_serve()
        www = tmp_path / "www"
        www.mkdir()
        (www / "index.html").write_text("hi")
        assert serve._resolve_static(str(www), "/index.html") is not None

    @pytest.mark.parametrize(
        "path", ["../secret.txt", "../../secret.txt", "/../../secret.txt", "..\\secret.txt"]
    )
    def test_traversal_resolves_to_none(self, tmp_path, path):
        serve = _load_serve()
        www = tmp_path / "www"
        www.mkdir()
        (tmp_path / "secret.txt").write_text("secret")
        assert serve._resolve_static(str(www), path) is None

    def test_symlink_escape_resolves_to_none(self, tmp_path):
        serve = _load_serve()
        www = tmp_path / "www"
        www.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("outside")
        os.symlink(str(outside), str(www / "link.txt"))
        assert serve._resolve_static(str(www), "/link.txt") is None


# ---------------------------------------------------------------- F3: loopback guard


def _req(host):
    client = SimpleNamespace(host=host, port=12345) if host is not None else None
    return SimpleNamespace(client=client)


class TestLoopbackGuard:
    @pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "127.1.2.3", "::1", "localhost"])
    def test_loopback_allowed(self, host):
        from backend.api.local_guard import is_loopback_client

        assert is_loopback_client(_req(host)) is True

    @pytest.mark.parametrize(
        "host", ["192.168.1.5", "10.0.0.9", "172.16.0.1", "8.8.8.8", "example.com", None, ""]
    )
    def test_non_loopback_denied(self, host):
        from backend.api.local_guard import is_loopback_client, require_loopback
        from fastapi import HTTPException

        assert is_loopback_client(_req(host)) is False
        with pytest.raises(HTTPException) as exc:
            require_loopback(_req(host))
        assert exc.value.status_code == 403

    def test_forwarded_headers_are_ignored(self):
        """X-Forwarded-For must never establish loopback (no header use)."""
        from backend.api.local_guard import is_loopback_client

        req = SimpleNamespace(
            client=SimpleNamespace(host="8.8.8.8", port=1),
            headers={"x-forwarded-for": "127.0.0.1"},
        )
        assert is_loopback_client(req) is False

    def test_testclient_host_is_in_process_only(self):
        """'testclient' is set by Starlette's in-process transport, never
        by a network socket peer, so allowing it cannot open a network
        hole; it only lets the project's own suite exercise guarded
        routes. Missing clients are still denied."""
        from backend.api.local_guard import is_loopback_client

        assert is_loopback_client(_req("testclient")) is True
        assert is_loopback_client(SimpleNamespace(client=None)) is False


PRIVILEGED_ROUTES = [
    ("DELETE", "/api/history", b""),
    ("DELETE", "/api/history/does-not-exist", b""),
    ("POST", "/api/models/reload", b""),
    ("POST", "/api/datasets/register", b""),
    ("POST", "/api/benchmark/run", b"dataset=nope"),
    ("POST", "/api/experiments/blind", b""),
    ("POST", "/api/experiments/nope/run", b""),
    ("POST", "/api/experiments/nope/reveal", b""),
    ("POST", "/api/experiments/nope/invalidate", b""),
    ("POST", "/api/models/nope/promote", b""),
]


class TestPrivilegedRouteCoverage:
    @staticmethod
    def _all_routes(routes):
        # This FastAPI version keeps included-router routes inside an
        # _IncludedRouter wrapper (original_router.routes).
        for route in routes:
            orig = getattr(route, "original_router", None)
            if orig is not None:
                yield from TestPrivilegedRouteCoverage._all_routes(orig.routes)
            else:
                yield route

    def test_every_privileged_route_carries_the_guard(self, app):
        from backend.api.local_guard import require_loopback

        # Route templates (live 403 tests below prove enforcement with
        # concrete ids; this test proves the wiring on the templates).
        wanted = {
            ("DELETE", "/api/history"),
            ("DELETE", "/api/history/{analysis_id}"),
            ("POST", "/api/models/reload"),
            ("POST", "/api/datasets/register"),
            ("POST", "/api/benchmark/run"),
            ("POST", "/api/experiments/blind"),
            ("POST", "/api/experiments/{experiment_id}/run"),
            ("POST", "/api/experiments/{experiment_id}/reveal"),
            ("POST", "/api/experiments/{experiment_id}/invalidate"),
            ("POST", "/api/models/{model_id}/promote"),
        }
        found = set()
        for route in self._all_routes(app.routes):
            methods = getattr(route, "methods", None) or set()
            path = getattr(route, "path", "")
            deps = getattr(route, "dependencies", []) or []
            if any(getattr(d, "dependency", None) is require_loopback for d in deps):
                for m in methods:
                    found.add((m, path))
        missing = wanted - found
        assert not missing, f"guard missing on: {sorted(missing)}"

    @pytest.mark.parametrize("method,path,query", PRIVILEGED_ROUTES)
    def test_non_loopback_gets_403_before_handler(self, app, method, path, query):
        # Guard runs before the handler, so no route can write anything.
        status, _ = call_app(app, path, method=method, client=("203.0.113.7", 1), body=b"{}", query=query)
        assert status == 403

    def test_loopback_delete_of_missing_id_is_404_not_403(self, app):
        # The only loopback live call: a delete that cannot write anything.
        status, _ = call_app(app, "/api/history/does-not-exist", method="DELETE")
        assert status == 404


# ---------------------------------------------------------------- F4: CORS


class TestCorsOrigins:
    def test_disallowed_origin_gets_no_grant(self):
        from fastapi.testclient import TestClient
        from backend.main import app

        client = TestClient(app, raise_server_exceptions=False)
        r = client.options(
            "/api/health",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert r.headers.get("access-control-allow-origin") in (None, "")

    def test_loopback_origin_is_granted(self):
        from fastapi.testclient import TestClient
        from backend.main import app

        client = TestClient(app, raise_server_exceptions=False)
        r = client.options(
            "/api/health",
            headers={
                "Origin": "http://127.0.0.1:8000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert r.headers.get("access-control-allow-origin") == "http://127.0.0.1:8000"

    def test_wildcard_origin_is_gone(self):
        from backend.main import app

        for middleware in app.user_middleware:
            if "CORSMiddleware" in str(middleware.cls):
                assert "*" not in middleware.kwargs.get("allow_origins", [])


# ---------------------------------------------------------------- F7: startup bind check


class TestBindHostCheck:
    @pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "localhost"])
    def test_loopback_binds_pass(self, host):
        from backend.api.local_guard import validate_bind_host

        validate_bind_host(host)  # must not raise

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.com", ""])
    def test_non_loopback_binds_fail_closed(self, host):
        from backend.api.local_guard import validate_bind_host

        with pytest.raises(RuntimeError):
            validate_bind_host(host)


# ---------------------------------------------------------------- F8: browser-origin protection (V5.1.8.60)
#
# What this proves -- and what it does NOT prove:
# - It proves a browser page from an untrusted origin cannot mutate
#   state through the browser (Origin is always sent by browsers on
#   POST/DELETE, including simple form POSTs).
# - It does NOT prove anything about non-browser callers: curl, scripts,
#   and tests can forge any Origin header. Missing Origin is therefore
#   a non-browser compatibility policy, not authentication and not
#   proof of safety. Loopback guard (F3) and the deployment boundary
#   (F9) are what constrain non-browser callers.
# - Multipart analyze calls use the raw ASGI harness with a distinct
#   loopback peer (127.0.0.2) so they never share TestClient's
#   per-IP rate-limit bucket with the rest of the suite.
# - Writes land in tmp dirs via tests/conftest.py (F10 isolation).


def _png_bytes():
    """A non-degenerate 256x256 noise PNG.

    A solid-color or tiny image is a degenerate input for the forensic
    pipeline (flat-region statistics divide by zero and produce NaN,
    which JSON serialization then rejects with a 500). Real photos
    never look like that, so the test image carries random pixel
    variance instead. Seeded for determinism.
    """
    import io
    import random

    from PIL import Image

    rng = random.Random(20261024)
    img = Image.new("RGB", (256, 256))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(256 * 256)])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _multipart_body(png: bytes):
    boundary = "----testboundary"
    head = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="t.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    content_type = f"multipart/form-data; boundary={boundary}".encode()
    return head + png + tail, content_type


def _analyze_via_asgi(app, origin=None, referer=None):
    """POST /api/analyze?save=false through the raw harness.

    Returns (status, counter_path_bytes_before, counter_path_bytes_after)
    only via the status here; counter assertions live in the dedicated
    test that owns tmp_path. Origin/Referer omitted when None.
    """
    body, content_type = _multipart_body(_png_bytes())
    headers = [(b"content-type", content_type)]
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    if referer is not None:
        headers.append((b"referer", referer.encode()))
    return call_app(
        app,
        "/api/analyze",
        method="POST",
        client=("127.0.0.2", 12345),
        body=body,
        query=b"save=false",
        headers=headers,
    )[0]


TRUSTED_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

HOSTILE_ORIGINS = [
    "https://evil.example",
    "http://192.168.1.5:8000",
    "http://localhost:9999",
    "http://127.0.0.1",  # default port 80, not allowlisted
    "https://127.0.0.1:8000",  # wrong scheme
    "null",
    "not-a-url",
    "http://",
    "http://evil@127.0.0.1:8000",  # userinfo smuggling
]


class TestBrowserOriginGuard:
    @pytest.mark.parametrize("origin", TRUSTED_ORIGINS)
    def test_trusted_origin_multipart_analyze_passes(self, app, origin):
        assert _analyze_via_asgi(app, origin=origin) == 200

    @pytest.mark.parametrize("origin", HOSTILE_ORIGINS)
    def test_hostile_origin_multipart_analyze_rejected(self, app, origin):
        assert _analyze_via_asgi(app, origin=origin) == 403

    def test_hostile_origin_rejected_before_handler(self, app, tmp_path):
        """A 403 from the guard must not increment the case counter."""
        import datetime
        import json

        counter = tmp_path / "seeded_case_counter.json"
        before = counter.read_bytes() if counter.exists() else b""
        assert _analyze_via_asgi(app, origin="https://evil.example") == 403
        assert (counter.read_bytes() if counter.exists() else b"") == before
        # And a trusted origin reaches the handler: counter increments.
        assert _analyze_via_asgi(app, origin="http://127.0.0.1:8000") == 200
        data = json.loads(counter.read_text())
        today = datetime.date.today().isoformat()
        seeded = json.loads(before.decode()) if before else {}
        assert data.get(today, 0) == seeded.get(today, 0) + 1

    def test_missing_origin_is_non_browser_policy(self, app):
        """No Origin -> allowed (curl/scripts/tests), NOT a safety claim."""
        assert _analyze_via_asgi(app) == 200

    def test_referer_is_deny_only(self, app):
        # No Origin + hostile Referer -> 403.
        assert _analyze_via_asgi(app, referer="https://evil.example/page") == 403
        # No Origin + malformed Referer -> 403.
        assert _analyze_via_asgi(app, referer="not-a-url") == 403
        # No Origin + allowlisted Referer -> same as missing Origin: allowed.
        assert _analyze_via_asgi(app, referer="http://127.0.0.1:8000/app") == 200

    def test_json_request_hostile_origin_403_trusted_reaches_handler(self, app):
        from fastapi.testclient import TestClient

        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/api/datasets/register", json={}, headers={"Origin": "https://evil.example"})
        assert r.status_code == 403
        # Trusted origin reaches the handler (handler rejects the empty
        # body itself; the point is it is NOT a 403 from the guard).
        r = client.post("/api/datasets/register", json={}, headers={"Origin": "http://127.0.0.1:8000"})
        assert r.status_code != 403

    def test_simple_form_post_hostile_origin_403(self, app):
        # A browser simple form POST always carries Origin; a hostile
        # one must be rejected before the handler runs.
        headers = [
            (b"content-type", b"application/x-www-form-urlencoded"),
            (b"origin", b"https://evil.example"),
        ]
        status, _ = call_app(
            app, "/api/benchmark/run", method="POST", body=b"dataset=nope", query=b"dataset=nope", headers=headers
        )
        assert status == 403
        headers[1] = (b"origin", b"http://127.0.0.1:8000")
        status, _ = call_app(
            app, "/api/benchmark/run", method="POST", body=b"dataset=nope", query=b"dataset=nope", headers=headers
        )
        assert status != 403

    def test_preflight_not_blocked_by_origin_guard(self, app):
        """OPTIONS has no side effects: CORS answers it, F8 never sees it."""
        from fastapi.testclient import TestClient

        client = TestClient(app, raise_server_exceptions=False)
        r = client.options(
            "/api/health",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
        assert r.status_code != 403
        assert r.headers.get("access-control-allow-origin") in (None, "")
        r = client.options(
            "/api/health",
            headers={"Origin": "http://127.0.0.1:8000", "Access-Control-Request-Method": "POST"},
        )
        assert r.headers.get("access-control-allow-origin") == "http://127.0.0.1:8000"


STATE_CHANGING_ROUTES = {
    ("DELETE", "/api/history"),
    ("DELETE", "/api/history/{analysis_id}"),
    ("POST", "/api/models/reload"),
    ("POST", "/api/datasets/register"),
    ("POST", "/api/benchmark/run"),
    ("POST", "/api/experiments/blind"),
    ("POST", "/api/experiments/{experiment_id}/run"),
    ("POST", "/api/experiments/{experiment_id}/reveal"),
    ("POST", "/api/experiments/{experiment_id}/invalidate"),
    ("POST", "/api/models/{model_id}/promote"),
    # Compute routes: every run_analysis call increments the case
    # counter, and analyze additionally writes history on save=true.
    # /api/robustness included: it calls run_analysis per transform
    # (routes.py via benchmark.robustness.evaluate_robustness).
    ("POST", "/api/analyze"),
    ("POST", "/api/compare"),
    ("POST", "/api/batch"),
    ("POST", "/api/explain"),
    ("POST", "/api/robustness"),
}


class TestOriginGuardCoverage:
    def test_every_state_changing_route_carries_origin_guard(self, app):
        from backend.api.local_guard import require_trusted_browser_origin

        found = set()
        for route in TestPrivilegedRouteCoverage._all_routes(app.routes):
            methods = getattr(route, "methods", None) or set()
            path = getattr(route, "path", "")
            deps = getattr(route, "dependencies", []) or []
            if any(getattr(d, "dependency", None) is require_trusted_browser_origin for d in deps):
                for m in methods:
                    found.add((m, path))
        missing = STATE_CHANGING_ROUTES - found
        assert not missing, f"origin guard missing on: {sorted(missing)}"


# ---------------------------------------------------------------- F9: deployment boundary (V5.1.8.60)
#
# The middleware gates on the transport PEER, not the bind address:
# it covers every startup path (including the uvicorn CLI invocation
# that bypasses the __main__ bind check) precisely because the app
# cannot reliably know its own bind host. Peer checks are NOT
# authentication -- see local_guard.py.


def _run_middleware(client):
    """Run LoopbackOnlyMiddleware over a dummy inner app.

    Returns (status_or_None, inner_was_called).
    """
    from backend.api.local_guard import LoopbackOnlyMiddleware

    state: dict = {}

    async def inner(scope, receive, send):
        state["called"] = True
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            state["status"] = message["status"]

    scope = {"type": "http", "method": "GET", "path": "/", "headers": [], "client": client}
    asyncio.run(LoopbackOnlyMiddleware(inner)(scope, receive, send))
    return state.get("status"), state.get("called", False)


class TestLoopbackMiddleware:
    @pytest.mark.parametrize("client", [("127.0.0.1", 1), ("127.0.0.2", 1), ("::1", 1), ("testclient", 50000)])
    def test_loopback_peer_reaches_inner_app(self, client):
        status, called = _run_middleware(client)
        assert (status, called) == (200, True)

    @pytest.mark.parametrize("client", [("203.0.113.7", 1), ("192.168.1.5", 1), ("example.com", 1), None])
    def test_non_loopback_peer_blocked_before_inner_app(self, client):
        status, called = _run_middleware(client)
        assert (status, called) == (403, False)

    def test_lifespan_scope_passes_through(self):
        from backend.api.local_guard import LoopbackOnlyMiddleware

        called = []

        async def inner(scope, receive, send):
            called.append(True)

        async def receive():
            return {"type": "lifespan.startup"}

        async def send(message):
            pass

        asyncio.run(LoopbackOnlyMiddleware(inner)({"type": "lifespan"}, receive, send))
        assert called == [True]


class TestGlobalMiddlewareOnApp:
    @pytest.mark.parametrize("path", ["/api/health", "/", "/api/docs"])
    def test_non_loopback_peer_blocked_on_every_route_kind(self, app, path):
        status, _ = call_app(app, path, client=("203.0.113.7", 1))
        assert status == 403

    def test_non_loopback_post_blocked_before_handler(self, app):
        status, _ = call_app(app, "/api/analyze", method="POST", client=("203.0.113.7", 1), body=b"{}")
        assert status == 403

    def test_loopback_requests_unaffected(self, app):
        status, _ = call_app(app, "/api/health")
        assert status == 200
        status, length = call_app(app, "/")
        assert status == 200 and length == INDEX_LEN
        status, _ = call_app(app, "/api/docs")
        assert status == 200

    def test_loopback_preflight_still_answered_by_cors(self, app):
        from fastapi.testclient import TestClient

        client = TestClient(app, raise_server_exceptions=False)
        r = client.options(
            "/api/health",
            headers={"Origin": "http://127.0.0.1:8000", "Access-Control-Request-Method": "GET"},
        )
        assert r.headers.get("access-control-allow-origin") == "http://127.0.0.1:8000"


class TestServeBindAddress:
    @pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "localhost"])
    def test_loopback_binds_accepted(self, host):
        serve = _load_serve()
        assert serve._bind_address(host) == host

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.com", ""])
    def test_non_loopback_binds_refused_without_serving(self, host):
        # Pure function test: _bind_address raises before any socket
        # is opened, so no server ever starts.
        serve = _load_serve()
        with pytest.raises(SystemExit):
            serve._bind_address(host)
