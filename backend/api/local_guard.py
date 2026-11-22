"""Loopback-only request guard (V5.1.8.59, findings F3/F7).

Privileged mutation routes must only be callable by a client on the
same machine. This module decides that from the transport peer
(``request.client.host``) -- never from forwarded headers, which any
caller can forge (``X-Forwarded-For`` / ``X-Real-IP`` are ignored).

How each client kind is handled:
- IPv4 loopback (``127.0.0.0/8``) and IPv6 loopback (``::1``): allowed.
  ``ipaddress.ip_address(host).is_loopback`` covers both ranges.
- Hostname ``localhost``: allowed. It resolves to loopback by
  definition; a manipulated local hosts file is a local-admin concern,
  not a remote-attacker vector.
- In-process test client (host ``testclient``): allowed. Starlette's
  TestClient sets this value from the in-process transport; a remote
  peer over a real socket always presents an IP address, so this value
  can never be produced by a network attacker. It exists only so the
  project's own test suite can exercise guarded routes in-process.
- Missing client, missing host, unparsable host, any other hostname,
  any non-loopback IP: DENIED (fail closed, HTTP 403).

This guard is an authorization boundary for local use, not
authentication: any local process or OS user can still call guarded
routes. It must not be described as authentication, and it does not
make non-loopback deployment safe -- see ``validate_bind_host``.
Token authentication does NOT exist in this codebase; a non-loopback
bind is therefore always refused by ``validate_bind_host``.

Browser-origin protection (V5.1.8.60, finding F8) -- see
``require_trusted_browser_origin``. It constrains BROWSER clients
only: a browser always sends an ``Origin`` header on POST/DELETE
(including simple form POSTs), so an untrusted page cannot mutate
state through the browser. It is NOT authentication -- any
non-browser caller can forge any ``Origin`` value -- and a missing
``Origin`` is a non-browser compatibility policy, not proof that a
request is safe. ``Referer`` is used only as a secondary rejection
signal (hostile/malformed Referer with no Origin -> 403); it never
grants anything and is never treated as a credential.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

_LOOPBACK_HOSTS = ("localhost", "testclient")

# The only browser origins that may perform state changes: the local
# frontend (dev server) and the API itself, on loopback. Mirrors the
# CORS allowlist in main.py; keep the two in sync.
_ALLOWED_BROWSER_ORIGINS = frozenset(
    {
        ("http", "127.0.0.1", 8000),
        ("http", "localhost", 8000),
        ("http", "127.0.0.1", 5173),
        ("http", "localhost", 5173),
    }
)


def is_loopback_host(host) -> bool:
    """True only for a loopback peer value (shared by both guards)."""
    if not host:
        return False
    if host in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_loopback_client(request: Request) -> bool:
    """True only for loopback peers (see module docstring)."""
    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client is not None else None
    return is_loopback_host(host)


class LoopbackOnlyMiddleware:
    """Pure-ASGI global loopback peer gate (V5.1.8.60, finding F9).

    Defense in depth alongside the per-route ``require_loopback``
    dependency and the ``__main__`` bind check: no non-loopback peer
    reaches ANY route (read-only, analysis, or mutation), regardless
    of how the server was started -- the documented ``uvicorn
    backend.main:app --host ...`` invocation bypasses ``__main__``,
    so a bind check there cannot cover it.

    The gate is on the transport PEER, not the bind address: a server
    bound to 0.0.0.0 still serves loopback peers, and a loopback-bound
    server still serves only loopback peers, so the peer is the
    correct invariant. It deliberately does NOT inspect the bind
    address, and it is NOT authentication: local processes and OS
    users retain full access. LAN access remains unsupported until a
    separately reviewed authentication design is approved.

    Non-HTTP scopes (lifespan) pass through untouched. The middleware
    never reads the request body: denied peers get a 403 before any
    handler or body processing runs.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            client = scope.get("client")
            host = client[0] if client else None
            if not is_loopback_host(host):
                response = JSONResponse(
                    {"detail": "This application is only available from the local machine."},
                    status_code=403,
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def require_loopback(request: Request) -> None:
    """FastAPI dependency: reject non-loopback callers of guarded routes."""
    if not is_loopback_client(request):
        raise HTTPException(
            status_code=403,
            detail="This operation is only available from the local machine.",
        )


def _origin_parts(value: str):
    """Parse an Origin/Referer value into (scheme, host, port) or None.

    Returns None for malformed values (no host, bad port, userinfo),
    so callers fail closed.
    """
    try:
        parsed = urlparse(value)
    except ValueError:
        return None
    if parsed.username or parsed.password:
        return None
    host = parsed.hostname
    if not host:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    return (parsed.scheme, host, port)


def _origin_is_allowed(value: str) -> bool:
    """True only for an exactly allowlisted loopback origin."""
    parts = _origin_parts(value)
    return parts is not None and parts in _ALLOWED_BROWSER_ORIGINS


def require_trusted_browser_origin(request: Request) -> None:
    """FastAPI dependency: browser-origin protection for mutations.

    - Origin present: must be an exactly allowlisted loopback origin,
      else 403. Hostile, malformed, ``null``, wrong-scheme, and
      wrong-port origins are all rejected.
    - Origin missing: allowed as a non-browser compatibility policy
      (curl, scripts, in-process tests send no Origin). This is NOT
      authentication and NOT proof of safety; loopback/auth guards
      still apply where wired.
    - Stage 3 (S3-6) Fetch Metadata defense in depth: with Origin
      missing, an explicitly ``cross-site`` Sec-Fetch-Site value is a
      browser cross-site request, so it is rejected. Non-browser
      clients send neither header and are unaffected. Only the exact
      ``cross-site`` value is rejected; ``same-origin``, ``same-site``
      and ``none`` pass through to the Referer check below. This is
      NOT authentication: any caller can forge or omit these headers;
      it only closes the browser-shaped hole in the missing-Origin
      policy. When Origin is present, the Origin decision stands on
      its own (a trusted Origin is not second-guessed by Fetch
      Metadata, per the owner decision's "when Origin is absent"
      wording).
    - Referer fallback (deny-only): with no Origin, a present Referer
      whose origin is not allowlisted (or is malformed) -> 403. An
      allowlisted or absent Referer changes nothing.
    """
    origin = request.headers.get("origin")
    if origin is not None:
        if not _origin_is_allowed(origin):
            raise HTTPException(
                status_code=403,
                detail="Browser origin not allowed for state-changing requests.",
            )
        return
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site is not None and fetch_site.lower() == "cross-site":
        raise HTTPException(
            status_code=403,
            detail="Cross-site request not allowed for state-changing requests.",
        )
    referer = request.headers.get("referer")
    if referer is not None and not _origin_is_allowed(referer):
        raise HTTPException(
            status_code=403,
            detail="Referer origin not allowed for state-changing requests.",
        )


def validate_bind_host(host: str) -> None:
    """Fail closed on non-loopback binds.

    Token authentication has NOT been implemented, so there is no
    approved mechanism that would make a non-loopback bind safe. Any
    non-loopback host raises, and the server must not start.
    """
    if host in ("localhost",):
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise RuntimeError(
        f"Refusing non-loopback bind host {host!r}: no authentication "
        "mechanism is implemented, so the server may only bind to a "
        "loopback address (127.0.0.1 or ::1)."
    )
