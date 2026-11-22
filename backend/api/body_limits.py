"""Transport-level request-body limits (Stage 3, S3-3).

How limits are measured (read this before changing any number):

- Semantic payload caps are enforced on *decoded* bytes by the route
  handlers: 25 MB per uploaded file (``_read_upload``) and 100 MB
  aggregate decoded file bytes per batch (batch handler). Those are the
  limits the API documents.
- This middleware enforces *transport* caps on total HTTP request-body
  bytes, which include multipart framing (boundaries, part headers).
  Transport caps are therefore the semantic cap plus a 1 MB framing
  allowance: 26 MB for single-file routes, 51 MB for compare (2 files),
  101 MB for batch (8 files). A request that passes the transport cap
  can still be rejected by the stricter semantic caps inside the
  handler; a request that exceeds the transport cap is rejected here
  before the body is ever parsed.

Two rejection paths, both returning HTTP 413 JSON:

- Declared ``Content-Length`` above the cap: rejected immediately, zero
  body bytes consumed.
- Chunked or misleading-length bodies: a counting ``receive`` wrapper
  raises ``HTTPException(413)`` the moment the cumulative body size
  crosses the cap, so the multipart parser never sees the rest of the
  body. The exception is handled by Starlette's ExceptionMiddleware
  (inner to this middleware), which renders the standard 413 JSON.

Only the listed upload routes are wrapped. Every other route, including
JSON API routes (which use the capped reader in ``routes.py``) and the
SPA/static serving, passes through untouched. This middleware never
reads the body itself beyond counting bytes, never trusts forwarded
headers, and makes no authentication decision.

Parameterized upload routes cannot be matched by the exact-path table
below (their path contains a variable segment), so they are matched by
an explicit pattern in ``_transport_cap``. Any new parameterized
upload route must be added there, with a cap, or it will silently
bypass transport protection (Stage 3 remediation, F1).
"""

import json

from fastapi import HTTPException

MB = 1024 * 1024
FRAMING_ALLOWANCE = 1 * MB

# Route path -> maximum total request-body bytes (semantic cap + framing).
BODY_LIMITS = {
    "/api/analyze": 25 * MB + FRAMING_ALLOWANCE,
    "/api/explain": 25 * MB + FRAMING_ALLOWANCE,
    "/api/robustness": 25 * MB + FRAMING_ALLOWANCE,
    "/api/compare": 50 * MB + FRAMING_ALLOWANCE,
    "/api/batch": 100 * MB + FRAMING_ALLOWANCE,
}

# Stage 3 remediation (F1): the blind-experiment run route carries a
# variable experiment id, so no exact table key can match it. Its
# transport cap mirrors the batch cap: 100 MB aggregate decoded
# uploads plus the framing allowance.
EXPERIMENT_RUN_TRANSPORT_CAP = 100 * MB + FRAMING_ALLOWANCE


def _transport_cap(path: str):
    """Transport cap for an upload path, or None if unwrapped.

    Exact table entries first; then the explicit parameterized
    pattern for ``POST /api/experiments/{experiment_id}/run``.
    The pattern requires the ``/run`` suffix so it cannot match
    unrelated experiment sub-paths (e.g. ``/reveal``).
    """
    cap = BODY_LIMITS.get(path)
    if cap is not None:
        return cap
    if path.startswith("/api/experiments/") and path.endswith("/run"):
        return EXPERIMENT_RUN_TRANSPORT_CAP
    return None

TOO_LARGE_DETAIL = "Request body too large."


class BodyLimitMiddleware:
    """Pure-ASGI body-size gate for upload routes (see module docstring)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        cap = _transport_cap(scope.get("path", ""))
        if cap is None:
            await self.app(scope, receive, send)
            return

        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        declared = headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > cap:
                    await self._send_413(send)
                    return
            except ValueError:
                pass  # Malformed length: fall through to the counter.

        seen = 0

        async def counting_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > cap:
                    raise HTTPException(status_code=413, detail=TOO_LARGE_DETAIL)
            return message

        await self.app(scope, counting_receive, send)

    @staticmethod
    async def _send_413(send):
        payload = json.dumps({"detail": TOO_LARGE_DETAIL}).encode()
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": payload})
