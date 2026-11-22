"""Truth Lenz AI - FastAPI application entrypoint.

Serves the JSON API under /api/* and the built frontend from /.
Privacy: uploads are processed in memory and never written to disk unless
the caller explicitly opts into history (save=true).
"""
from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.body_limits import BodyLimitMiddleware
from .api.local_guard import LoopbackOnlyMiddleware, validate_bind_host
from .api.routes import router

app = FastAPI(
    title="Truth Lenz AI",
    description="AI Image Authenticity & Digital Forensics Platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    # Loopback origins only (V5.1.8.59, F4). CORS is defense in depth,
    # not authentication: it gates whether a browser page may READ a
    # cross-origin response, it does not stop a request being sent.
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_methods=["GET", "POST", "DELETE"],
    # Stage 3 (S3-5): explicit header allowlist. The frontend sends
    # exactly one custom header (Content-Type: application/json, in
    # api.ts); uploads send browser-set multipart Content-Type, which
    # is a CORS-safelisted simple-request header. A wildcard here
    # would matter the moment any credential or Authorization header
    # is introduced, so it is closed now while the list is provably
    # complete. Origins stay loopback-only; credentials stay disabled.
    allow_headers=["Content-Type"],
)

# V5.1.8.60 (F9): global loopback peer gate, added after CORS so it
# runs outermost (first). It gates on the transport peer, not the
# bind address, so non-loopback peers are rejected even on the
# uvicorn CLI invocation that bypasses the __main__ bind check below.
# It does NOT prevent a non-loopback socket from opening; it only
# guarantees no non-loopback peer reaches any route (see README).
# Loopback peers pass through to CORS and routing unchanged;
# non-loopback peers get a 403 before anything else runs. This is
# defense in depth, not authentication.
#
# Stage 3 (S3-3): the body-limit middleware is added BEFORE the
# loopback gate in add order, so in execution order it runs AFTER it:
# Loopback -> BodyLimit -> CORS -> router. Non-loopback peers still
# get their 403 first; loopback upload requests then get size-gated
# before any handler or multipart parsing runs.
app.add_middleware(BodyLimitMiddleware)
app.add_middleware(LoopbackOnlyMiddleware)

app.include_router(router)


@app.exception_handler(Exception)
async def _safe_errors(request: Request, exc: Exception):
    # Never leak tracebacks to clients.
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred during analysis. "
                           "Please try again with a different image."},
    )


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

    @app.get("/{path:path}")
    async def _spa(path: str):
        # API routes are matched first; everything else serves the SPA.
        # V5.1.8.59 (F1): canonical-path containment. A plain prefix
        # check is not enough (symlink escapes, prefix collisions), so
        # resolve both sides and require commonpath equality.
        # Stage 3 (S3-1): unknown /api/* paths must NOT fall through to
        # the SPA with HTTP 200 — API clients would misparse HTML as
        # JSON and typos would silently "succeed". Return a stable
        # JSON 404 instead. Checked before the file branch so no file
        # under dist/ can ever satisfy an /api/* path.
        if path == "api" or path.startswith("api/"):
            return JSONResponse(
                status_code=404,
                content={"detail": "API route not found."},
            )
        base = os.path.realpath(FRONTEND_DIR)
        candidate = os.path.realpath(os.path.join(FRONTEND_DIR, path))
        try:
            inside = os.path.commonpath([base, candidate]) == base
        except ValueError:
            inside = False
        if path and os.path.isfile(candidate) and inside:
            return FileResponse(candidate)
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    # V5.1.8.59 (F7), corrected V5.1.8.61: fail closed on non-loopback
    # binds. No token authentication exists, so a non-loopback host
    # must not start. The SAME variable is validated and then passed
    # to uvicorn.run, so a future edit cannot change the bind without
    # also changing what is validated.
    _BIND_HOST = "127.0.0.1"
    validate_bind_host(_BIND_HOST)
    uvicorn.run("backend.main:app", host=_BIND_HOST, port=8000, reload=False)
