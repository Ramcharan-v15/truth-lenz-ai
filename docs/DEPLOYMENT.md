# TRUTH LENZ AI — Deployment and Production Bundle Guide

This document states the only supported deployment, the protections it
does and does not provide, and the required production-bundle procedure.
It is part of the Stage 3 (S3-7/S3-8) security hardening. If any part of
this document conflicts with an operational habit, this document wins:
do not improvise a deployment outside it.

## 1. Supported deployment: single-process, loopback-only

- Start the backend exactly like this, from the repository root:

  ```bash
  .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
  ```

  (The `python backend/main.py` entry point additionally refuses
  non-loopback binds before starting; the uvicorn CLI path does not
  have that check, so the host flag above is load-bearing — never
  change it without re-reading section 2.)
- One process, no workers, no reload in production use. Rate limiting
  is an in-memory per-process store; multi-worker deployment would
  multiply the limits and is not supported.
- The frontend is served by the backend itself from the tracked
  `frontend/dist` bundle (see section 4). For development only,
  `npm run dev` proxies `/api` to port 8000.

## 2. What is NOT supported

- **No reverse proxy.** Do not put nginx, Caddy, Apache, a cloud load
  balancer, or any other proxy in front of this application. Forwarded
  headers (`X-Forwarded-For`, `X-Real-IP`, `Forwarded`) are untrusted
  by design: the loopback gate and the rate limiter key on the
  transport peer only, and any proxy would either break that invariant
  (all clients sharing the proxy's IP) or invite spoofing. Proxy
  support requires a separately approved trusted-proxy design.
- **No LAN, no public internet, no network deployment of any kind.**
  The loopback-peer middleware rejects non-loopback peers from every
  route, but that is a safety net, not a deployment model. Binding to
  `0.0.0.0` or any non-loopback address is prohibited (see section 3).
- **No multi-user deployment.** This is a single-trusted-user local
  tool. There are no accounts, roles, or per-user data separation.

## 3. Authentication: not implemented — say so plainly

**There is no authentication in this application.** Loopback binding,
the loopback-peer middleware, browser-origin checks, Fetch Metadata
checks, and CORS are not authentication and must never be described as
such. Any local process or OS user can call every route, read every
analysis and registry entry, and perform every privileged mutation
(history deletion, model reload, dataset registration, experiment
create/reveal/invalidate, model promotion, benchmark runs).

Consequences:

- Never start the server on a non-loopback host.
- Never expose the port beyond the local machine (no port forwarding,
  no container port publishing to a host interface, no tunnels that
  accept remote peers).
- Network use of any kind requires implementing authentication first,
  as a separately authorized project. Until then, LAN/internet use is
  unsupported, full stop.

## 4. Production bundle procedure (tracked `frontend/dist`)

`frontend/dist` is tracked in git and is what the backend serves. It
goes stale whenever frontend source changes without a rebuild — this
already happened once (Stage 1/2 fixes were unserved until rebuilt).
The rule:

1. After any frontend source change, and only after frontend tests
   pass, rebuild from the repository root:

   ```bash
   cd frontend && npm run build
   ```

   Do not install or upgrade dependencies as part of a rebuild unless
   separately approved. The build writes only to `frontend/dist`
   (vite empties that directory first, so old hashed assets are
   replaced, not accumulated).
2. Verify the rebuild before relying on it:
   - `git status --short` shows only the expected dist changes
     (modified `index.html`, one deleted old hashed JS file, one new
     hashed JS file; CSS unchanged unless styles changed).
   - The new bundle contains the current source markers — run the
     freshness regression test:
     `pytest tests/test_stage3_security.py::TestBundleFreshness -q`
     It fails closed if source fixes exist but the bundle is stale.
   - Run the production-path browser tests:
     `pytest tests/test_production_bundle.py -q`
     They serve the real `frontend/dist` through the real backend on
     loopback with intercepted API responses.
3. If a rebuild produces a bad bundle, roll back by restoring the
   previous dist files from the Stage 3 external snapshot
   (`~/workspace/truth-lenz-ai-stage3-snapshot-2026-10-24/files/frontend/dist/`),
   by explicit file copy — never by `git checkout`, `reset`, or
   `clean`, which could destroy unrelated uncommitted work.

Note for the Stage 3 release itself: Stage 3 changed no frontend
source (all changes are backend, tests, and docs), so no rebuild was
performed or needed; the tracked bundle was verified current by the
freshness test and the production-path tests instead.

## 5. Request limits (summary; code is authoritative)

- Uploads: 25 MB per file (decoded bytes); 100 MB aggregate decoded
  file bytes per batch; transport body caps of 26 MB (single-file
  routes), 51 MB (compare, 2 files), 101 MB (batch) including
  multipart framing. Oversize bodies get HTTP 413; chunked bodies are
  aborted mid-stream once the cap is crossed; rejected uploads are
  closed and their spool storage released.
- Blind-experiment runs (`POST /api/experiments/{id}/run`): 20 files
  maximum, 25 MB per file (decoded bytes), 100 MB aggregate decoded
  bytes, 101 MB transport cap including multipart framing. All files
  are read and validated before any blind inference runs; a limit
  breach rejects the whole run with 413 (size) or 400 (file count)
  and no inference is performed. The run route's parameterized path
  is matched by an explicit pattern in the body-limit middleware
  (exact-path keys cannot match a variable experiment id).
- Privileged JSON bodies (dataset registration, blind-experiment
  create/reveal/invalidate, model promotion): 1 MB.
- Rate limits (per client IP, 60-second windows): analysis routes
  30/min; privileged mutations 30/min; destructive history deletes
  10/min; benchmark runs 5/min. All return HTTP 429 with the same
  message shape; windows recover automatically.
- CORS: loopback origins only (ports 8000/5173), methods GET/POST/
  DELETE, allowed headers exactly `Content-Type`, credentials
  disabled. Browser-origin and Fetch Metadata checks constrain
  browser clients only and are not authentication (see section 3).

## 6. Scientific-integrity note

Security work changes nothing about validation: TRUTH LENZ AI is not
validated as a real-world AI-image detector, and no security fix in
Stage 3 introduces, implies, or relies on any accuracy or calibration
claim. Model weights, thresholds, datasets, validation results, and
review artifacts are untouched by deployment and bundle operations.
