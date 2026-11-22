#!/usr/bin/env python3
"""Production-style static server for Truth Lenz AI frontend.

Serves frontend/dist with SPA fallback AND reverse-proxies /api/* to the
backend (default http://127.0.0.1:8000). Without the proxy, the built app's
relative /api calls fail.

Usage:
    python serve.py [--port 5173] [--api http://127.0.0.1:8000] [--dir frontend/dist]
    [--host 127.0.0.1]

The static server is loopback-only by design (V5.1.8.60, F9): --host
accepts only loopback addresses and anything else is refused before
any socket is opened. This mirrors backend/api/local_guard.py without
importing it, so serve.py keeps zero third-party dependencies.
"""
import argparse
import ipaddress
import os
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse


def _bind_address(host: str) -> str:
    """Return host if it is loopback, else refuse before serving.

    Accepts 127.0.0.0/8, ::1, and localhost. Anything else raises
    SystemExit: there is no authentication mechanism that would make
    a non-loopback bind safe, so the server must not start.
    """
    if host == "localhost":
        return host
    try:
        if ipaddress.ip_address(host).is_loopback:
            return host
    except ValueError:
        pass
    raise SystemExit(
        f"Refusing non-loopback bind host {host!r}: serve.py is "
        "loopback-only (no authentication is implemented)."
    )


def _resolve_static(www_dir, request_path):
    """Return the file to serve for a static request, or None.

    V5.1.8.59 (F2): canonical-path containment, mirroring the FastAPI
    SPA fallback. Plain prefix comparison is not enough (symlink
    escapes, prefix collisions), so both sides are resolved and
    ``os.path.commonpath`` must equal the www root. Returns None for
    any path escaping the directory, including traversal and symlink
    escapes; the caller then falls back to index.html.
    """
    base = os.path.realpath(www_dir)
    candidate = os.path.realpath(os.path.join(www_dir, request_path.lstrip("/")))
    try:
        inside = os.path.commonpath([base, candidate]) == base
    except ValueError:
        inside = False
    if inside and os.path.isfile(candidate):
        return candidate
    return None


class Handler(BaseHTTPRequestHandler):
    api_base = "http://127.0.0.1:8000"
    www_dir = "frontend/dist"

    def _proxy_api(self):
        url = self.api_base + self.path
        data = None
        if self.command in ("POST", "PUT", "PATCH"):
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length) if length else None
        req = urllib.request.Request(url, data=data, method=self.command)
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length"):
                req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            body = f'{{"error": "proxy failed: {e}"}}'.encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _serve_static(self):
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            self._proxy_api()
            return
        if path == "/":
            self._send_file(os.path.join(self.www_dir, "index.html"))
            return
        resolved = _resolve_static(self.www_dir, path)
        if resolved is not None:
            self._send_file(resolved)
        else:
            # SPA fallback (also the answer for escaping paths).
            self._send_file(os.path.join(self.www_dir, "index.html"))

    def _send_file(self, fpath):
        try:
            with open(fpath, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            return
        ctype = "text/html"
        if fpath.endswith(".js"):
            ctype = "application/javascript"
        elif fpath.endswith(".css"):
            ctype = "text/css"
        elif fpath.endswith(".json"):
            ctype = "application/json"
        elif fpath.endswith(".png"):
            ctype = "image/png"
        elif fpath.endswith(".svg"):
            ctype = "image/svg+xml"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _serve_static

    def do_POST(self):
        if urlparse(self.path).path.startswith("/api/"):
            self._proxy_api()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5173)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--dir", default="frontend/dist")
    args = ap.parse_args()
    host = _bind_address(args.host)
    Handler.api_base = args.api.rstrip("/")
    Handler.www_dir = args.dir
    print(f"Serving {args.dir} on {host}:{args.port}, /api -> {Handler.api_base}",
          flush=True)
    HTTPServer((host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
