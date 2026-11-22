"""Portable test-server helpers (Windows + POSIX).

Why this module exists: the vite/backend server fixtures previously
launched ``npm`` (a ``.cmd`` shim on Windows, not directly executable
by CreateProcess), used ``.venv/bin/python`` (Windows uses
``.venv\\Scripts\\python.exe``), wrote probe images to hard-coded
``/tmp/...`` (no ``/tmp`` on Windows), and cleaned up with
``os.killpg``/``SIGKILL`` (Unix-only). Each of those passes on Linux
and fails on Windows with ``FileNotFoundError: [WinError 2]`` or
``AttributeError`` — exactly the Windows-only suite failures this
module fixes. Test purposes, assertions, and honesty states are
unchanged; only how servers are launched, waited on, and stopped
changes.

Design notes:
- Vite is launched directly via the node binary
  (``node_modules/vite/bin/vite.js``), not via ``npm run dev``.
  The frontend ``dev`` script is exactly ``vite``, so the server is
  identical; launching node directly avoids the ``npm.cmd`` wrapper
  entirely, and the vite process is the direct child, so
  terminate/kill cleans it up on both platforms without process
  groups.
- Backend servers use ``sys.executable`` (the interpreter running
  the tests, i.e. the venv python on both platforms), never a
  hard-coded venv path.
- Process creation uses ``start_new_session`` on POSIX and
  ``CREATE_NEW_PROCESS_GROUP`` on Windows; cleanup is
  terminate → wait → kill → wait, with no Unix-only APIs.
- Probe images go to ``tempfile.gettempdir()`` (``/tmp`` on Linux,
  ``%TEMP%`` on Windows), never a hard-coded ``/tmp``.
- Readiness polls both IPv6 and IPv4 loopback for vite, because
  vite binds the IPv6 loopback and ``localhost`` resolution differs
  by platform.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def vite_command(repo: str, port: int) -> list[str]:
    """Command launching the vite dev server for ``repo/frontend``."""
    node = shutil.which("node")
    if not node:
        raise RuntimeError(
            "node executable not found on PATH; install Node.js to run "
            "frontend dev-server tests")
    vite_js = os.path.join(repo, "frontend", "node_modules", "vite",
                           "bin", "vite.js")
    if not os.path.isfile(vite_js):
        raise RuntimeError(
            f"vite binary not found: {vite_js}; run 'npm install' in "
            "frontend/ first")
    return [node, vite_js, "--port", str(port), "--strictPort"]


def backend_command(port: int) -> list[str]:
    """Command launching the backend (uvicorn) under the test python."""
    return [sys.executable, "-m", "uvicorn", "backend.main:app",
            "--host", "127.0.0.1", "--port", str(port)]


def popen_kwargs() -> dict:
    """Platform-appropriate process-group kwargs for Popen."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def stop_process(proc: subprocess.Popen, timeout: float = 10) -> None:
    """Terminate a test server, escalating to kill on timeout.

    Safe to call on an already-exited process. Never raises for a
    missing process; other OSError failures are also swallowed here
    because fixture teardown must not mask the test result.
    """
    try:
        proc.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            pass


def _wait_ready(urls: list[str], proc: subprocess.Popen,
                timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                continue
        time.sleep(1)
    stop_process(proc)
    raise RuntimeError(f"server did not become ready in {timeout:.0f}s "
                       f"(tried {', '.join(urls)})")


def start_vite_server(repo: str, port: int,
                      timeout: float = 60) -> subprocess.Popen:
    """Start vite dev for ``repo/frontend``; return the ready process."""
    proc = subprocess.Popen(
        vite_command(repo, port),
        cwd=os.path.join(repo, "frontend"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        **popen_kwargs())
    _wait_ready([f"http://[::1]:{port}/", f"http://127.0.0.1:{port}/"],
                proc, timeout=timeout)
    return proc


def start_backend_server(repo: str, port: int,
                         timeout: float = 60) -> subprocess.Popen:
    """Start the backend on 127.0.0.1:port; return the ready process."""
    proc = subprocess.Popen(
        backend_command(port),
        cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        **popen_kwargs())
    _wait_ready([f"http://127.0.0.1:{port}/"], proc, timeout=timeout)
    return proc


def probe_path(name: str) -> str:
    """Portable probe-image path (temp dir, never hard-coded /tmp)."""
    return os.path.join(tempfile.gettempdir(), name)
