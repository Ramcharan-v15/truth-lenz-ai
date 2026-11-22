"""Windows-portability regression tests for test-server helpers.

These tests pin the portable APIs introduced in
``tests/portable_servers.py`` after the Windows suite failed with
``FileNotFoundError: [WinError 2]`` in the vite/backend server
fixtures. They do NOT run a browser or a vite server; they assert
the launch commands and cleanup paths themselves are portable.

Under the old behaviour these tests fail: there was no
``portable_servers`` module, vite was launched via ``npm`` (a
``.cmd`` shim on Windows), backend servers used ``.venv/bin/python``,
probe images went to hard-coded ``/tmp/...``, and cleanup used
Unix-only ``os.killpg``/``SIGKILL``. The old code passed on Linux
and failed only on Windows, so these tests prove the new code uses
portable APIs — they cannot reproduce a Windows CreateProcess
failure on Linux, and they do not claim to.
"""

import os
import subprocess
import sys
import tempfile

import portable_servers as ps


def test_vite_command_uses_node_not_npm():
    cmd = ps.vite_command(ps.REPO, 5199)
    assert "npm" not in cmd, f"vite must not be launched via npm: {cmd}"
    assert os.path.basename(cmd[0]).startswith("node")
    assert cmd[1].endswith(os.path.join("vite", "bin", "vite.js"))
    assert os.path.isfile(cmd[1])
    assert "--port" in cmd and "5199" in cmd and "--strictPort" in cmd


def test_vite_command_missing_node_raises_clearly(monkeypatch):
    monkeypatch.setattr(ps.shutil, "which", lambda name: None)
    try:
        ps.vite_command(ps.REPO, 5199)
    except RuntimeError as exc:
        assert "node" in str(exc).lower()
    else:  # pragma: no cover - explicit failure path
        raise AssertionError("expected RuntimeError when node is missing")


def test_backend_command_uses_sys_executable():
    cmd = ps.backend_command(8001)
    # Portability is USING sys.executable, not avoiding a venv path
    # value: on a Linux venv, sys.executable IS .../.venv/bin/python,
    # and on a Windows venv it IS ...\.venv\Scripts\python.exe.
    # A hard-coded ".venv/bin/python" string fails on Windows even
    # though it happens to equal sys.executable on Linux.
    assert cmd[0] == sys.executable, (
        f"backend must run under the test interpreter, not a hard-coded "
        f"venv path: {cmd[0]}")
    assert "uvicorn" in cmd and "backend.main:app" in cmd
    assert "127.0.0.1" in cmd and "8001" in cmd


def test_popen_kwargs_are_platform_portable():
    kwargs = ps.popen_kwargs()
    if os.name == "nt":
        assert kwargs.get("creationflags") == subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert kwargs.get("start_new_session") is True
    # The kwargs must actually be accepted by Popen on this platform.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
    try:
        assert proc.poll() is None
    finally:
        ps.stop_process(proc)
    assert proc.poll() is not None


def test_stop_process_terminates_child():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert proc.poll() is None
    ps.stop_process(proc, timeout=10)
    assert proc.poll() is not None, "stop_process must terminate the child"


def test_stop_process_on_exited_process_does_not_raise():
    proc = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait(timeout=10)
    ps.stop_process(proc)  # must not raise for an already-exited process


def test_probe_path_follows_tempdir_not_hardcoded_tmp(monkeypatch):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: "/fake/tmp")
    assert ps.probe_path("stage2_probe.png") == "/fake/tmp/stage2_probe.png"
    # And without monkeypatching, it is the real temp dir on this machine.
    monkeypatch.undo()
    assert ps.probe_path("stage2_probe.png") == os.path.join(
        tempfile.gettempdir(), "stage2_probe.png")
