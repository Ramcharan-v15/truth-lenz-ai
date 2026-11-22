"""Test isolation (V5.1.8.60, F10).

Every test runs with runtime-write paths redirected to ``tmp_path`` so
the suite can never modify the user's real files:

- ``backend/services/storage.py::DB_PATH`` (analysis history SQLite)
- ``backend/services/pipeline.py::_CASE_COUNTER_PATH`` (case counter)
- ``backend/validation/__init__.py::RESULTS_DIR`` (experiment records)
- ``backend/validation/__init__.py::COUNTER_PATH`` (experiment counter)
- ``backend/datasets/registry.py::_REGISTRY_PATH`` (dataset registry)

All five are module-level constants read at call time, so
``monkeypatch.setattr`` on the module redirects every writer,
including API handlers (paths are never cached at import time).

Seeded copies (not empty tmp dirs): the small real files are copied
into tmp before each test, so tests that legitimately assert on
existing workspace state (notably ``test_v51.py``, which checks the
Columbia dataset registration, its licence notes, its quality audit,
and the presence of a real blind experiment) see exactly the state
they saw before isolation -- while every write still lands in tmp.
Files are copied only if present, so a fresh clone without them
still works.

Seeded files use distinct ``seeded_*`` names: existing fixtures
already reuse conventional tmp names (``test_v5_datasets``'s ``iso``
fixture uses ``tmp_path / "registry.json"`` expecting an empty
registry, and ``test_v4``/``test_v5_benchmark`` use
``tmp_path / "results"`` and ``tmp_path / "counter.json"``).
Seeding under those same names would pollute those fixtures with
real state; distinct names keep both behaviors intact.

The experiment-results copy is scoped to ``test_v51`` only: it is
the sole module asserting on pre-existing experiment records, and
copying the results tree per test would fill small tmp filesystems
over a full run. Every other module either creates its own records
or redirects ``RESULTS_DIR`` itself.

One deliberate exception: ``history.db`` (~11 MB) is NOT copied --
no test reads pre-existing history rows (the history tests create,
read, and delete their own entries), and copying it per test would
be wasteful. History tests therefore start from an empty tmp
database, which is exactly what they need.

Deliberately NOT redirected here:
- ``backend/models/promotion.py`` paths (``models/registry.json``,
  ``models/promotions.json``): promotion tests already back up and
  restore those files themselves (see ``test_v5_blind.py``'s
  ``registry_backup`` fixture), and redirecting the model registry
  would break tests that must read the real committed model entry.
  That fixture is the documented isolation for promotion tests.
- Fixture/test data under ``test_data/`` and dataset manifests
  under ``datasets/manifests/`` (reads only, never written).

If a test genuinely needs persistent state, it must declare that
requirement with its own tmp-based fixture (module-scoped tmp, still
never the real files) rather than weakening this fixture.
"""

import os
import shutil

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _copy_if_present(src, dst):
    if os.path.isfile(src):
        shutil.copyfile(src, dst)


def _copytree_if_present(src, dst):
    if os.path.isdir(src):
        shutil.copytree(src, dst)


@pytest.fixture(autouse=True)
def _isolate_runtime_writes(tmp_path, monkeypatch, request):
    from backend import validation as val
    from backend.datasets import registry as reg
    from backend.services import pipeline, storage

    # Seed tmp copies of the small real files (see module docstring).
    # Distinct seeded_* names avoid colliding with existing fixtures
    # that reuse conventional tmp names expecting empty state.
    registry_tmp = tmp_path / "seeded_registry.json"
    _copy_if_present(os.path.join(REPO, "datasets", "registry.json"), str(registry_tmp))
    counter_tmp = tmp_path / "seeded_case_counter.json"
    _copy_if_present(
        os.path.join(REPO, "backend", "data", "case_counter.json"), str(counter_tmp)
    )
    exp_counter_tmp = tmp_path / "seeded_counter.json"
    _copy_if_present(
        os.path.join(REPO, "validation", "experiment_counter.json"), str(exp_counter_tmp)
    )
    results_tmp = tmp_path / "seeded_results"
    if "test_v51" in request.node.nodeid:
        _copytree_if_present(os.path.join(REPO, "validation", "results"), str(results_tmp))

    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "seeded_history.db"))
    monkeypatch.setattr(pipeline, "_CASE_COUNTER_PATH", str(counter_tmp))
    monkeypatch.setattr(val, "RESULTS_DIR", str(results_tmp))
    monkeypatch.setattr(val, "COUNTER_PATH", str(exp_counter_tmp))
    monkeypatch.setattr(reg, "_REGISTRY_PATH", str(registry_tmp))
