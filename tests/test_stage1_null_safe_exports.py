"""Stage 1 null-safe results and exports (V5.1.8.67).

Covers H2 (PDF/CSV/JSON exports on a fallback report whose probabilities are
None), L11 (empty-history average confidence is None, not a fabricated 0%),
and the shared export helpers. Frontend rendering (H1, M6, M7, L12, L13) is
covered by the browser tests in test_stage1_frontend_fallback.py, which run
the real Analyze/History/Batch pages.

All state is isolated by tests/conftest.py (runtime writes redirect to
tmp_path); no test writes to real project databases, counters, registries,
or manifests. Probe images are PIL synthetics; no pilot images are opened.
"""

import io
import json
import random
import re
import zlib

import pytest
from PIL import Image

from backend.analyzers import ensemble as ens_mod
from backend.services import pipeline, storage
from backend.services.reporting import (
    _csv_val,
    _pct,
    _pct_raw,
    _plain,
    to_csv,
    to_json,
    to_pdf,
)


# ---------------------------------------------------------------- helpers

def _reject_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")


def _make_noise_png(size=256):
    rng = random.Random(7)
    img = Image.new("RGB", (size, size))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                 for _ in range(size * size)])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fallback_report():
    """A genuine fallback report: ensemble.combine forced to raise."""
    orig = ens_mod.EnsembleEngine.combine

    def boom(self, ctx, results):
        raise RuntimeError("stage1-test forced combine failure")

    ens_mod.EnsembleEngine.combine = boom
    try:
        return pipeline.run_analysis("noise.png", "image/png", _make_noise_png())
    finally:
        ens_mod.EnsembleEngine.combine = orig


def _pdf_stream_text(pdf: bytes) -> bytes:
    """Decode reportlab content streams (ASCII85 + Flate, or plain Flate)
    with stdlib only, so assertions can be made on the rendered text
    rather than just the byte header. Image/font streams are skipped
    implicitly: their bytes fail both decodings."""
    import base64

    out = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", pdf, re.DOTALL):
        chunk = m.group(1).strip(b"\r\n")
        for attempt in (lambda: zlib.decompress(chunk),
                        lambda: zlib.decompress(base64.a85decode(chunk, adobe=True))):
            try:
                out.append(attempt())
                break
            except Exception:
                continue
    return b"\n".join(out)


# ---------------------------------------------------------------- H2 helpers

def test_pct_helper_never_zero_fills_and_never_crashes():
    assert _pct(None, 1) == "unavailable"
    assert _pct(float("nan"), 1) == "unavailable"
    assert _pct(float("inf"), 1) == "unavailable"
    assert _pct(59.6, 1) == "59.6%"
    assert _pct(0, 1) == "0.0%", "a real zero stays a zero, it is not missing"
    assert _csv_val(None) == "unavailable"
    assert _csv_val(32.0) == 32.0
    assert _pct_raw(None) == "unavailable"
    assert _pct_raw(65.5) == "65.5%"
    assert _plain(None) == "unavailable"
    assert _plain(12) == "12"


# ---------------------------------------------------------------- H2 exports

def test_exports_succeed_on_fallback_report():
    rep = _fallback_report()
    ens = rep["ensemble"]
    assert ens["verdict"] == "INCONCLUSIVE"
    assert ens["ai_probability"] is None
    assert ens["manipulation_probability"] is None
    assert ens["authenticity_probability"] is None
    assert ens["signal_disagreement"] is None

    # JSON: strict, no NaN/Infinity constants.
    raw = to_json(rep)
    json.loads(raw, parse_constant=_reject_constant)

    # CSV: unavailable values are explicit, never silent blanks.
    csv_text = to_csv(rep)
    for field in ("ai_probability", "manipulation_probability",
                  "authenticity_probability"):
        line = next(l for l in csv_text.splitlines() if f",{field}," in l)
        assert line.endswith(",unavailable"), line

    # PDF: succeeds (previously TypeError) and labels the missing values.
    pdf = to_pdf(rep)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000
    text = _pdf_stream_text(pdf)
    assert b"unavailable" in text, "PDF must label missing probabilities"
    assert b"None%" not in text, "PDF must never render a None measurement"


def test_exports_unchanged_on_numeric_report():
    rep = pipeline.run_analysis("noise.png", "image/png", _make_noise_png())
    ens = rep["ensemble"]
    assert ens["ai_probability"] is not None

    raw = to_json(rep)
    parsed = json.loads(raw, parse_constant=_reject_constant)
    assert parsed["ensemble"]["ai_probability"] == ens["ai_probability"]

    csv_text = to_csv(rep)
    ai_line = next(l for l in csv_text.splitlines() if ",ai_probability," in l)
    assert ai_line.endswith(f",{ens['ai_probability']}"), ai_line
    for field in ("ai_probability", "manipulation_probability",
                  "authenticity_probability", "confidence"):
        line = next(l for l in csv_text.splitlines() if f",{field}," in l)
        assert not line.endswith(",unavailable"), line

    pdf = to_pdf(rep)
    assert pdf.startswith(b"%PDF")
    text = _pdf_stream_text(pdf)
    assert f"{ens['ai_probability']:.1f}%".encode() in text


# ---------------------------------------------------------------- L11 stats

def test_empty_history_average_confidence_is_none_not_zero():
    st = storage.stats()
    assert st["total"] == 0
    assert st["avg_confidence"] is None, "no measurements must not fabricate 0%"


def test_history_average_tracks_saved_reports_including_fallback():
    numeric = pipeline.run_analysis("noise.png", "image/png", _make_noise_png())
    storage.save_analysis(numeric)
    st = storage.stats()
    assert st["total"] == 1
    assert st["avg_confidence"] == round(numeric["ensemble"]["confidence"], 1)

    storage.save_analysis(_fallback_report())  # confidence 0.0, a real value
    st = storage.stats()
    assert st["total"] == 2
    assert st["avg_confidence"] is not None
    expected = round((numeric["ensemble"]["confidence"] + 0.0) / 2, 1)
    assert st["avg_confidence"] == expected
