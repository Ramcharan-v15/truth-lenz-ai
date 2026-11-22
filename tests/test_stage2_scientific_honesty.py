"""Stage 2 scientific-honesty regression tests (S2-1..S2-4).

Covers: suppressed ML never presented as calibrated (contract, PDF, CSV,
JSON, explanations); all-UNABLE verdict reason no longer claims low
resolution without a measurement; thin-coverage ML withheld by the
majority rule; valid ML rendering unchanged; API/UI/export consistency.

All pipeline runs go through the test suite's conftest isolation (tmp
redirects for case counter, history DB, etc.). Nothing here touches the
real working-tree counters, pilot images, or frozen artifacts.
"""

import base64
import io
import json
import random
import re
import zlib

import pytest
from PIL import Image

from backend.services import pipeline, reporting


# ---------------------------------------------------------------- helpers

def _png_bytes(kind: str, size: int = 256) -> bytes:
    rng = random.Random(42)
    img = Image.new("RGB", (size, size))
    if kind == "noise":
        px = [(rng.randrange(256), rng.randrange(256), rng.randrange(256))
              for _ in range(size * size)]
    elif kind == "solid":
        px = [(40, 90, 140)] * (size * size)
    else:
        px = [(0, 0, 0)] * (size * size)
    img.putdata(px)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _inject_nan_scores(monkeypatch, names):
    """Force the named analyzers to UNABLE via non-finite scores (the
    V5.1.8.62 failure path: base.run turns a NaN score into UNABLE)."""
    for a in pipeline.ANALYZERS:
        if a.name not in names:
            continue
        orig = a.analyze

        def bad(ctx, orig=orig):
            r = orig(ctx)
            r.ai_score = float("nan")
            r.manip_score = float("nan")
            r.confidence = float("nan")
            return r

        monkeypatch.setattr(a, "analyze", bad)


def _all_names():
    return {a.name for a in pipeline.ANALYZERS}


def _unable_report(monkeypatch):
    _inject_nan_scores(monkeypatch, _all_names())
    return pipeline.run_analysis("noise.png", "image/png", _png_bytes("noise"))


def _thin_report(monkeypatch):
    # Only "noise" stays usable; every other analyzer goes UNABLE.
    _inject_nan_scores(monkeypatch, _all_names() - {"noise"})
    return pipeline.run_analysis("noise.png", "image/png", _png_bytes("noise"))


def _usable_count(report):
    return sum(
        1 for s in report["signals"]
        if s["status"] != "unable" and s["confidence"] > 0
    )


def _pdf_stream_text(data: bytes) -> str:
    """Extract text from a reportlab PDF via stdlib only (streams are
    ASCII85+Flate). Same approach as the Stage 1 export tests."""
    out = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        raw = m.group(1).strip(b"\r\n")
        try:
            out.append(zlib.decompress(raw).decode("latin-1", "replace"))
        except Exception:
            try:
                out.append(zlib.decompress(base64.a85decode(raw, adobe=True)).decode("latin-1", "replace"))
            except Exception:
                pass
    # ReportLab renders the evidence-graph paragraph with word-level Tj
    # strings and escapes parentheses in literal strings, so raw stream
    # text never contains phrases like "Calibrated P(AI)" contiguously.
    # _pdf_flat_text below reconstructs readable phrases from the
    # literal strings instead.
    return "".join(out).replace("\xa0", " ")


def _pdf_flat_text(data: bytes) -> str:
    """Reconstruct readable phrases from PDF literal strings: join every
    (...) Tj string with a space, unescape \\( \\) \\\\, collapse space."""
    raw = _pdf_stream_text(data)
    parts = []
    for m in re.finditer(r"\((?:\\.|[^\\()])*\)", raw):
        s = m.group(0)[1:-1]
        s = s.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")
        parts.append(s)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _suppressed_entry(report):
    dets = [d for d in report["ensemble"]["detectors"]
            if d.get("probability_ai") is not None]
    assert dets, "expected at least one detector entry with a raw probability"
    return dets[0]


# ---------------------------------------------------------------- S2-1

def test_suppressed_ml_never_presented_as_calibrated(monkeypatch):
    report = _unable_report(monkeypatch)
    ens = report["ensemble"]
    assert ens["ml_available"] is False
    assert ens["ml_ai_probability"] is None
    assert ens["ml_suppressed_reason"] and "No usable forensic signals" in ens["ml_suppressed_reason"]

    # Contract: the raw detector output is preserved as provenance, but
    # the entry is marked unavailable + suppressed, never presentable
    # as a calibrated probability.
    d = _suppressed_entry(report)
    assert d["available"] is False
    assert d["suppressed"] is True
    assert d["suppressed_reason"] == ens["ml_suppressed_reason"]
    assert d["model_status"] == "ready"  # provenance preserved, not hidden

    # Explanations: the forensic text must not label the withheld value
    # as a calibrated probability.
    forensic = report["explanations"]["forensic"]
    assert "ML probability withheld" in forensic
    assert f"calibrated P(AI)={d['probability_ai']}%" not in forensic
    assert "calibrated P(AI)=withheld" in forensic
    # Technical explanation only mentions ML when it is available.
    assert "ML detector P(AI)" not in report["explanations"]["technical"]


def test_suppressed_ml_exports_say_withheld(monkeypatch):
    report = _unable_report(monkeypatch)
    ens = report["ensemble"]

    pdf = reporting.to_pdf(report)
    assert pdf[:4] == b"%PDF"
    text = _pdf_flat_text(pdf)
    assert "ML probability withheld" in text
    assert "ML MODELS: output withheld" in text
    # No calibrated-ML row for a withheld output anywhere in the PDF.
    assert "Calibrated P(AI)" not in text
    assert "No trained ML detector is configured" not in text
    assert "No ML model active" not in text

    csv_text = reporting.to_csv(report)
    rows = list(__import__("csv").reader(io.StringIO(csv_text)))
    cal = [r for r in rows if r[:2] == ["ml_detector", "calibrated_ai_probability"]]
    assert cal and cal[0][2] == "withheld"
    reason_rows = [r for r in rows if r[:2] == ["ml_detector", "suppression_reason"]]
    assert reason_rows and reason_rows[0][2] == ens["ml_suppressed_reason"]


def test_api_json_csv_pdf_agree_on_suppression(monkeypatch):
    """S2-4 consistency: the same report says 'withheld' in JSON, CSV,
    PDF and explanations — no surface contradicts another."""
    report = _unable_report(monkeypatch)
    ens = report["ensemble"]

    parsed = json.loads(reporting.to_json(report))
    assert parsed["ensemble"]["ml_ai_probability"] is None
    assert parsed["ensemble"]["ml_suppressed_reason"] == ens["ml_suppressed_reason"]
    assert any(d.get("suppressed") for d in parsed["ensemble"]["detectors"])

    csv_text = reporting.to_csv(report)
    assert "withheld" in csv_text
    pdf_text = _pdf_stream_text(reporting.to_pdf(report))
    assert "ML probability withheld" in pdf_text
    assert "ML probability withheld" in report["explanations"]["forensic"]


# ---------------------------------------------------------------- S2-2

def test_all_unable_reason_is_neutral_not_resolution(monkeypatch):
    report = _unable_report(monkeypatch)
    ens = report["ensemble"]
    assert ens["verdict"] == "INCONCLUSIVE"
    assert "Resolution too low" not in ens["verdict_reason"]
    assert "Insufficient usable forensic evidence" in ens["verdict_reason"]
    # The quality signal is UNABLE, so no pixel measurement exists to
    # support a resolution claim.
    quality = next(s for s in report["signals"] if s["name"] == "quality")
    assert quality["status"] == "unable"
    assert "width" not in quality["technical"]


def test_genuine_small_image_keeps_resolution_reason():
    # A measured 32x32 image: dimensions exist and are below the floor,
    # so the resolution reason is evidence-supported and must stay.
    report = pipeline.run_analysis("tiny.png", "image/png", _png_bytes("solid", 32))
    ens = report["ensemble"]
    quality = next(s for s in report["signals"] if s["name"] == "quality")
    assert quality["technical"].get("width") == 32
    assert ens["verdict"] == "INCONCLUSIVE"
    assert "Resolution too low" in ens["verdict_reason"]


# ---------------------------------------------------------------- S2-3

def test_one_signal_ml_withheld(monkeypatch):
    report = _thin_report(monkeypatch)
    ens = report["ensemble"]
    assert _usable_count(report) == 1
    assert ens["ml_available"] is False
    assert ens["ml_ai_probability"] is None
    assert ens["ml_suppressed_reason"] and "Only 1 of 10" in ens["ml_suppressed_reason"]
    # No ML percentage leaks into the verdict-facing explanations.
    assert "ML detector P(AI)" not in report["explanations"]["technical"]
    # Quality is UNABLE here, yet the reason must not blame resolution.
    assert "Resolution too low" not in ens["verdict_reason"]


def test_majority_rule_boundary(monkeypatch):
    # Majority rule, pinned from both sides via the actual usable count:
    # available exactly when at least half the analyzers are usable.
    names = sorted(_all_names())
    _inject_nan_scores(monkeypatch, set(names[:5]))
    report = pipeline.run_analysis("noise.png", "image/png", _png_bytes("noise"))
    n = _usable_count(report)
    assert report["ensemble"]["ml_available"] is (n * 2 >= len(report["signals"]))


def test_nine_usable_still_available(monkeypatch):
    # Guard for the V5.1.8.64 semantics: near-full coverage stays available.
    _inject_nan_scores(monkeypatch, {"noise"})
    report = pipeline.run_analysis("noise.png", "image/png", _png_bytes("noise"))
    ens = report["ensemble"]
    assert ens["ml_available"] is True
    assert ens["ml_ai_probability"] is not None
    assert ens["ml_suppressed_reason"] is None


# ---------------------------------------------------------------- S2-4 / valid ML

def test_valid_ml_rendering_unchanged():
    root = __import__("os").path.join(
        __import__("os").path.dirname(__import__("os").path.dirname(
            __import__("os").path.abspath(__file__))), "test_data")
    cases = [
        ("authentic_camera/canon_photo.jpg", "image/jpeg", 0.1),
        ("ai_generated/sd_output.png", "image/png", 10.9),
    ]
    for rel, mime, ml in cases:
        with open(__import__("os").path.join(root, rel), "rb") as fh:
            data = fh.read()
        report = pipeline.run_analysis(__import__("os").path.basename(rel), mime, data)
        ens = report["ensemble"]
        assert ens["ml_available"] is True
        assert ens["ml_ai_probability"] == ml
        assert ens["ml_suppressed_reason"] is None
        d = _suppressed_entry(report)
        assert d["available"] is True
        assert not d.get("suppressed")
        csv_text = reporting.to_csv(report)
        rows = list(__import__("csv").reader(io.StringIO(csv_text)))
        cal = [r for r in rows if r[:2] == ["ml_detector", "calibrated_ai_probability"]]
        assert cal and cal[0][2] == str(ml)
        pdf_text = _pdf_flat_text(reporting.to_pdf(report))
        assert f"Calibrated P(AI) {ml}%" in pdf_text
