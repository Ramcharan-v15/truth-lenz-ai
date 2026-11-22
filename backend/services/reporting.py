"""Report generation: JSON / CSV / PDF forensic reports (V2).

PDF structure:
  1. Executive summary (+ verdict)
  2. Key metrics
  3. OBSERVED EVIDENCE (forensic signals)
  4. MODEL INFERENCE (ML detector info: raw vs calibrated, preprocessing)
  5. Metadata & provenance (incl. C2PA)
  6. Heatmaps / visualizations
  7. Evidence graph (text tree)
  8. INTERPRETATION (why this verdict)
  9. Reproducibility
  10. LIMITATIONS
Observed evidence, model inference, interpretation and limitations are
clearly separated sections - never mixed.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import math
from typing import Any, Dict, List

# Stage 1 (H2) null-safety helpers: a missing measurement is rendered as the
# explicit token 'unavailable' — never zero-filled, never formatted with a
# numeric format string (which raises TypeError on None). Valid numbers render
# exactly as before, so numeric reports are unchanged.
UNAVAILABLE = "unavailable"


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _pct(value: Any, digits: int = 1) -> str:
    """0–100 measurement as 'xx.x%' or 'unavailable'."""
    return f"{value:.{digits}f}%" if _is_num(value) else UNAVAILABLE


def _plain(value: Any) -> str:
    """Any measurement value as text, or 'unavailable' when missing."""
    return str(value) if value is not None else UNAVAILABLE


def _csv_val(value: Any) -> Any:
    """CSV cell: missing measurements become the explicit token, not a blank."""
    return UNAVAILABLE if value is None else value


def _pct_raw(value: Any) -> str:
    """Number rendered as-is with a % suffix (the original ML-row format),
    or 'unavailable' when missing (never 'None%')."""
    return f"{value}%" if value is not None else UNAVAILABLE

VERDICT_LABELS = {
    "LIKELY_AUTHENTIC": "Likely Authentic",
    "LIKELY_AI_GENERATED": "Likely AI-Generated",
    "LIKELY_MANIPULATED": "Likely Manipulated",
    "INCONCLUSIVE": "Inconclusive",
}


def to_json(report: Dict[str, Any]) -> str:
    # V5.1.8.62: strict serialization - a report containing NaN/Infinity
    # raises instead of emitting invalid JSON.
    return json.dumps(report, indent=2, allow_nan=False)


def to_csv(report: Dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    ens = report["ensemble"]
    repro = report.get("reproducibility", {})
    w.writerow(["section", "field", "value"])
    w.writerow(["identity", "analysis_id", report["analysis_id"]])
    w.writerow(["identity", "report_hash", report.get("report_hash", "")])
    w.writerow(["identity", "filename", report["filename"]])
    w.writerow(["identity", "analyzed_at", report["analyzed_at"]])
    w.writerow(["identity", "dimensions", f"{report['width']}x{report['height']}"])
    w.writerow(["identity", "image_sha256", repro.get("image_sha256", "")])
    w.writerow(["verdict", "verdict", VERDICT_LABELS.get(ens["verdict"], ens["verdict"])])
    w.writerow(["verdict", "ai_probability", _csv_val(ens["ai_probability"])])
    w.writerow(["verdict", "manipulation_probability", _csv_val(ens["manipulation_probability"])])
    w.writerow(["verdict", "authenticity_probability", _csv_val(ens["authenticity_probability"])])
    w.writerow(["verdict", "confidence", _csv_val(ens["confidence"])])
    w.writerow(["verdict", "confidence_label", ens["confidence_label"]])
    for d in ens.get("detectors", []):
        w.writerow(["ml_detector", "name", d.get("detector_name")])
        w.writerow(["ml_detector", "version", d.get("detector_version")])
        w.writerow(["ml_detector", "model_type", d.get("model_type")])
        w.writerow(["ml_detector", "status", d.get("model_status")])
        w.writerow(["ml_detector", "raw_score", d.get("raw_score")])
        w.writerow(["ml_detector", "calibrated_ai_probability", "withheld" if d.get("suppressed") else _csv_val(d.get("probability_ai"))])
        if d.get("suppressed_reason"):
            w.writerow(["ml_detector", "suppression_reason", d.get("suppressed_reason")])
        w.writerow(["ml_detector", "calibration", d.get("calibration")])
    w.writerow(["signal", "name | ai_score | manip_score | confidence | status", ""])
    for s in report["signals"]:
        w.writerow(["signal", s["display_name"],
                    f"{s['ai_score']} | {s['manip_score']} | {s['confidence']} | {s['status']}"])
    w.writerow(["evidence", "analyzer | direction | text", ""])
    for s in report["signals"]:
        for e in s["evidence"]:
            w.writerow(["evidence", s["display_name"], f"{e['direction']} | {e['text']}"])
    w.writerow(["reproducibility", "app_version", repro.get("app_version", "")])
    w.writerow(["reproducibility", "backend_version", repro.get("backend_version", "")])
    w.writerow(["reproducibility", "config_version", repro.get("config_version", "")])
    w.writerow(["reproducibility", "config_sha256", repro.get("config_sha256", "")])
    return buf.getvalue()


# ---------------------------------------------------------------------------
def _b64_to_rlimage(data_uri: str, width_mm: float):
    """Decode a data-URI PNG into a reportlab Image, or None."""
    from reportlab.lib.units import mm
    from reportlab.platypus import Image as RLImage
    try:
        raw = base64.b64decode(data_uri.split(",", 1)[1])
        buf = io.BytesIO(raw)
        from PIL import Image as PILImage
        with PILImage.open(io.BytesIO(raw)) as im:
            w, h = im.size
        aspect = h / max(w, 1)
        return RLImage(buf, width=width_mm * mm, height=width_mm * aspect * mm)
    except Exception:
        return None


def to_pdf(report: Dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ens = report["ensemble"]
    repro = report.get("reproducibility", {})
    expl = report.get("explanations", {})
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=16 * mm, bottomMargin=16 * mm)
    styles = getSampleStyleSheet()
    story: List = []
    ACCENT = colors.HexColor("#0ea5e9")
    MUTED = colors.HexColor("#64748b")

    def p(text, style="BodyText", size=10, color=colors.black, space=6):
        s = styles[style].__class__(styles[style].name, parent=styles[style])
        s.fontSize, s.leading, s.textColor, s.spaceAfter = size, size * 1.35, color, space
        story.append(Paragraph(text, s))

    def h2(text):
        p(text, style="Heading2", size=13, color=colors.HexColor("#0f172a"), space=6)

    def h3(text):
        p(text, style="Heading3", size=11, color=colors.HexColor("#1e293b"), space=4)

    def table(data, widths, header_color=colors.HexColor("#334155")):
        t = Table(data, colWidths=[w * mm for w in widths])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), header_color),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f1f5f9")]),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

    # ---- header ----
    p("TRUTH LENZ AI", size=20, color=colors.HexColor("#0ea5e9"), space=2)
    p("Hybrid AI-Detection + Digital Image Forensics Platform", size=10, color=MUTED, space=2)
    p("See beyond the pixels.", size=9, color=MUTED, space=8)

    # ---- 1. Case Information (V4, spec §37) ----
    h2("1. Case Information")
    table([
        ["Field", "Value"],
        ["Case number", report.get("case_id", "n/a")],
        ["Analysis ID", report["analysis_id"]],
        ["Report hash (SHA-256)", (report.get("report_hash", "")[:32] + "...")],
        ["Analysis timestamp", report["analyzed_at"]],
        ["Application version", repro.get("app_version", "")],
        ["Backend version", repro.get("backend_version", "")],
        ["Configuration", f"v{repro.get('config_version', '')} (sha256 {repro.get('config_sha256', '')})"],
        ["Models", "; ".join(f"{d.get('name')} v{d.get('version')}" for d in repro.get("detectors", [])) or "none configured"],
        ["Validation status", "NOT YET VALIDATED ON REAL-WORLD DATA (synthetic evaluation only)"],
    ], [60, 100])

    # ---- 2. Image Identity ----
    h2("2. Image Identity")
    table([
        ["Field", "Value"],
        ["Filename", report["filename"]],
        ["SHA-256", repro.get("image_sha256", "n/a")],
        ["Perceptual hash", repro.get("perceptual_hash", "not computed")],
        ["Dimensions", f"{report['width']}x{report['height']}"],
        ["File size", f"{report['file_size'] / 1024:.1f} KB"],
        ["MIME type", report.get("mime_type", "n/a")],
    ], [60, 100])
    img = _b64_to_rlimage(report.get("preview", ""), 60)
    if img:
        story.append(img)
        story.append(Spacer(1, 4))
    p(f"File: <b>{report['filename']}</b> &nbsp; "
      f"Analysis ID: <b>{report['analysis_id']}</b> &nbsp; "
      f"Date: {report['analyzed_at']}", size=9, color=MUTED, space=6)
    verdict_label = VERDICT_LABELS.get(ens["verdict"], ens["verdict"])
    vcolor = {"LIKELY_AI_GENERATED": colors.HexColor("#dc2626"),
              "LIKELY_MANIPULATED": colors.HexColor("#d97706"),
              "LIKELY_AUTHENTIC": colors.HexColor("#16a34a"),
              "INCONCLUSIVE": colors.HexColor("#64748b")}[ens["verdict"]]
    p(f"VERDICT: {verdict_label}", size=16, color=vcolor, space=4)
    p(expl.get("basic", ens.get("verdict_explanation", "")), size=10, space=6)
    p(ens.get("verdict_reason", ""), size=9, color=MUTED, space=8)

    # ---- 2. key metrics ----
    h2("13. Ensemble Decision")
    table([
        ["Metric", "Value"],
        ["AI-generation probability", _pct(ens["ai_probability"], 1)],
        ["Manipulation probability", _pct(ens["manipulation_probability"], 1)],
        ["Authentic-capture probability", _pct(ens["authenticity_probability"], 1)],
        ["Confidence", f"{_pct(ens['confidence'], 1)} ({ens['confidence_label']})"],
        ["Signals agree", "Yes" if ens["signals_agree"] else "No"],
        ["ML detector active", "Yes" if ens.get("ml_available") else "No - forensic signals only"],
    ], [80, 80], header_color=ACCENT)

    # ---- 3. observed evidence ----
    h2("3. Observed Evidence")
    p("Directly measured properties of the image. These are observations, not conclusions.",
      size=9, color=MUTED, space=4)
    sdata = [["Signal", "AI", "Manip.", "Conf.", "Status"]]
    for s in report["signals"]:
        sdata.append([s["display_name"], f"{s['ai_score']:.0f}", f"{s['manip_score']:.0f}",
                      f"{s['confidence']:.0f}%", s["status"]])
    table(sdata, [58, 22, 22, 22, 36])
    h3("Evidence for the verdict")
    for e in ens["strongest_evidence"][:10]:
        p(f"&bull; <b>[{e['analyzer']}]</b> {e['text']}", size=9, space=3)
    if ens["evidence_against"]:
        h3("Evidence against the verdict")
        for e in ens["evidence_against"][:8]:
            p(f"&bull; <b>[{e['analyzer']}]</b> {e['text']}", size=9, space=3)
    if ens["uncertain_factors"]:
        h3("Uncertain factors")
        for e in ens["uncertain_factors"][:6]:
            p(f"&bull; <b>[{e['analyzer']}]</b> {e['text']}", size=9, space=3)

    # ---- 4. model inference ----
    h2("10. ML Inference")
    p("Outputs of trained models, kept separate from forensic observations. "
      "Raw scores are never presented as probabilities without documented calibration.",
      size=9, color=MUTED, space=4)
    dets = ens.get("detectors", [])
    if ens.get("ml_suppressed_reason"):
        # Stage 2 (S2-1): a withheld output is never presented as a
        # calibrated probability, and the model must not be described
        # as unconfigured when it is configured but withheld.
        p("ML probability withheld: " + ens["ml_suppressed_reason"] + " "
          "The trained model is configured, but its output is not presented "
          "as a calibrated probability.", size=10, space=8)
    elif dets and any(d.get("available") for d in dets):
        for d in dets:
            if not d.get("available"):
                continue
            h3(f"{d.get('detector_name')} v{d.get('detector_version')}")
            table([
                ["Field", "Value"],
                ["Backend", d.get("model_type", "")],
                ["Status", d.get("model_status", "")],
                ["Raw model score", str(d.get("raw_score"))],
                ["Calibrated P(AI)", _pct_raw(d.get("probability_ai"))],
                ["Calibration", d.get("calibration", "")],
                ["Confidence", _pct_raw(d.get("confidence"))],
                ["Inference time", f"{d.get('inference_time_ms')} ms"],
                ["Generator label", d.get("generator_label") or "n/a"],
            ], [60, 100])
            prep = d.get("preprocessing", {})
            if prep:
                p("Preprocessing: " + "; ".join(
                    f"{s.get('step')}" for s in prep.get("steps", [])) +
                  f" (input {prep.get('model_input_dimensions')}, "
                  f"normalization {prep.get('normalization')})", size=8, color=MUTED, space=3)
            p("Limitations: " + d.get("limitations", ""), size=8, color=MUTED, space=8)
    else:
        p("No trained ML detector is configured. This analysis used forensic "
          "signals only (model status: MODEL_NOT_CONFIGURED).", size=10, space=8)

    # ---- 5. metadata & provenance ----
    h2("4. Metadata")
    md = next((s for s in report["signals"] if s["name"] == "metadata"), None)
    if md:
        tech = md.get("technical", {})
        table([
            ["Field", "Value"],
            ["C2PA status", str(tech.get("c2pa_status", "missing")).upper()],
            ["C2PA details", "; ".join(f"{k}={v}" for k, v in
                                       tech.get("c2pa_info", {}).items()) or "n/a"],
            ["EXIF entries", str(tech.get("exif_count", tech.get("exif_entries", "n/a")))],
            ["ICC profile", str(tech.get("icc_profile", "n/a"))],
            ["XMP", str(tech.get("xmp", "n/a"))],
            ["Software tags", str(tech.get("software", "n/a"))[:120]],
        ], [60, 100])
        p("Note: missing provenance is never treated as evidence of AI generation.",
          size=8, color=MUTED, space=6)

    # ---- 5. Provenance (V4: separated from metadata) ----
    h2("5. Provenance")
    prov = ens.get("provenance", {})
    table([
        ["Field", "Value"],
        ["Provenance AI score", _plain(prov.get("ai_score"))],
        ["Provenance confidence", _pct_raw(prov.get("confidence"))],
        ["Assessment", str(prov.get("note", "n/a"))[:200]],
    ], [60, 100])
    p("Provenance is evidence about claimed origin, not pixel authenticity. "
      "Valid provenance does not prove pixels are unaltered.",
      size=8, color=MUTED, space=6)

    # ---- 6. heatmaps ----
    h2("6. Pixel Forensics")
    p("Visualizations highlight regions of interest. They do not imply certainty.",
      size=9, color=MUTED, space=4)
    heat_defs = [("pixel", "ela", "ELA (error level analysis)"),
                 ("frequency", "spectrum", "Frequency spectrum"),
                 ("manipulation", "manipulation_overlay", "Manipulation heatmap"),
                 ("statistics", "histogram_rgb", "RGB histogram"),
                 ("noise", "noise_map", "Noise map")]
    shown = 0
    for sig_name, vis_key, label in heat_defs:
        sig = next((s for s in report["signals"] if s["name"] == sig_name), None)
        uri = (sig or {}).get("visuals", {}).get(vis_key, "")
        if uri:
            h3(label)
            img = _b64_to_rlimage(uri, 120)
            if img:
                story.append(img)
                story.append(Spacer(1, 6))
                shown += 1
    if not shown:
        p("No visualizations available for this analysis.", size=9, color=MUTED)

    # ---- 7. evidence graph ----
    h2("3b. Evidence Graph")
    p("How the verdict was assembled. Select nodes in the app for full detail.",
      size=9, color=MUTED, space=4)
    graph_lines = [
        "FINAL VERDICT: " + verdict_label,
        f"  P(AI) {_pct(ens['ai_probability'], 0)} | P(manip) {_pct(ens['manipulation_probability'], 0)} "
        f"| confidence {ens['confidence_label']}",
        "      |",
        "      +-- FORENSICS "
        f"(weight {ens.get('hybrid_weights', {}).get('forensics', 1.0)}): "
        f"P(AI) {_pct_raw(ens.get('forensic_ai_probability'))}",
    ]
    for s in report["signals"][:10]:
        graph_lines.append(f"      |     +-- {s['display_name']}: "
                           f"AI {s['ai_score']:.0f} / manip {s['manip_score']:.0f} "
                           f"({s['confidence']:.0f}% conf)")
    if ens.get("ml_suppressed_reason"):
        graph_lines.append("      +-- ML MODELS: output withheld")
    elif dets and any(d.get("available") for d in dets):
        graph_lines.append(
            f"      +-- ML MODELS (weight {ens.get('hybrid_weights', {}).get('ml', 0)}):")
        for d in dets:
            if d.get("available"):
                graph_lines.append(
                    f"      |     +-- {d.get('detector_name')} v{d.get('detector_version')}: "
                    f"P(AI) {_pct_raw(d.get('probability_ai'))} [{d.get('calibration')}]")
    else:
        graph_lines.append("      +-- ML MODELS: not configured")
    prov = ens.get("provenance", {})
    graph_lines.append(
        f"      +-- PROVENANCE (weight {ens.get('hybrid_weights', {}).get('provenance', 0.6)}): "
        f"{prov.get('note', '')}")
    p("<font face=\"Courier\" size=\"8\">" +
      "<br/>".join(l.replace(" ", "&nbsp;") for l in graph_lines) +
      "</font>", size=8, space=8)

    # ---- 7/8/9. Compression, Frequency/Noise, Manipulation (V4) ----
    def _sig(name):
        return next((s for s in report["signals"] if s["name"] == name), None)

    h2("7. Compression Analysis")
    comp = _sig("compression")
    if comp:
        tech = comp.get("technical", {})
        table([
            ["Metric", "Value"],
            ["AI score", f"{comp['ai_score']:.0f}/100"],
            ["Estimated JPEG quality", str(tech.get("estimated_jpeg_quality", "n/a"))],
            ["Quantization tables", str(tech.get("quant_tables", tech.get("qtable_info", "n/a")))[:80]],
            ["Recompression detected", str(tech.get("recompression", "n/a"))],
        ], [60, 100])

    h2("8. Frequency / Noise Analysis")
    freq, noise = _sig("frequency"), _sig("noise")
    rows = [["Metric", "Value"]]
    if freq:
        ft = freq.get("technical", {})
        rows += [["Spectral slope", str(round(ft.get("spectral_slope", 0), 3))],
                 ["Spectral peaks", str(ft.get("spectral_peak_count", "n/a"))],
                 ["AI score", f"{freq['ai_score']:.0f}/100"]]
    if noise:
        nt = noise.get("technical", {})
        rows += [["Global noise std", str(round(nt.get("global_noise_std", 0), 5))],
                 ["AI score", f"{noise['ai_score']:.0f}/100"]]
    table(rows, [60, 100])

    h2("9. Manipulation Analysis")
    manip = _sig("manipulation")
    if manip:
        table([
            ["Metric", "Value"],
            ["Manipulation score", f"{manip['manip_score']:.0f}/100"],
            ["Copy-move blocks", str(manip.get("technical", {}).get("clone_blocks", "n/a"))],
            ["Max NCC", str(manip.get("technical", {}).get("max_ncc", "n/a"))],
        ], [60, 100])

    # ---- 11. OOD / Uncertainty (V4) ----
    h2("11. OOD / Uncertainty")
    table([
        ["Field", "Value"],
        ["ML OOD status", str(ens.get("ml_ood_status", "UNKNOWN")).replace("_", " ")],
        ["Signal disagreement", _plain(ens.get("signal_disagreement"))],
        ["Signals agree", "Yes" if ens.get("signals_agree") else "No"],
        ["False-positive risk", str((ens.get("false_positive_risk") or {}).get("level", "n/a"))],
    ], [60, 100])
    p("OOD (out-of-distribution) does not mean AI-generated. It means the input "
      "differs from the ML model's training distribution — treat probabilities "
      "with extra caution.", size=8, color=MUTED, space=6)

    # ---- 12. Calibration (V4) ----
    h2("12. Calibration")
    if ens.get("ml_suppressed_reason"):
        p("ML probability withheld: " + ens["ml_suppressed_reason"],
          size=9, color=MUTED)
    elif dets and any(d.get("available") for d in dets):
        for d in dets:
            if d.get("available"):
                table([
                    ["Field", "Value"],
                    ["Model", f"{d.get('detector_name')} v{d.get('detector_version')}"],
                    ["Raw score", str(d.get("raw_score"))],
                    ["Calibrated P(AI)", _pct_raw(d.get("probability_ai"))],
                    ["Calibration method", str(d.get("calibration"))],
                ], [60, 100])
    else:
        p("No ML model active — no calibration to report.", size=9, color=MUTED)
    p("Raw model scores are never presented as probabilities.",
      size=8, color=MUTED, space=6)

    # ---- 8. interpretation ----
    h2("14. Interpretation")
    p("What the evidence means - distinct from the observations above.", size=9,
      color=MUTED, space=4)
    p("<b>Technical:</b> " + expl.get("technical", ""), size=9, space=4)
    for line in ens.get("why", []):
        p("&bull; " + line, size=9, space=3)
    if ens.get("confidence_penalties"):
        h3("Confidence penalties applied")
        for pen in ens["confidence_penalties"]:
            p("&bull; " + pen, size=9, space=3)

    fp = ens.get("false_positive_risk")
    if fp and fp.get("level") != "LOW":
        h3("False-positive risk: " + fp["level"])
        for r in fp.get("reasons", []):
            p("&bull; " + r, size=9, space=3)
        p(fp.get("note", ""), size=8, color=MUTED, space=6)

    unc = ens.get("uncertainty")
    if unc:
        h3(unc["title"])
        for party in unc.get("parties", []):
            p(f"&bull; <b>{party['side']}</b>: {party['ai_probability']}% AI — {party['position']}",
              size=9, space=3)
        p(unc["explanation"], size=9, space=6)

    # ---- 9. reproducibility ----
    h2("16. Reproducibility")
    table([
        ["Field", "Value"],
        ["Analysis ID", repro.get("analysis_id", "")],
        ["Report hash (SHA-256)", (report.get("report_hash", "")[:32] + "...")],
        ["Image SHA-256", (repro.get("image_sha256", "")[:32] + "...")],
        ["App version", repro.get("app_version", "")],
        ["Backend version", repro.get("backend_version", "")],
        ["Config version", repro.get("config_version", "")],
        ["Config SHA-256", repro.get("config_sha256", "")],
        ["Detectors", "; ".join(
            f"{d.get('name')} v{d.get('version')} ({d.get('status')})"
            for d in repro.get("detectors", [])) or "none configured"],
        ["Analyzed at", repro.get("analyzed_at", "")],
    ], [60, 100])
    p("The original image is not stored. Re-running the same image bytes through "
      "the same app/backend/config/detector versions reproduces this report.",
      size=8, color=MUTED, space=6)

    # ---- 10. limitations ----
    h2("15. Limitations")
    p("What this analysis cannot do.", size=9, color=MUTED, space=4)
    for lim in report["limitations"]:
        p("&bull; " + lim, size=8, color=MUTED, space=3)

    story.append(Spacer(1, 12))
    p("Generated by Truth Lenz AI v" + repro.get("app_version", "2.0.0"),
      size=9, color=MUTED, space=2)
    p("Probabilistic forensic analysis - not definitive proof. Do not use as the "
      "sole basis for high-stakes decisions.", size=8, color=MUTED)

    doc.build(story)
    return buf.getvalue()


def validation_report_pdf(experiment: Dict[str, Any]) -> bytes:
    """V5 validation experiment report (§50). Every field from the experiment
    record — nothing fictional. Sections mirror the per-analysis PDF."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    repro = experiment.get("reproducibility", {})
    metrics = experiment.get("metrics", {})
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=16 * mm, bottomMargin=16 * mm)
    styles = getSampleStyleSheet()
    story: List = []
    ACCENT = colors.HexColor("#0ea5e9")
    MUTED = colors.HexColor("#64748b")

    def p(text, style="BodyText", size=10, color=colors.black, space=6):
        s = styles[style].__class__(styles[style].name, parent=styles[style])
        s.fontSize, s.leading, s.textColor, s.spaceAfter = size, size * 1.35, color, space
        story.append(Paragraph(text, s))

    def h2(text):
        p(text, style="Heading2", size=13, color=colors.HexColor("#0f172a"), space=6)

    def table(data, widths, header_color=colors.HexColor("#334155")):
        t = Table(data, colWidths=[w * mm for w in widths])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), header_color),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f1f5f9")]),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

    p("TRUTH LENZ AI", size=20, color=colors.HexColor("#0ea5e9"), space=2)
    p("Validation Experiment Report", size=14, color=colors.HexColor("#0f172a"), space=2)
    p("Real-world scientific evaluation — measured results only.", size=9, color=MUTED, space=8)

    h2("1. Experiment identity")
    table([
        ["Field", "Value"],
        ["Experiment ID", experiment.get("experiment_id", "n/a")],
        ["Status", experiment.get("status", "n/a")],
        ["Dataset ID", repro.get("dataset_id", "n/a")],
        ["Dataset version", repro.get("dataset_version", "n/a")],
        ["Dataset hash", repro.get("dataset_hash", "n/a")],
        ["Model ID", repro.get("model_id", "n/a")],
        ["Model version", repro.get("model_version", "n/a")],
        ["Model hash", repro.get("model_hash", "n/a")],
        ["Configuration hash", repro.get("configuration_hash", repro.get("config_hash", "n/a"))],
        ["Random seed", str(repro.get("random_seed", repro.get("seed", "n/a")))],
        ["Timestamp", repro.get("created_at", repro.get("timestamp", "n/a"))],
        ["Results hash", experiment.get("results_hash", "n/a")],
    ], [60, 100])

    h2("2. Metrics (measured)")
    if metrics:
        mdata = [["Metric", "Value"]]
        for k in ("n", "accuracy", "balanced_accuracy", "precision", "recall",
                  "specificity", "fpr", "fnr", "f1", "roc_auc", "pr_auc",
                  "brier_score", "ece", "inconclusive_rate"):
            if k in metrics:
                v = metrics[k]
                mdata.append([k, f"{v:.4f}" if isinstance(v, float) else str(v)])
        table(mdata, [60, 100], header_color=ACCENT)
        if "confusion_matrix" in metrics:
            p(f"Confusion matrix: {metrics['confusion_matrix']}", size=9, space=4)
        for w in metrics.get("warnings", []):
            p("&bull; " + str(w), size=8, color=MUTED, space=3)
    else:
        p("No metrics recorded for this experiment.", size=9, color=MUTED)

    h2("3. Leakage report")
    leak = experiment.get("leakage_report", {})
    if leak:
        table([["Check", "Result"]] +
              [[k, str(v)[:60]] for k, v in leak.items()], [60, 100])
    else:
        p("No leakage report attached.", size=9, color=MUTED)

    h2("4. Limitations")
    p("Metrics apply only to the evaluated dataset and model version. "
      "Synthetic results do not imply real-world performance. "
      "Small samples cannot justify strong conclusions.", size=9, color=MUTED, space=4)

    h2("5. Reproducibility")
    p("Rerun from the recorded dataset/model/config hashes and seed. "
      f"Results hash: {experiment.get('results_hash', 'n/a')}",
      size=8, color=MUTED, space=6)

    story.append(Spacer(1, 12))
    p("Generated by Truth Lenz AI v5 — validation infrastructure.", size=9, color=MUTED)
    doc.build(story)
    return buf.getvalue()
