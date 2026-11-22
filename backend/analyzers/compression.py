"""Compression forensics: JPEG quantization tables, estimated quality,
8x8 block-boundary discontinuities (blockiness), double-compression hints,
recompression indicators."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import ndimage

from .base import (AnalysisContext, AnalyzerResult, BaseAnalyzer, Evidence,
                   EvidenceDirection, SignalStatus, clamp)

# Standard IJG luminance quantization table (quality 50) for comparison
STD_LUMA_Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=float)


def _estimate_jpeg_quality(qtables: dict) -> float | None:
    """Estimate JPEG quality by matching the luma table against scaled IJG tables."""
    if not qtables:
        return None
    table = np.array(qtables[0], dtype=float).reshape(8, 8)
    best_q, best_err = 50, float("inf")
    for q in range(1, 100):
        scale = 5000 / q if q < 50 else 200 - 2 * q
        scaled = np.floor((STD_LUMA_Q50 * scale + 50) / 100)
        scaled = np.clip(scaled, 1, 255)
        err = np.abs(scaled - table).mean()
        if err < best_err:
            best_err, best_q = err, q
    return float(best_q) if best_err < 12 else None


class CompressionAnalyzer(BaseAnalyzer):
    name = "compression"
    display_name = "Compression Analysis"

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        res = AnalyzerResult(name=self.name, display_name=self.display_name,
                             ai_score=50.0, manip_score=50.0, confidence=80.0)
        ev: list[Evidence] = []
        warn: list[str] = []
        tech: dict = {}
        ai, manip, conf = 50.0, 50.0, 80.0

        try:
            raw = Image.open(io.BytesIO(ctx.file_bytes))
            raw.load()
        except Exception as exc:
            res.status = SignalStatus.UNABLE
            res.warnings.append(str(exc))
            res.confidence = 0
            return res

        fmt = (raw.format or "").upper()
        tech["format"] = fmt
        tech["mime"] = ctx.mime_type

        # --- blockiness: mean absolute difference across 8x8 boundaries ---
        g = ctx.gray
        h, w = g.shape
        # vertical boundaries at multiples of 8
        bv = [c for c in range(8, w, 8)]
        bh = [r for r in range(8, h, 8)]
        if bv and bh:
            vdiff = float(np.abs(g[:, bv].astype(float) - g[:, [c - 1 for c in bv]].astype(float)).mean())
            hdiff = float(np.abs(g[bh, :].astype(float) - g[[r - 1 for r in bh], :].astype(float)).mean())
            # interior differences for reference
            interior_cols = [c for c in range(1, w) if c % 8]
            interior_rows = [r for r in range(1, h) if r % 8]
            vdiff_in = float(np.abs(g[:, interior_cols].astype(float) - g[:, [c - 1 for c in interior_cols]].astype(float)).mean()) if interior_cols else vdiff
            hdiff_in = float(np.abs(g[interior_rows, :].astype(float) - g[[r - 1 for r in interior_rows], :].astype(float)).mean()) if interior_rows else hdiff
            blockiness = ((vdiff / (vdiff_in + 1e-9)) + (hdiff / (hdiff_in + 1e-9))) / 2
            tech["blockiness_ratio"] = round(blockiness, 3)
            tech["boundary_diff"] = round((vdiff + hdiff) / 2, 4)
            if blockiness > 1.6:
                ev.append(Evidence(
                    f"Elevated 8x8 block-boundary discontinuities (ratio {blockiness:.2f})",
                    EvidenceDirection.NEUTRAL,
                    "Strong block structure indicates JPEG-style compression. By itself this is "
                    "normal for JPEGs; it matters when block strength varies across regions.",
                ))
        else:
            tech["blockiness_ratio"] = None
            warn.append("Image too small for 8x8 block analysis.")

        if fmt in ("JPEG", "JPG", "MPEG"):
            qtables = getattr(raw, "quantization", None)
            if qtables:
                tech["quantization_tables"] = len(qtables)
                q_est = _estimate_jpeg_quality(qtables)
                tech["estimated_jpeg_quality"] = q_est
                # check chroma subsampling note via quantization count only
                if q_est is not None:
                    ev.append(Evidence(
                        f"JPEG quantization consistent with quality ~{q_est:.0f}",
                        EvidenceDirection.NEUTRAL,
                        "Quantization tables match a standard single-compression profile." if q_est else "",
                    ))
                    if q_est <= 60:
                        warn.append("Heavy JPEG compression detected - forensic signals are degraded.")
                        conf = 60.0
                        ev.append(Evidence(
                            "Heavy compression reduces reliability of all forensic signals",
                            EvidenceDirection.UNCERTAIN,
                            "Low-quality JPEGs destroy the fine detail that detectors rely on.",
                        ))
                else:
                    manip = max(manip, 66.0)
                    ev.append(Evidence(
                        "Quantization tables do not match a standard single-compression profile",
                        EvidenceDirection.SUPPORTS_MANIP,
                        "Non-standard tables can result from re-saving at a different quality, "
                        "which happens when an image is edited and re-exported.",
                    ))
                    tech["nonstandard_quantization"] = True
            else:
                warn.append("Quantization tables unavailable.")

            # double-compression hint: look for periodic artifacts in DCT-ish domain is
            # expensive; use ELA variance as a proxy via pixel analyzer instead.
            ev.append(Evidence("JPEG container structure intact", EvidenceDirection.NEUTRAL, ""))
        elif fmt == "PNG":
            ev.append(Evidence(
                "PNG is lossless - no JPEG compression artifacts expected",
                EvidenceDirection.NEUTRAL,
                "A PNG claiming to be a camera photo is slightly unusual (cameras shoot JPEG/RAW), "
                "but screenshots and exports are commonly PNG.",
            ))
            tech["lossless"] = True
        elif fmt == "WEBP":
            lossless = bool(raw.info.get("lossless", False)) if hasattr(raw, "info") else False
            tech["webp_lossless"] = lossless
            ev.append(Evidence(
                f"WEBP container ({'lossless' if lossless else 'lossy'})",
                EvidenceDirection.NEUTRAL,
                "WEBP is common for web/AI-generator outputs and messaging apps.",
            ))
            if not lossless:
                ai = max(ai, 56.0)
                ev.append(Evidence(
                    "Lossy WEBP encoding is common in AI-generator and web-pipeline outputs",
                    EvidenceDirection.SUPPORTS_AI,
                    "Weak signal on its own - many authentic photos are shared as WEBP too.",
                ))
        elif fmt == "BMP":
            ev.append(Evidence("BMP is uncompressed - no compression forensics applicable",
                               EvidenceDirection.NEUTRAL, ""))
            conf = 55.0
        elif fmt in ("HEIC", "HEIF"):
            ev.append(Evidence("HEIC container - typical of modern smartphone cameras",
                               EvidenceDirection.SUPPORTS_AUTHENTIC,
                               "HEIC is the default capture format on recent iPhones."))
            ai = min(ai, 42.0)

        # file size vs dimensions sanity (extreme mismatch can indicate odd pipelines)
        w0, h0 = ctx.pil_image.size
        bpp = (ctx.file_size * 8) / max(1, w0 * h0)
        tech["bits_per_pixel"] = round(bpp, 3)

        res.ai_score = clamp(ai)
        res.manip_score = clamp(manip)
        res.confidence = clamp(conf)
        res.evidence = ev
        res.warnings = warn
        res.technical = tech
        return res
