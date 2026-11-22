"""Consistency analysis: internal EXIF/metadata contradictions and
claimed-format-vs-content mismatches.

Contradictions undermine claimed provenance; consistency supports it.
Never claims metadata proves authenticity.
"""
from __future__ import annotations

from .base import (AnalysisContext, AnalyzerResult, BaseAnalyzer, Evidence,
                   EvidenceDirection, SignalStatus, clamp)


class ConsistencyAnalyzer(BaseAnalyzer):
    name = "consistency"
    display_name = "Consistency Check"

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        res = AnalyzerResult(name=self.name, display_name=self.display_name,
                             ai_score=50.0, manip_score=50.0, confidence=70.0)
        ev: list[Evidence] = []
        tech: dict = {}
        ai, manip, conf = 50.0, 50.0, 70.0
        contradictions = 0

        exif = ctx.exif or {}

        # 1) EXIF dimensions vs actual dimensions
        exif_w = exif.get("ExifImageWidth") or exif.get("PixelXDimension")
        exif_h = exif.get("ExifImageHeight") or exif.get("PixelYDimension")
        w, h = ctx.pil_image.size
        try:
            if exif_w and exif_h and (int(exif_w) != w or int(exif_h) != h):
                contradictions += 1
                manip = max(manip, 64.0)
                ev.append(Evidence(
                    f"EXIF claims {exif_w}x{exif_h} but file is {w}x{h}",
                    EvidenceDirection.SUPPORTS_MANIP,
                    "Dimension mismatch suggests resizing or metadata transplant after editing.",
                ))
                tech["dimension_mismatch"] = True
        except (ValueError, TypeError):
            pass

        # 2) software claims camera, but no camera fields (or vice versa)
        software = str(exif.get("Software", "")).lower()
        make = str(exif.get("Make", "")).lower()
        if software and make and "photoshop" not in software and "gimp" not in software:
            # camera make + unknown software: fine
            pass
        if make and not any(k in exif for k in ("Model", "DateTimeOriginal", "ISOSpeedRatings")):
            contradictions += 1
            ev.append(Evidence(
                "Camera make tag present but all capture details missing",
                EvidenceDirection.UNCERTAIN,
                "Could be stripped metadata or a transplanted tag; reduces provenance value.",
            ))
            tech["thin_camera_metadata"] = True
            conf = min(conf, 60.0)

        # 3) timestamps from the future or wildly inconsistent
        for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
            val = str(exif.get(key, ""))
            if val.startswith("20"):
                try:
                    year = int(val[:4])
                    if year > 2026 or year < 1990:
                        contradictions += 1
                        ev.append(Evidence(
                            f"Suspicious timestamp in {key}: {val}",
                            EvidenceDirection.UNCERTAIN,
                            "Implausible dates suggest metadata editing or camera misconfiguration.",
                        ))
                except ValueError:
                    pass

        # 4) format vs content: e.g. PNG with JPEG-style quantization? (can't happen;
        # instead check alpha claims)
        if ctx.mime_type == "image/jpeg" and ctx.extra.get("has_alpha"):
            contradictions += 1
            ev.append(Evidence("JPEG claims transparency metadata - container mismatch",
                               EvidenceDirection.UNCERTAIN, ""))

        # 5) C2PA present but no EXIF at all (unusual combination worth noting)
        if ctx.c2pa_present and not exif:
            ev.append(Evidence(
                "Content Credentials present without any EXIF",
                EvidenceDirection.NEUTRAL,
                "Unusual but possible; credential validation is out of scope for local analysis.",
            ))

        tech["contradiction_count"] = contradictions
        if contradictions == 0:
            if exif:
                ev.append(Evidence(
                    "Internal metadata is self-consistent",
                    EvidenceDirection.SUPPORTS_AUTHENTIC,
                    "No contradictions found between timestamps, dimensions and software tags. "
                    "Consistency supports - but cannot prove - claimed provenance.",
                ))
                ai = min(ai, 47.0)
                manip = min(manip, 47.0)
            else:
                ev.append(Evidence("No metadata to check for consistency",
                                   EvidenceDirection.NEUTRAL, ""))
                conf = 50.0
        elif contradictions == 1:
            manip = max(manip, 60.0)
        else:
            manip = max(manip, 72.0)
            ai = max(ai, 58.0)

        res.ai_score = clamp(ai)
        res.manip_score = clamp(manip)
        res.confidence = clamp(conf)
        res.evidence = ev
        res.technical = tech
        res.status = SignalStatus.WARNING if contradictions else SignalStatus.OK
        return res
