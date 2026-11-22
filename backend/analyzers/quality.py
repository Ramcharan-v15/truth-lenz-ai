"""Image quality assessment + screenshot/resize detection.

Quality gates detector reliability: poor quality -> lower confidence.
Screenshot detection matters because screenshots strip camera metadata and
alter forensic signals.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .base import (AnalysisContext, AnalyzerResult, BaseAnalyzer, Evidence,
                   EvidenceDirection, SignalStatus, clamp)

# common screen / device resolutions (w, h)
SCREEN_RESOLUTIONS = {
    (1920, 1080), (1366, 768), (1536, 864), (1440, 900), (2560, 1440),
    (1280, 720), (3840, 2160), (1600, 900), (1024, 768), (1280, 800),
    (1170, 2532), (1284, 2778), (1179, 2556), (1290, 2796),  # iPhones
    (1080, 2400), (1080, 2340), (1440, 3200), (1440, 3088),  # Android
    (2048, 2732), (1668, 2388), (1640, 2360),  # iPads
    (1024, 1024), (512, 512), (768, 768),  # common AI-generator squares
}

AI_COMMON_SIZES = {(1024, 1024), (512, 512), (768, 768), (1024, 1792), (1792, 1024)}


class QualityAnalyzer(BaseAnalyzer):
    name = "quality"
    display_name = "Image Quality Assessment"

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        res = AnalyzerResult(name=self.name, display_name=self.display_name,
                             ai_score=50.0, manip_score=50.0, confidence=90.0)
        ev: list[Evidence] = []
        warn: list[str] = []
        tech: dict = {}
        ai, conf = 50.0, 90.0

        w, h = ctx.pil_image.size
        pixels = w * h
        tech["width"] = w
        tech["height"] = h
        tech["megapixels"] = round(pixels / 1e6, 2)
        tech["aspect_ratio"] = round(w / h, 3)

        # blur: variance of Laplacian on 0-255 scale (threshold ~60 separates sharp/blurry)
        lap_var = float(ndimage.laplace(ctx.gray * 255.0).var())
        tech["blur_laplacian_variance"] = round(lap_var, 2)
        blurry = lap_var < 60

        # resolution tier (based on native pixel count)
        if pixels >= 8_000_000:
            tier, tier_score = "Excellent", 95
        elif pixels >= 2_000_000:
            tier, tier_score = "Good", 80
        elif pixels >= 150_000:
            tier, tier_score = "Fair", 60
        else:
            tier, tier_score = "Poor", 35
        if blurry and tier in ("Excellent", "Good"):
            tier, tier_score = "Fair", 60
            warn.append("Image is blurry despite high resolution.")
        tech["quality_tier"] = tier
        tech["quality_score"] = tier_score
        ev.append(Evidence(
            f"Image quality: {tier} ({w}x{h}, blur metric {lap_var:.0f})",
            EvidenceDirection.NEUTRAL,
            "Quality does not indicate authenticity - it gates how much the other "
            "signals can be trusted. Poor quality lowers overall confidence.",
        ))
        if tier == "Poor":
            conf = 55.0
            warn.append("Very low resolution - forensic evidence is thin.")
        elif blurry:
            conf = 70.0

        # --- screenshot detection ---
        screenshot_signals = 0
        reasons = []
        if (w, h) in SCREEN_RESOLUTIONS or (h, w) in SCREEN_RESOLUTIONS:
            # exclude AI-common squares from counting as screenshots
            if (w, h) not in AI_COMMON_SIZES:
                screenshot_signals += 1
                reasons.append("dimensions match a common screen resolution")
        # UI-like borders: long straight high-contrast horizontal/vertical lines
        # near the frame edges
        g = ctx.small_gray
        sx = np.abs(ndimage.sobel(g, axis=1))
        sy = np.abs(ndimage.sobel(g, axis=0))
        edge = (sx + sy) > 0.35
        eh, ew = edge.shape
        top_band = edge[: eh // 20, :].mean(axis=0)
        bottom_band = edge[-eh // 20:, :].mean(axis=0)
        if (top_band > 0.5).mean() > 0.4 or (bottom_band > 0.5).mean() > 0.4:
            screenshot_signals += 1
            reasons.append("long straight high-contrast edges near frame borders (UI chrome)")
        if not ctx.exif:
            screenshot_signals += 1
            reasons.append("no embedded metadata")
        is_screenshot = screenshot_signals >= 2
        tech["screenshot_likely"] = is_screenshot
        tech["screenshot_signals"] = reasons
        if is_screenshot:
            warn.append("Possible screenshot detected - screenshots strip original camera "
                        "metadata and alter forensic signals.")
            ev.append(Evidence(
                "Possible screenshot detected",
                EvidenceDirection.UNCERTAIN,
                "Evidence: " + "; ".join(reasons) + ". Screenshots inherit the forensic "
                "properties of a screen capture, not the original photo.",
            ))
            conf = min(conf, 65.0)

        # --- AI-common size note ---
        if (w, h) in AI_COMMON_SIZES or (h, w) in AI_COMMON_SIZES:
            ai = max(ai, 58.0)
            ev.append(Evidence(
                f"Dimensions {w}x{h} match common AI-generator output sizes",
                EvidenceDirection.SUPPORTS_AI,
                "Weak signal - these sizes are also used for avatars/thumbnails.",
            ))
            tech["ai_common_size"] = True

        # --- resize likelihood: dimensions not matching typical camera ratios ---
        # (very weak; only reported as context)
        typical_ratios = {4/3, 3/2, 16/9, 1/1}
        ratio = w / h
        if min(abs(ratio - r) for r in typical_ratios) > 0.08 and not is_screenshot:
            tech["unusual_aspect_ratio"] = True
            ev.append(Evidence(
                f"Unusual aspect ratio {ratio:.2f}",
                EvidenceDirection.NEUTRAL,
                "May indicate cropping or resizing; not diagnostic alone.",
            ))

        res.ai_score = clamp(ai)
        res.manip_score = 50.0
        res.confidence = clamp(conf)
        res.evidence = ev
        res.warnings = warn
        res.technical = tech
        res.status = SignalStatus.WARNING if (is_screenshot or tier in ("Poor",)) else SignalStatus.OK
        return res
