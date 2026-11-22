"""Noise forensics: sensor-noise estimation in flat regions, consistency of
that noise across the frame, channel behavior.

Camera captures show spatially consistent noise in flat areas.
Synthetic images are often implausibly clean or show patterned noise.
"""
from __future__ import annotations

import numpy as np

from .base import (AnalysisContext, AnalyzerResult, BaseAnalyzer, Evidence,
                   EvidenceDirection, SignalStatus, clamp)
from .noise_utils import flat_region_noise


class NoiseAnalyzer(BaseAnalyzer):
    name = "noise"
    display_name = "Noise Analysis"

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        res = AnalyzerResult(name=self.name, display_name=self.display_name,
                             ai_score=50.0, manip_score=50.0, confidence=75.0)
        ev: list[Evidence] = []
        warn: list[str] = []
        tech: dict = {}
        ai, manip, conf = 50.0, 50.0, 75.0

        g = ctx.small_gray
        noise_map, flat_mask, flat_frac = flat_region_noise(g, grid=24)
        tech["flat_region_fraction"] = round(flat_frac, 3)

        if flat_frac < 0.08 or flat_mask.sum() < 8:
            res.status = SignalStatus.WARNING
            res.confidence = 35.0
            warn.append("Too few flat regions for reliable noise estimation.")
            ev.append(Evidence("Insufficient flat areas to estimate sensor noise",
                               EvidenceDirection.UNCERTAIN,
                               "Highly textured images defeat noise estimation; this signal abstains."))
            res.evidence, res.warnings, res.technical = ev, warn, tech
            return res

        flat_noise = noise_map[flat_mask]
        global_noise = float(np.median(flat_noise))
        cv = float(flat_noise.std() / (global_noise + 1e-9))
        consistency = clamp(100 * (1 - min(cv / 0.8, 1.0)))
        tech["noise_std"] = round(global_noise, 5)
        tech["flat_noise_cv"] = round(cv, 3)
        tech["noise_consistency_score"] = round(consistency, 1)

        smooth_frac = float((flat_noise < global_noise * 0.35).mean())
        hot_frac = float((flat_noise > global_noise * 2.5).mean())
        tech["oversmooth_flat_fraction"] = round(smooth_frac, 3)
        tech["hot_noise_flat_fraction"] = round(hot_frac, 3)

        # channel behavior on the full frame (cheap, approximate)
        rgb = ctx.rgb
        ch = [float(np.abs(rgb[..., i] - rgb[..., i].mean()).mean()) for i in range(3)]
        tech["channel_spread"] = round(float((max(ch) - min(ch)) / (sum(ch) / 3 + 1e-9)), 3)

        res.visuals["noise_map"] = {
            "note": "Estimated noise per block over flat regions (brighter = noisier)",
            "grid": (noise_map / (noise_map.max() + 1e-9)).round(3).tolist(),
            "flat_mask": flat_mask.astype(int).tolist(),
        }

        # --- camera-like vs synthetic ---
        camera_points = 0.0
        if 0.0015 <= global_noise <= 0.03:
            camera_points += 35
            ev.append(Evidence(
                f"Noise level within typical camera range (std {global_noise:.4f})",
                EvidenceDirection.SUPPORTS_AUTHENTIC, ""))
            ai = min(ai, 44.0)
        elif global_noise < 0.0008:
            camera_points -= 35
            ai = 76.0
            ev.append(Evidence(
                f"Implausibly low noise (std {global_noise:.5f})",
                EvidenceDirection.SUPPORTS_AI,
                "Even clean studio photos retain measurable sensor noise. Near-zero noise is "
                "characteristic of synthetic renders or aggressive denoising."))
        else:
            ev.append(Evidence(
                f"Elevated noise (std {global_noise:.4f})",
                EvidenceDirection.NEUTRAL,
                "High noise suggests high ISO or heavy compression - not diagnostic of AI."))
            conf = 62.0

        if consistency >= 70:
            camera_points += 30
            ev.append(Evidence(
                f"Noise consistent across flat regions ({consistency:.0f}/100)",
                EvidenceDirection.SUPPORTS_AUTHENTIC,
                "Even noise in flat areas matches single-capture sensor behavior."))
        elif consistency < 40:
            camera_points -= 15
            manip = max(manip, 66.0)
            ev.append(Evidence(
                f"Noise inconsistent across flat regions ({consistency:.0f}/100)",
                EvidenceDirection.SUPPORTS_MANIP,
                "Spliced or locally processed regions often carry different noise."))

        if smooth_frac > 0.25:
            ai = max(ai, 66.0)
            ev.append(Evidence(
                f"{smooth_frac * 100:.0f}% of flat regions are over-smooth",
                EvidenceDirection.SUPPORTS_AI,
                "Patches with almost no noise amid normal ones suggest synthetic smoothing."))

        if hot_frac > 0.20:
            manip = max(manip, 62.0)
            ev.append(Evidence(
                f"{hot_frac * 100:.0f}% of flat regions show hot (elevated) noise",
                EvidenceDirection.SUPPORTS_MANIP,
                "Localized noise spikes can mark pasted or re-compressed regions."))

        camera_like = clamp(50 + camera_points, 2, 98)
        tech["camera_like_noise_pct"] = round(camera_like, 1)
        tech["synthetic_noise_pct"] = round(100 - camera_like, 1)

        res.ai_score = clamp(ai)
        res.manip_score = clamp(manip)
        res.confidence = clamp(conf)
        res.evidence = ev
        res.warnings = warn
        res.technical = tech
        return res
