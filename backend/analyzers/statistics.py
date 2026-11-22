"""Image statistics: channel means/medians/std, luminance, saturation,
contrast, entropy, edge density, histograms. Mostly descriptive - weak
signal on its own, strong support for charts and quality assessment."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .base import (AnalysisContext, AnalyzerResult, BaseAnalyzer, Evidence,
                   EvidenceDirection, SignalStatus, clamp)


def _entropy(hist: np.ndarray) -> float:
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


class StatisticsAnalyzer(BaseAnalyzer):
    name = "statistics"
    display_name = "Image Statistics"

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        res = AnalyzerResult(name=self.name, display_name=self.display_name,
                             ai_score=50.0, manip_score=50.0, confidence=85.0)
        rgb = ctx.rgb
        gray = ctx.gray
        ev: list[Evidence] = []
        tech: dict = {}

        means = rgb.mean(axis=(0, 1))
        medians = np.median(rgb, axis=(0, 1))
        stds = rgb.std(axis=(0, 1))
        tech["mean_rgb"] = [round(float(v * 255), 1) for v in means]
        tech["median_rgb"] = [round(float(v * 255), 1) for v in medians]
        tech["std_rgb"] = [round(float(v * 255), 1) for v in stds]

        lum = gray
        tech["luminance_mean"] = round(float(lum.mean() * 255), 1)
        tech["luminance_std"] = round(float(lum.std() * 255), 1)
        # contrast: RMS contrast
        tech["contrast_rms"] = round(float(lum.std()), 4)

        # saturation via HSV-ish max-min
        mx = rgb.max(axis=-1)
        mn = rgb.min(axis=-1)
        sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
        tech["saturation_mean"] = round(float(sat.mean()), 4)
        tech["saturation_std"] = round(float(sat.std()), 4)

        # entropy of luminance
        hist_lum, _ = np.histogram(lum, bins=256, range=(0, 1))
        ent = _entropy(hist_lum)
        tech["luminance_entropy_bits"] = round(ent, 2)
        ev.append(Evidence(
            f"Luminance entropy {ent:.2f} bits (max 8.0)",
            EvidenceDirection.NEUTRAL,
            "Entropy measures tonal richness. Very low entropy can indicate flat synthetic "
            "renders or heavy smoothing; typical photos sit between ~6.5 and 7.8 bits.",
        ))
        ai = 50.0
        if ent < 5.0:
            ai = 68.0
            ev.append(Evidence("Unusually low tonal entropy for a natural photograph",
                               EvidenceDirection.SUPPORTS_AI,
                               "Flat, low-detail renders and heavily smoothed images score low here."))
        elif ent > 7.9:
            ev.append(Evidence("Very high entropy - rich detail, typical of noisy natural captures",
                               EvidenceDirection.SUPPORTS_AUTHENTIC, ""))

        # edge density via Sobel on small gray (fast)
        sg = ctx.small_gray
        sx = ndimage.sobel(sg, axis=1)
        sy = ndimage.sobel(sg, axis=0)
        mag = np.hypot(sx, sy)
        edge_density = float((mag > 0.12).mean())
        tech["edge_density"] = round(edge_density, 4)
        ev.append(Evidence(
            f"Edge density {edge_density * 100:.2f}% of pixels",
            EvidenceDirection.NEUTRAL,
            "Fraction of pixels on strong edges. Extremely low values suggest synthetic "
            "smoothness or heavy blur; extremely high values can indicate screenshots/text.",
        ))
        if edge_density < 0.01:
            ai = max(ai, 64.0)
            ev.append(Evidence("Very low edge density - unusually smooth image structure",
                               EvidenceDirection.SUPPORTS_AI, ""))

        # channel balance anomaly: strong single-channel dominance is unusual
        dom = float(means.max() - means.min())
        tech["channel_dominance"] = round(dom * 255, 1)

        # histograms for charts (64 bins to keep payload small)
        hists = {}
        for i, ch in enumerate("rgb"):
            h, edges = np.histogram(rgb[..., i], bins=64, range=(0, 1))
            hists[ch] = {"counts": h.tolist(), "bins": (edges * 255).round(1).tolist()}
        h_lum, e_lum = np.histogram(lum, bins=64, range=(0, 1))
        hists["luminance"] = {"counts": h_lum.tolist(), "bins": (e_lum * 255).round(1).tolist()}
        res.visuals["histograms"] = hists

        res.ai_score = clamp(ai)
        res.manip_score = 50.0
        res.confidence = 85.0
        res.evidence = ev
        res.technical = tech
        return res
