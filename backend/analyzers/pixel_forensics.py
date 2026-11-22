"""Pixel-level forensics: Error Level Analysis (ELA), texture uniformity,
local structure anomalies.

ELA: re-save the image at a fixed JPEG quality and amplify the difference.
Regions with different compression histories light up differently. ELA
highlights *potentially* suspicious regions - it never proves manipulation.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from .base import (AnalysisContext, AnalyzerResult, BaseAnalyzer, Evidence,
                   EvidenceDirection, SignalStatus, clamp)
from ..utils.image_io import png_data_uri


def compute_ela(pil_rgb: Image.Image, quality: int = 90) -> tuple[Image.Image, np.ndarray]:
    """Return (ELA visualization image, normalized 0..1 heat array)."""
    buf = io.BytesIO()
    pil_rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(pil_rgb, resaved)
    # amplify
    extrema = diff.getextrema()
    max_diff = max(e[1] for e in extrema) or 1
    scale = min(255.0 / max_diff * 12, 40.0)
    ela = ImageEnhance.Brightness(diff).enhance(scale)
    heat = np.asarray(diff.convert("L"), dtype=np.float32) / 255.0
    return ela, heat


class PixelForensicsAnalyzer(BaseAnalyzer):
    name = "pixel"
    display_name = "Pixel Forensics"

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        res = AnalyzerResult(name=self.name, display_name=self.display_name,
                             ai_score=50.0, manip_score=50.0, confidence=70.0)
        ev: list[Evidence] = []
        warn: list[str] = []
        tech: dict = {}
        ai, manip, conf = 50.0, 50.0, 70.0

        pil = ctx.pil_image
        ela_img, heat = compute_ela(pil)
        # downscale ELA visual for payload
        ela_small = ela_img.copy()
        ela_small.thumbnail((640, 640), Image.LANCZOS)
        res.visuals["ela"] = png_data_uri(ela_small)
        res.visuals["ela_heat_grid"] = (heat[::8, ::8]).round(3).tolist()

        ela_mean = float(heat.mean())
        ela_std = float(heat.std())
        tech["ela_mean"] = round(ela_mean, 4)
        tech["ela_std"] = round(ela_std, 4)

        # hotspot fraction: regions far above the frame's own ELA level
        hot = heat > (ela_mean + 2.5 * ela_std)
        hot_frac = float(hot.mean())
        tech["ela_hotspot_fraction"] = round(hot_frac, 4)

        if hot_frac > 0.02:
            manip = 68.0
            ev.append(Evidence(
                f"ELA highlights {hot_frac * 100:.1f}% of the frame as potentially suspicious regions",
                EvidenceDirection.SUPPORTS_MANIP,
                "These regions have a different error response than their surroundings, which can "
                "happen with splicing or local re-saving. High-contrast edges also light up in ELA - "
                "this is a lead for investigation, not proof of editing.",
            ))
        else:
            ev.append(Evidence(
                "ELA response is uniform across the frame",
                EvidenceDirection.SUPPORTS_AUTHENTIC,
                "Uniform error levels are consistent with a single compression history.",
            ))
            manip = min(manip, 45.0)

        # --- texture uniformity: local variance of Laplacian (micro-texture) ---
        from scipy import ndimage
        lap = ndimage.laplace(ctx.small_gray)
        # block-wise texture energy
        gh, gw = lap.shape
        grid = 16
        ys = np.linspace(0, gh, grid + 1, dtype=int)
        xs = np.linspace(0, gw, grid + 1, dtype=int)
        tex = np.zeros((grid, grid))
        for i in range(grid):
            for j in range(grid):
                tex[i, j] = np.abs(lap[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]).mean()
        tex_cv = float(tex.std() / (tex.mean() + 1e-9))
        tech["texture_cv"] = round(tex_cv, 3)
        if tex_cv > 1.4:
            ev.append(Evidence(
                "Micro-texture varies strongly between regions",
                EvidenceDirection.NEUTRAL,
                "Natural scenes vary too; combined with noise/ELA signals this can support a "
                "manipulation hypothesis.",
            ))

        # --- unnatural repetition via autocorrelation of residual ---
        resid = ctx.small_gray - ndimage.gaussian_filter(ctx.small_gray, 1.5)
        ac = np.fft.ifft2(np.abs(np.fft.fft2(resid)) ** 2).real
        ac = np.fft.fftshift(ac)  # zero-lag moves to the center
        ac = ac / (ac.max() + 1e-12)
        h2, w2 = ac.shape
        # exclude the broad central DC lobe (proportional to image size)
        excl = max(16, h2 // 16, w2 // 16)
        cy, cz = h2 // 2, w2 // 2
        ring = ac.copy()
        ring[max(0, cy - excl):cy + excl, max(0, cz - excl):cz + excl] = 0
        peak = float(ring.max())
        tech["autocorr_peak"] = round(peak, 3)
        if peak > 0.35:
            ai = max(ai, 66.0)
            ev.append(Evidence(
                f"Periodic micro-texture detected (autocorrelation peak {peak:.2f})",
                EvidenceDirection.SUPPORTS_AI,
                "Repeating fine structure is common in diffusion/GAN outputs (tiled patterns, "
                "repeated skin pores). Regular real-world textures can also trigger this.",
            ))

        res.visuals["texture_grid"] = (tex / (tex.max() + 1e-9)).round(3).tolist()

        res.ai_score = clamp(ai)
        res.manip_score = clamp(manip)
        res.confidence = clamp(conf)
        res.evidence = ev
        res.warnings = warn
        res.technical = tech
        return res
