"""Frequency-domain forensics: 2D FFT, radial power spectrum, high/low
frequency energy balance, and detection of abnormal spectral peaks.

Generative upsampling (transposed convolutions, diffusion decoders) can leave
periodic fingerprints - sharp peaks in the spectrum that natural images lack.
Absence of peaks proves nothing; new generators are cleaner.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from .base import (AnalysisContext, AnalyzerResult, BaseAnalyzer, Evidence,
                   EvidenceDirection, SignalStatus, clamp)
from ..utils.image_io import png_data_uri


class FrequencyAnalyzer(BaseAnalyzer):
    name = "frequency"
    display_name = "Frequency Analysis"

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        res = AnalyzerResult(name=self.name, display_name=self.display_name,
                             ai_score=50.0, manip_score=50.0, confidence=70.0)
        ev: list[Evidence] = []
        warn: list[str] = []
        tech: dict = {}
        ai, manip, conf = 50.0, 50.0, 70.0

        g = ctx.small_gray  # already <=512px, float32 0..1
        h, w = g.shape
        # window to reduce edge artifacts
        wy = np.hanning(h)[:, None]
        wx = np.hanning(w)[None, :]
        gw = (g - g.mean()) * wy * wx

        F = np.fft.fftshift(np.fft.fft2(gw))
        mag = np.abs(F)
        power = mag ** 2

        # radial profile
        cy, cx = h // 2, w // 2
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_max = r.max()
        nbins = 64
        radial = np.zeros(nbins)
        counts = np.zeros(nbins)
        idx = np.clip((r / r_max * nbins).astype(int), 0, nbins - 1)
        np.add.at(radial, idx, power)
        np.add.at(counts, idx, 1)
        radial = radial / np.maximum(counts, 1)
        # drop the DC bin: it dominates total power and would flatten everything else
        radial_nodc = radial[1:] / (radial[1:].sum() + 1e-12)
        nb = len(radial_nodc)
        low_e = float(radial_nodc[: nb // 3].sum())
        mid_e = float(radial_nodc[nb // 3: 2 * nb // 3].sum())
        high_e = float(radial_nodc[2 * nb // 3:].sum())
        tech["low_freq_energy"] = round(low_e, 4)
        tech["mid_freq_energy"] = round(mid_e, 4)
        tech["high_freq_energy"] = round(high_e, 4)

        # spectral slope: fit log(power) ~ slope * log(freq). Natural images sit
        # around -2..-4; over-smoothed synthetic images fall off much faster.
        freqs = (np.arange(1, nb) + 1) / nb
        pr = radial_nodc[1:]
        mask = pr > 0
        slope = float(np.polyfit(np.log(freqs[mask]), np.log(pr[mask] + 1e-12), 1)[0])
        tech["spectral_slope"] = round(slope, 2)

        ev.append(Evidence(
            f"Spectral slope {slope:.2f} (natural images typically -2 to -4)",
            EvidenceDirection.NEUTRAL,
            "The slope describes how fast fine detail fades. Over-smoothed synthetic images "
            "show much steeper falloff.",
        ))
        if slope < -6.0:
            ai = max(ai, 68.0)
            ev.append(Evidence(
                f"Abnormally steep spectral falloff (slope {slope:.2f})",
                EvidenceDirection.SUPPORTS_AI,
                "Fine detail vanishes far faster than in natural captures - characteristic of "
                "over-smoothed synthetic rendering.",
            ))
        elif -4.5 <= slope <= -2.0:
            ai = min(ai, 45.0)
            ev.append(Evidence(
                "Spectral falloff within the natural range",
                EvidenceDirection.SUPPORTS_AUTHENTIC, ""))
        elif slope > -1.0:
            ev.append(Evidence(
                "Unusually flat spectrum (excess high-frequency content)",
                EvidenceDirection.NEUTRAL,
                "Can indicate heavy noise, oversharpening or certain synthetic textures."))
            conf = 60.0

        # --- spectral peak detection (exclude DC and very-low-freq region) ---
        # Real generative fingerprints are a SMALL number of sharp, isolated peaks.
        # Dozens of "peaks" means the detector is seeing noise, not a fingerprint.
        logmag = np.log1p(mag)
        local_max = ndimage.maximum_filter(logmag, size=7)
        peaks_mask = (logmag == local_max)
        rr = r / r_max
        peaks_mask &= (rr > 0.12) & (rr < 0.95)
        med = float(np.median(logmag[peaks_mask])) if peaks_mask.any() else float(np.median(logmag))
        std = float(logmag[peaks_mask].std()) if peaks_mask.any() else float(logmag.std())
        strong = peaks_mask & (logmag > med + 6.0 * std)
        labeled, n_peaks = ndimage.label(strong)
        tech["spectral_peak_count"] = int(n_peaks)
        tech["spectral_peak_threshold"] = round(med + 6.0 * std, 3)

        if n_peaks > 30:
            conf = 50.0
            ev.append(Evidence(
                f"Spectrum too irregular for reliable peak analysis ({n_peaks} candidate peaks)",
                EvidenceDirection.UNCERTAIN,
                "Excessive candidate peaks indicate detector noise rather than a generative "
                "fingerprint; this signal is discounted.",
            ))
        elif n_peaks >= 6:
            ai = 76.0
            ev.append(Evidence(
                f"{n_peaks} abnormal spectral peaks detected",
                EvidenceDirection.SUPPORTS_AI,
                "Repeated sharp peaks away from DC are a known fingerprint of some generative "
                "upsampling pipelines. Modern generators may not show this.",
            ))
        elif n_peaks >= 3:
            ai = max(ai, 62.0)
            ev.append(Evidence(
                f"{n_peaks} mild spectral peaks detected",
                EvidenceDirection.SUPPORTS_AI,
                "A few peaks can also come from regular textures (fabrics, screens) - interpret with care.",
            ))
        else:
            ev.append(Evidence(
                "No abnormal spectral peaks - spectrum follows a natural falloff",
                EvidenceDirection.SUPPORTS_AUTHENTIC,
                "",
            ))
            ai = min(ai, 45.0)

        # --- visual: log-magnitude spectrum as PNG data URI ---
        try:
            spec = np.log1p(mag)
            spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-12)
            spec_img = Image.fromarray((spec * 255).astype(np.uint8), mode="L")
            spec_img = spec_img.resize((256, 256), Image.BILINEAR)
            res.visuals["spectrum"] = png_data_uri(spec_img.convert("RGB"))
            # radial profile for charting
            res.visuals["radial_profile"] = {
                "frequencies": np.linspace(0, 1, nb).round(3).tolist(),
                "power": radial_nodc.round(6).tolist(),
                "note": "DC component excluded; power normalized over remaining spectrum",
            }
        except Exception as exc:
            warn.append(f"Could not render spectrum image: {exc}")

        res.ai_score = clamp(ai)
        res.manip_score = clamp(manip)
        res.confidence = clamp(conf)
        res.evidence = ev
        res.warnings = warn
        res.technical = tech
        return res
