"""AI artifact analysis: heuristic indicators of synthetic generation -
unnatural symmetry, over-smooth regions, repeated micro-patterns, abnormal
fine-detail distribution.

These are HEURISTICS, clearly labelled as such. They complement (never
replace) a trained detector model, which plugs in via AIDetectorProvider.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .base import (AnalysisContext, AnalyzerResult, BaseAnalyzer, Evidence,
                   EvidenceDirection, SignalStatus, clamp)


class AIArtifactAnalyzer(BaseAnalyzer):
    name = "ai_artifacts"
    display_name = "AI Artifact Analysis"

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        res = AnalyzerResult(name=self.name, display_name=self.display_name,
                             ai_score=50.0, manip_score=50.0, confidence=60.0,
                             status=SignalStatus.WARNING)
        ev: list[Evidence] = []
        tech: dict = {}
        ai, conf = 50.0, 60.0
        res.warnings.append(
            "Heuristic analysis only - not a trained AI detector. "
            "No trained model is connected; see Detector info."
        )

        g = ctx.small_gray
        h, w = g.shape

        # 1) mirror symmetry: unnaturally perfect bilateral symmetry
        left = g[:, : w // 2]
        right = np.fliplr(g[:, w - w // 2:])
        lh, lw = left.shape
        rh, rw = right.shape
        mh, mw = min(lh, rh), min(lw, rw)
        a, b = left[:mh, :mw].ravel(), right[:mh, :mw].ravel()
        # V5.1.8.62 (confirmed V5.1.8.61 root cause): the old guard
        # `a.std() > 1e-9` measured variance in float32, whose summation
        # noise (~3e-8) on a constant image exceeds the threshold, while
        # np.corrcoef's path yields exactly-zero variance -> 0/0 -> NaN.
        # Constancy is therefore tested exactly (max == min, no summation
        # involved), and the correlation itself is finite-checked. An
        # unmeasurable correlation is recorded as unavailable (None)
        # with a warning - never fabricated as 0.0, which would claim
        # "no correlation" for a measurement that is undefined.
        if (a.size == 0 or b.size == 0
                or a.max() == a.min() or b.max() == b.min()):
            sym = None
        else:
            sym = float(np.corrcoef(a, b)[0, 1])
            if not np.isfinite(sym):
                sym = None
        if sym is None:
            tech["mirror_symmetry_corr"] = None
            res.warnings.append(
                "Mirror symmetry unmeasurable: an image half is constant, "
                "so bilateral correlation is undefined (recorded as "
                "unavailable, not zero)."
            )
            ev.append(Evidence("Mirror symmetry unmeasurable (constant image region)",
                               EvidenceDirection.NEUTRAL, ""))
        elif sym > 0.92:
            tech["mirror_symmetry_corr"] = round(sym, 3)
            ai = max(ai, 72.0)
            ev.append(Evidence(
                f"Near-perfect bilateral symmetry (r={sym:.2f})",
                EvidenceDirection.SUPPORTS_AI,
                "Generators sometimes mirror features; real scenes rarely correlate this strongly.",
            ))
        else:
            tech["mirror_symmetry_corr"] = round(sym, 3)
            ev.append(Evidence("No abnormal global symmetry", EvidenceDirection.NEUTRAL, ""))

        # 2) over-smooth fraction: blocks with almost no detail
        lap = np.abs(ndimage.laplace(g))
        grid = 16
        ys = np.linspace(0, h, grid + 1, dtype=int)
        xs = np.linspace(0, w, grid + 1, dtype=int)
        detail = np.zeros((grid, grid))
        for i in range(grid):
            for j in range(grid):
                detail[i, j] = lap[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean()
        smooth_frac = float((detail < detail.mean() * 0.15).mean())
        tech["oversmooth_block_fraction"] = round(smooth_frac, 3)
        if smooth_frac > 0.35:
            ai = max(ai, 68.0)
            ev.append(Evidence(
                f"{smooth_frac * 100:.0f}% of blocks lack fine detail",
                EvidenceDirection.SUPPORTS_AI,
                "Large detail-free zones (plastic skin, empty backgrounds) are common in synthetic portraits.",
            ))

        # 3) detail distribution skew: natural images spread detail; synthetic
        #    images often concentrate it in a subject with dead background
        d = detail / (detail.max() + 1e-9)
        concentrated = float((d > 0.5).mean())
        tech["high_detail_block_fraction"] = round(concentrated, 3)
        if concentrated < 0.08 and smooth_frac > 0.3:
            ai = max(ai, 64.0)
            ev.append(Evidence(
                "Detail concentrated in few regions with dead surroundings",
                EvidenceDirection.SUPPORTS_AI,
                "Subject-on-blur generative portrait pattern.",
            ))

        # 4) repeated pattern score via block-hash collisions (lightweight)
        bh, bw = 16, 16
        hashes = {}
        collisions = 0
        total = 0
        gs = (g * 255).astype(np.uint8)
        for yy in range(0, h - bh, bh):
            for xx in range(0, w - bw, bw):
                block = gs[yy:yy + bh, xx:xx + bw]
                if block.std() < 4:  # ignore flat blocks - they collide trivially
                    continue
                # coarse hash: 4x4 mean-pooled, quantized
                small = block.reshape(4, 4, 4, 4).mean(axis=(1, 3))
                key = tuple((small / 32).astype(int).ravel().tolist())
                total += 1
                if key in hashes:
                    collisions += 1
                else:
                    hashes[key] = (yy, xx)
        rep_ratio = collisions / max(total, 1)
        tech["repeated_block_ratio"] = round(rep_ratio, 4)
        if rep_ratio > 0.25:
            ai = max(ai, 66.0)
            ev.append(Evidence(
                f"High micro-pattern repetition ({rep_ratio * 100:.1f}% duplicate blocks)",
                EvidenceDirection.SUPPORTS_AI,
                "Tiled/repeated micro-texture is a known diffusion/GAN artifact. Flat real "
                "backgrounds (sky, walls) can also repeat - context matters.",
            ))
        else:
            ev.append(Evidence("No unusual micro-pattern repetition", EvidenceDirection.NEUTRAL, ""))

        # 5) edge coherence: synthetic boundaries can be overly smooth
        sx = ndimage.sobel(g, axis=1)
        sy = ndimage.sobel(g, axis=0)
        mag = np.hypot(sx, sy)
        strong_edges = mag > np.percentile(mag, 97)
        edge_frac = float(strong_edges.mean())
        tech["strong_edge_fraction"] = round(edge_frac, 4)

        if ai <= 52:
            ev.append(Evidence(
                "No strong heuristic AI artifacts found",
                EvidenceDirection.SUPPORTS_AUTHENTIC,
                "Heuristics are weak individually; absence here only slightly favors authentic.",
            ))
            ai = 46.0

        res.ai_score = clamp(ai)
        res.manip_score = 50.0
        res.confidence = clamp(conf)
        res.evidence = ev
        res.technical = tech
        return res
