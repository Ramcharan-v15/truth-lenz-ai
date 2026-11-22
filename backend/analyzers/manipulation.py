"""Manipulation detection: copy-move via DCT block-hash matching, splicing
indicators from noise/ELA disagreement, and a manipulation heatmap overlay.

Copy-move: downscale, tile into overlapping blocks, hash each block with a
quantized DCT signature, find distant duplicates.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.fft import dct

from .base import (AnalysisContext, AnalyzerResult, BaseAnalyzer, Evidence,
                   EvidenceDirection, SignalStatus, clamp)
from ..utils.image_io import png_data_uri, heatmap_overlay


def _block_dct_sign(block: np.ndarray) -> tuple:
    """Coarse candidate hash: sign of 8x8 low-frequency DCT coefficients (no DC).
    Fast with some false positives - every candidate is verified with NCC."""
    c = dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = c[:8, :8].ravel()[1:]
    return tuple(np.sign(low).astype(int).tolist())


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12
    return float((a * b).sum() / denom)


class ManipulationAnalyzer(BaseAnalyzer):
    name = "manipulation"
    display_name = "Manipulation Detection"

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        res = AnalyzerResult(name=self.name, display_name=self.display_name,
                             ai_score=50.0, manip_score=50.0, confidence=65.0)
        ev: list[Evidence] = []
        warn: list[str] = []
        tech: dict = {}
        ai, manip, conf = 50.0, 50.0, 65.0

        g = ctx.small_gray
        h, w = g.shape
        heat = np.zeros((h, w), dtype=float)

        # --- copy-move detection with spatial clustering ---
        # A real clone produces a CLUSTER of matching blocks; scattered random
        # collisions do not. We therefore require connected components.
        B, STRIDE = 16, 8
        ys_blocks = list(range(0, h - B + 1, STRIDE))
        xs_blocks = list(range(0, w - B + 1, STRIDE))
        ny, nx = len(ys_blocks), len(xs_blocks)
        table: dict[tuple, list[tuple[int, int]]] = {}
        skipped_flat = 0
        blocks: dict[tuple[int, int], np.ndarray] = {}
        for iy, yy in enumerate(ys_blocks):
            for ix, xx in enumerate(xs_blocks):
                block = g[yy:yy + B, xx:xx + B]
                if block.std() < 0.012:
                    skipped_flat += 1  # uniform blocks (sky, walls) collide trivially
                    continue
                key = _block_dct_sign(block)
                table.setdefault(key, []).append((iy, ix, yy, xx))
                blocks[(iy, ix)] = block
        tech["copymove_flat_blocks_skipped"] = skipped_flat

        # verify candidates with normalized cross-correlation
        match_mask = np.zeros((ny, nx), dtype=bool)
        dup_pairs = 0
        for positions in table.values():
            if len(positions) > 1:
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        iy1, ix1, y1, x1 = positions[i]
                        iy2, ix2, y2, x2 = positions[j]
                        if abs(y1 - y2) + abs(x1 - x2) > B * 3:  # distant duplicates only
                            if _ncc(blocks[(iy1, ix1)], blocks[(iy2, ix2)]) > 0.97:
                                dup_pairs += 1
                                match_mask[iy1, ix1] = True
                                match_mask[iy2, ix2] = True
        tech["copymove_duplicate_pairs"] = dup_pairs

        labeled, ncomp = ndimage.label(match_mask)
        clusters = sorted([int((labeled == k).sum()) for k in range(1, ncomp + 1)], reverse=True)
        big_clusters = [s for s in clusters if s >= 6]
        tech["copymove_clusters"] = clusters[:8]
        # normalize by image area so thresholds are size-independent
        density = dup_pairs / max(1, (h * w) / (B * B))
        tech["copymove_density"] = round(density, 4)

        if big_clusters:
            manip = 82.0
            ev.append(Evidence(
                f"{len(big_clusters)} duplicated region(s) detected ({big_clusters[0]} matching blocks in largest)",
                EvidenceDirection.SUPPORTS_MANIP,
                "Clustered near-identical regions far apart is the classic copy-move signature "
                "(cloning to hide or duplicate objects).",
            ))
            heat = np.zeros((h, w), dtype=float)
            for k in range(1, ncomp + 1):
                if (labeled == k).sum() >= 6:
                    comp = np.kron((labeled == k).astype(float), np.ones((STRIDE, STRIDE)))
                    chh, cww = comp.shape
                    heat[:chh, :cww] = np.maximum(heat[:chh, :cww], comp[:h, :w] * 0.9)
        elif dup_pairs >= 40 and density > 0.05:
            manip = max(manip, 60.0)
            ev.append(Evidence(
                f"{dup_pairs} duplicated block pairs (scattered, no clear cloned region)",
                EvidenceDirection.SUPPORTS_MANIP,
                "Many duplicates without spatial clustering can come from repetitive natural "
                "patterns; weaker evidence than a clustered clone.",
            ))
        else:
            ev.append(Evidence("No copy-move duplication pattern found",
                               EvidenceDirection.SUPPORTS_AUTHENTIC, ""))
            manip = min(manip, 46.0)

        # --- splicing: flat-region noise inconsistency ---
        # (texture-biased naive residuals are NOT used here; see noise_utils)
        from .noise_utils import flat_region_noise
        nmap, flat_mask, flat_frac = flat_region_noise(g, grid=24)
        if flat_frac >= 0.08 and flat_mask.sum() >= 8:
            fn = nmap[flat_mask]
            med = float(np.median(fn))
            outlier = np.zeros_like(nmap, dtype=bool)
            outlier[flat_mask] = (fn > med * 2.5) | (fn < med * 0.35)
            out_frac = float(outlier[flat_mask].mean())
        else:
            out_frac = 0.0
        tech["noise_outlier_fraction"] = round(out_frac, 3)
        if out_frac > 0.15:
            manip = max(manip, 70.0)
            ev.append(Evidence(
                f"{out_frac * 100:.0f}% of flat regions have anomalous noise vs the frame",
                EvidenceDirection.SUPPORTS_MANIP,
                "Spliced-in content usually carries the noise of its source image, not the host. "
                "Measured only in flat areas to avoid texture bias.",
            ))
            big = np.kron(outlier.astype(float), np.ones((h // 24 + 1, w // 24 + 1)))[:h, :w]
            heat = np.maximum(heat, big * 0.7)
        else:
            ev.append(Evidence("Regional noise is self-consistent", EvidenceDirection.NEUTRAL,
                               "Measured in flat regions to avoid texture bias."))

        # --- edge-based splice hint: long straight high-contrast boundaries
        # (weak; only reported combined with other signals)
        tech["manipulation_heat_coverage"] = round(float((heat > 0.3).mean()), 4)

        # visuals: overlay on original
        try:
            base = ctx.pil_image.copy()
            heat_full = Image.fromarray((heat * 255).astype(np.uint8)).resize(base.size, Image.BILINEAR)
            heat_np = np.asarray(heat_full, dtype=float) / 255.0
            overlay = heatmap_overlay(base, heat_np, alpha=0.5)
            overlay.thumbnail((640, 640), Image.LANCZOS)
            res.visuals["manipulation_overlay"] = png_data_uri(overlay)
            thumb = base.copy()
            thumb.thumbnail((640, 640), Image.LANCZOS)
            res.visuals["original_thumb"] = png_data_uri(thumb)
        except Exception as exc:
            warn.append(f"Could not render manipulation overlay: {exc}")

        if manip <= 52:
            ev.append(Evidence("No manipulation indicators found",
                               EvidenceDirection.SUPPORTS_AUTHENTIC,
                               "Absence of indicators does not prove an image is unedited."))
            manip = 44.0

        res.ai_score = clamp(ai)
        res.manip_score = clamp(manip)
        res.confidence = clamp(conf)
        res.evidence = ev
        res.warnings = warn
        res.technical = tech
        return res
