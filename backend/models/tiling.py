"""Tile-based inference (Phase 5).

For large images, run the detector on overlapping tiles and aggregate.
Produces a spatial confidence map showing WHERE the model is most suspicious.

Only used when the detector opts in (supports_tiling=True).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from PIL import Image

from ..utils.image_io import png_data_uri


def generate_tiles(pil_image: Image.Image, tile_size: int = 512,
                   overlap: float = 0.25) -> List[Dict[str, Any]]:
    """Yield overlapping tiles with their boxes. Preserves aspect ratio
    (no forced resize); tiles at native resolution where possible."""
    w, h = pil_image.size
    stride = max(1, int(tile_size * (1 - overlap)))
    tiles = []
    # cover the image; snap the last tile to the far edge
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    if xs[-1] + tile_size < w:
        xs.append(w - tile_size)
    if ys[-1] + tile_size < h:
        ys.append(h - tile_size)
    for y in ys:
        for x in xs:
            box = (x, y, min(x + tile_size, w), min(y + tile_size, h))
            tiles.append({"box": box, "image": pil_image.crop(box)})
    return tiles


def aggregate_tile_predictions(
    pil_image: Image.Image,
    tiles: List[Dict[str, Any]],
    predict: Callable[[Image.Image], float],
) -> Tuple[float, str, Dict[str, Any]]:
    """Run predict(tile)->P(AI) on each tile. Returns (mean, heatmap_uri, info).

    Aggregation: mean of tile probabilities. The spatial map accumulates
    per-pixel mean probability (overlapping regions averaged).
    """
    w, h = pil_image.size
    acc = np.zeros((h, w), dtype=np.float64)
    cnt = np.zeros((h, w), dtype=np.float64)
    probs: List[float] = []
    for t in tiles:
        p = float(np.clip(predict(t["image"]), 0.0, 1.0))
        probs.append(p)
        x0, y0, x1, y1 = t["box"]
        acc[y0:y1, x0:x1] += p
        cnt[y0:y1, x0:x1] += 1.0
    spatial = acc / np.maximum(cnt, 1e-9)
    mean_p = float(np.mean(probs)) if probs else 0.5

    # render blue->red heatmap
    s = np.clip(spatial, 0, 1)
    cmap = np.stack([
        np.clip(1.6 * s - 0.2, 0, 1),
        np.clip(1.4 * s - 0.55 * np.abs(s - 0.5), 0, 1),
        np.clip(1.2 - 1.6 * s, 0, 1),
    ], axis=-1)
    heat_img = Image.fromarray((cmap * 255).astype(np.uint8))
    heat_img.thumbnail((640, 640), Image.BILINEAR)
    return mean_p, png_data_uri(heat_img), {
        "tile_count": len(tiles),
        "tile_size": tiles[0]["image"].size if tiles else None,
        "tile_probabilities": [round(p, 3) for p in probs],
        "tile_std": round(float(np.std(probs)), 3) if probs else 0.0,
        "max_tile_probability": round(float(np.max(probs)), 3) if probs else 0.0,
    }
