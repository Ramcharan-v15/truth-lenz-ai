"""Shared noise-estimation helper.

Forensically sound approach: estimate noise ONLY in flat (low-gradient)
regions, because texture inflates naive residual statistics. A Laplacian-based
estimator with MAD is robust; flat-region selection removes texture bias.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# std of the 4-neighbour Laplacian of i.i.d. noise with std sigma is sigma*sqrt(20)
_LAPLACIAN_NORM = np.sqrt(20.0)


def flat_region_noise(g: np.ndarray, grid: int = 24,
                      flat_quantile: float = 0.30):
    """Estimate per-block noise over flat regions.

    Returns (noise_map, flat_mask, flat_fraction):
      noise_map  - grid x grid estimated noise std per block
      flat_mask  - bool grid, True where the block is flat enough to trust
      flat_fraction - fraction of blocks deemed flat
    """
    h, w = g.shape
    gx = ndimage.sobel(g, axis=1)
    gy = ndimage.sobel(g, axis=0)
    grad = np.hypot(gx, gy)
    lap = ndimage.laplace(g)

    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    flatness = np.zeros((grid, grid))
    noise = np.zeros((grid, grid))
    for i in range(grid):
        for j in range(grid):
            y0, y1, x0, x1 = ys[i], ys[i + 1], xs[j], xs[j + 1]
            flatness[i, j] = grad[y0:y1, x0:x1].mean()
            lb = lap[y0:y1, x0:x1].ravel()
            med = np.median(lb)
            mad = np.median(np.abs(lb - med))
            noise[i, j] = mad * 1.4826 / _LAPLACIAN_NORM

    thresh = float(np.quantile(flatness, flat_quantile))
    flat_mask = flatness <= max(thresh, 1e-6)
    flat_fraction = float(flat_mask.mean())
    return noise, flat_mask, flat_fraction
