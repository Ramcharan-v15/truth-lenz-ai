"""Safe image decoding and shared numpy conversions."""
from __future__ import annotations

import io
import json
import os
from typing import Tuple

import numpy as np
from PIL import Image, ImageFile

# Refuse to decode absurd images (decompression-bomb protection on top of PIL's own).
Image.MAX_IMAGE_PIXELS = 178_956_970  # ~178 MP, matches PIL default warning level
ImageFile.LOAD_TRUNCATED_IMAGES = False

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "detector_config.json")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def decode_image(data: bytes, max_dim: int = 2048) -> Tuple[Image.Image, dict]:
    """Safely decode uploaded bytes to an RGB PIL image.

    Returns (image, info) where info carries original dimensions and whether
    the image was downscaled for analysis. Raises ValueError on failure.
    """
    try:
        img = Image.open(io.BytesIO(data))
    except Exception as exc:
        raise ValueError(f"Could not decode image file: {exc}") from exc

    # Fully load now so truncated/corrupt files fail here, not later.
    try:
        img.load()
    except Exception as exc:
        raise ValueError(f"Image file appears corrupted or truncated: {exc}") from exc

    info = {"original_size": img.size, "original_mode": img.mode}
    # Normalize to RGB for analysis (keep alpha note for metadata).
    has_alpha = img.mode in ("RGBA", "LA", "PA")
    info["has_alpha"] = has_alpha
    if img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
        has_alpha = img.mode == "RGBA"
        info["has_alpha"] = has_alpha
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        # Composite over white for analysis; note transparency separately.
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg

    w, h = img.size
    info["downscaled"] = False
    if max(w, h) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        info["downscaled"] = True
        info["analysis_size"] = img.size

    return img, info


def to_float_arrays(pil_image: Image.Image, small_max: int = 512):
    """Return (rgb, gray, small_gray) float32 arrays in 0..1."""
    rgb = np.asarray(pil_image, dtype=np.float32) / 255.0
    # ITU-R BT.601 luma
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    small = pil_image.copy()
    small.thumbnail((small_max, small_max), Image.LANCZOS)
    small_arr = np.asarray(small, dtype=np.float32) / 255.0
    small_gray = 0.299 * small_arr[..., 0] + 0.587 * small_arr[..., 1] + 0.114 * small_arr[..., 2]
    return rgb.astype(np.float32), gray.astype(np.float32), small_gray.astype(np.float32)


def png_data_uri(pil_image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG data URI for frontend display."""
    import base64
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def heatmap_overlay(base: Image.Image, heat: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Blend a 0..1 heat array over base using a blue->red colormap."""
    from PIL import Image as PImage
    h, w = heat.shape
    base_r = base.resize((w, h)).convert("RGB")
    heat = np.clip(heat, 0, 1)
    # simple turbo-like colormap: blue -> cyan -> yellow -> red
    r = np.clip(1.5 * np.abs(heat - 0.75) * -1 + 1.2 * heat, 0, 1)
    cmap = np.stack([
        np.clip(1.6 * heat - 0.2, 0, 1),
        np.clip(1.4 * heat - 0.55 * np.abs(heat - 0.5), 0, 1),
        np.clip(1.2 - 1.6 * heat, 0, 1),
    ], axis=-1)
    cmap_img = PImage.fromarray((cmap * 255).astype(np.uint8)).resize(base_r.size, PImage.BILINEAR)
    return PImage.blend(base_r, cmap_img, alpha)
