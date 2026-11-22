"""Generate a synthetic test dataset (no copyrighted material).

Categories:
  authentic_camera/ - photo-like texture + sensor noise + camera EXIF
  ai_generated/     - images carrying AI-generator metadata signatures
  edited/           - copy-move clones, editing-software tags
  screenshots/      - screen-sized UI-like images without metadata
  compressed/       - heavily compressed images
"""
from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo
from scipy import ndimage

ROOT = os.path.join(os.path.dirname(__file__), "..", "test_data")


def photo_like(seed: int = 42, n: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1, (n, n))
    tex = ndimage.gaussian_filter(white, 2.0)
    tex = (tex - tex.min()) / (np.ptp(tex) + 1e-9)
    rgb = np.stack([tex] * 3, axis=-1) * np.array([0.52, 0.44, 0.36])
    rgb += rng.normal(0, 0.015, (n, n, 3))  # sensor-like noise
    return np.clip(rgb, 0, 1)


def save_jpeg(arr: np.ndarray, path: str, q: int = 92, exif=None):
    im = Image.fromarray((arr * 255).astype("uint8"))
    kw = {"format": "JPEG", "quality": q}
    if exif is not None:
        kw["exif"] = exif
    im.save(path, **kw)


def camera_exif(make="Canon", model="Canon EOS R5"):
    ex = Image.Exif()
    ex[271] = make
    ex[272] = model
    ex[306] = "2024:05:01 10:00:00"
    ex[36867] = "2024:05:01 10:00:00"
    ex[36868] = "2024:05:01 10:00:00"
    ex[33434] = 100
    return ex


def main():
    for d in ("authentic_camera", "ai_generated", "edited", "screenshots", "compressed"):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)

    base = photo_like()

    # authentic_camera
    save_jpeg(base, f"{ROOT}/authentic_camera/canon_photo.jpg", 92, camera_exif())
    save_jpeg(photo_like(7), f"{ROOT}/authentic_camera/nikon_photo.jpg", 90,
              camera_exif("NIKON CORPORATION", "NIKON D850"))

    # ai_generated: SD signature in PNG text chunk
    im = Image.fromarray((base * 255).astype("uint8"))
    info = PngInfo()
    info.add_text("parameters", "Steps: 30, Sampler: DPM++ 2M Karras, Model: stable diffusion xl v1.0")
    info.add_text("Software", "Stable Diffusion WebUI")
    im.save(f"{ROOT}/ai_generated/sd_output.png", pnginfo=info)
    # smooth synthetic jpeg (no signature - heuristic-only case)
    smooth = ndimage.gaussian_filter(base, 2.0)
    save_jpeg(np.clip(smooth, 0, 1), f"{ROOT}/ai_generated/smooth_synth.jpg", 92)

    # edited: copy-move clone
    im3 = Image.fromarray((base * 255).astype("uint8"))
    patch = im3.crop((100, 100, 180, 180))
    im3.paste(patch, (300, 300))
    im3.save(f"{ROOT}/edited/copymove_clone.jpg", format="JPEG", quality=95)
    # editing software tag
    ex = Image.Exif()
    ex[305] = "Adobe Photoshop 24.0 (Windows)"
    save_jpeg(base, f"{ROOT}/edited/photoshop_export.jpg", 92, ex)

    # screenshots: 1920x1080 UI-like, no metadata
    shot = Image.new("RGB", (1920, 1080), (24, 28, 36))
    dr = ImageDraw.Draw(shot)
    dr.rectangle([0, 0, 1920, 64], fill=(45, 52, 64))       # title bar
    dr.rectangle([0, 64, 220, 1080], fill=(32, 38, 48))      # sidebar
    for y in range(140, 1000, 46):
        dr.rectangle([260, y, 1880, y + 26], fill=(52, 60, 74))  # text lines
    shot.save(f"{ROOT}/screenshots/ui_screenshot.png")

    # compressed: heavy jpeg
    save_jpeg(base, f"{ROOT}/compressed/heavy_q30.jpg", 30)
    save_jpeg(base, f"{ROOT}/compressed/medium_q60.jpg", 60)

    print("test dataset written to", ROOT)


if __name__ == "__main__":
    main()
