"""Standardized model preprocessing pipeline (Phase 4).

Every transformation applied before model inference is RECORDED - never silent.
The preprocessing record travels with the detector result and appears in
developer diagnostics and reports (reproducibility).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class PreprocessingRecord:
    original_dimensions: Tuple[int, int] = (0, 0)
    original_mode: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    model_input_dimensions: Tuple[int, int] = (0, 0)
    normalization: str = ""
    color_conversion: str = ""

    def log(self, step: str, **params: Any) -> None:
        self.steps.append({"step": step, **params})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_dimensions": list(self.original_dimensions),
            "original_mode": self.original_mode,
            "model_input_dimensions": list(self.model_input_dimensions),
            "normalization": self.normalization,
            "color_conversion": self.color_conversion,
            "steps": self.steps,
        }


class StandardPreprocessor:
    """Deterministic preprocessing with full recording.

    Config keys:
      target_size: (w, h) model input, e.g. (224, 224)
      crop_strategy: 'center' | 'resize_only' | 'letterbox'
      normalization: 'imagenet' | 'zero_one' | 'minus_one_one' | 'none'
      color: 'rgb' (models consume RGB)
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.target_size: Tuple[int, int] = tuple(cfg.get("target_size", (224, 224)))  # type: ignore
        self.crop_strategy: str = cfg.get("crop_strategy", "center")
        self.normalization: str = cfg.get("normalization", "zero_one")
        self.color: str = cfg.get("color", "rgb")

    def describe(self) -> Dict[str, Any]:
        return {
            "target_size": list(self.target_size),
            "crop_strategy": self.crop_strategy,
            "normalization": self.normalization,
            "color": self.color,
        }

    def process(self, pil_image: Image.Image) -> Tuple[np.ndarray, PreprocessingRecord]:
        rec = PreprocessingRecord(
            original_dimensions=pil_image.size,
            original_mode=pil_image.mode,
            normalization=self.normalization,
            color_conversion=f"{pil_image.mode} -> RGB" if pil_image.mode != "RGB" else "none",
        )
        img = pil_image
        if img.mode != "RGB":
            img = img.convert("RGB")
            rec.log("color_convert", source=rec.original_mode, target="RGB")

        w, h = img.size
        tw, th = self.target_size
        if self.crop_strategy == "center":
            # resize so the smaller side matches, then center-crop
            scale = max(tw / w, th / h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            img = img.resize((nw, nh), Image.BILINEAR)
            rec.log("resize", from_size=[w, h], to_size=[nw, nh], method="bilinear")
            left, top = (nw - tw) // 2, (nh - th) // 2
            img = img.crop((left, top, left + tw, top + th))
            rec.log("center_crop", box=[left, top, left + tw, top + th])
        elif self.crop_strategy == "letterbox":
            scale = min(tw / w, th / h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            resized = img.resize((nw, nh), Image.BILINEAR)
            canvas = Image.new("RGB", (tw, th), (0, 0, 0))
            canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
            img = canvas
            rec.log("letterbox", from_size=[w, h], to_size=[tw, th], pad="black")
        else:  # resize_only
            img = img.resize((tw, th), Image.BILINEAR)
            rec.log("resize", from_size=[w, h], to_size=[tw, th], method="bilinear",
                    aspect_preserved=False)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.normalization == "imagenet":
            mean = np.array(self.IMAGENET_MEAN, dtype=np.float32)
            std = np.array(self.IMAGENET_STD, dtype=np.float32)
            arr = (arr - mean) / std
            rec.log("normalize", scheme="imagenet", mean=list(mean), std=list(std))
        elif self.normalization == "minus_one_one":
            arr = arr * 2.0 - 1.0
            rec.log("normalize", scheme="[-1, 1]")
        else:
            rec.log("normalize", scheme="[0, 1]")

        rec.model_input_dimensions = (tw, th)
        # channel-first tensor, the common model input layout
        tensor = np.transpose(arr, (2, 0, 1)).astype(np.float32)
        return tensor, rec
